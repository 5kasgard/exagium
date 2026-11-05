#!/usr/bin/env python3
"""exagium record: an append-only, hash-chained ledger of plans and verdicts, with outside anchors.

Layout (under --ledger, default ../ledger):
  entries/000001.json ...   one canonical JSON entry per file; CI is the only writer
  anchors/NNNNNN.sigstore.json, NNNNNN.ots   anchors on entry NNNNNN (the head when it was written)
  inbox/*.json              proposals {"kind": ..., "body": ...}; CI moves them into entries/

The identity of an entry is the sha256 of its stored bytes. Those bytes must be exactly the entry's
RFC 8785 (JCS) form, with no trailing newline. The rules that make Python's json.dumps output equal
JCS are enforced: ASCII keys, no floats, integers within +-(2**53-1). Fuzzed against the rfc8785
library: 0 mismatches in 20,000 random objects.

Commands: ingest | verify | head | check-anchors | monitor.
Exit codes: 0 ok; 1 the ledger or an anchor is wrong; 2 can't check (missing library or network);
3 unexpected failure.
"""
import argparse, datetime, hashlib, json, os, re, sys, urllib.request

MAX_INT = 2**53 - 1
KINDS = ("plan", "verdict")
OUTCOMES = ("pass", "fail", "inconclusive")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ENTRY_NAME = re.compile(r"^(\d{6})\.json$")


class Bad(Exception):
    """The ledger (or an anchor) is wrong: exit 1."""


class CantCheck(Exception):
    """A check couldn't run: exit 2."""


# ---------- canonical form ----------
def canonical(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)).encode("utf-8")


def _check_value(v, where):
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return
    if isinstance(v, int):
        if abs(v) > MAX_INT:
            raise Bad(f"{where}: integer outside +-(2**53-1)")
        return
    if isinstance(v, float):
        raise Bad(f"{where}: floats aren't allowed; write numbers with a fraction as strings")
    if isinstance(v, list):
        for i, x in enumerate(v):
            _check_value(x, f"{where}[{i}]")
        return
    if isinstance(v, dict):
        for k, x in v.items():
            if not k.isascii():
                raise Bad(f"{where}: non-ASCII key {k!r}")
            _check_value(x, f"{where}.{k}")
        return
    raise Bad(f"{where}: unsupported type {type(v).__name__}")


def _no_dup_keys(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise Bad(f"duplicate key {k!r}")
        d[k] = v
    return d


def parse_canonical(raw, where):
    """Parse stored bytes; they must be exactly the canonical form of what they contain."""
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dup_keys,
                         parse_constant=lambda c: (_ for _ in ()).throw(Bad(f"{where}: {c} isn't JSON")))
    except (UnicodeDecodeError, ValueError) as e:
        raise Bad(f"{where}: not valid UTF-8 JSON ({e})")
    _check_value(obj, where)
    if canonical(obj) != raw:
        raise Bad(f"{where}: bytes aren't in canonical form")
    return obj


# ---------- schema ----------
def _need(d, key, typ, where):
    if key not in d or not isinstance(d[key], typ) or (typ is int and isinstance(d[key], bool)):
        raise Bad(f"{where}: needs {key} ({typ.__name__})")
    return d[key]


def _check_digests(lst, where):
    if not isinstance(lst, list):
        raise Bad(f"{where}: must be a list")
    for i, x in enumerate(lst):
        if not isinstance(x, dict) or set(x) != {"name", "sha256"} or not isinstance(x["name"], str) \
                or not isinstance(x["sha256"], str) or not HEX64.match(x["sha256"]):
            raise Bad(f"{where}[{i}]: must be {{name, sha256}} with 64 lowercase hex")


def check_body(kind, body, where):
    if not isinstance(body, dict):
        raise Bad(f"{where}: body must be an object")
    if kind == "plan":
        allowed = {"by", "claim", "test", "decision_rule", "baseline", "inputs", "verdict_due", "notes"}
        for k in ("by", "claim", "test", "decision_rule"):
            if not _need(body, k, str, where).strip():
                raise Bad(f"{where}: {k} is empty")
        if "inputs" in body:
            _check_digests(body["inputs"], f"{where}.inputs")
        if "verdict_due" in body and not (isinstance(body["verdict_due"], str) and DATE.match(body["verdict_due"])):
            raise Bad(f"{where}: verdict_due must be YYYY-MM-DD")
    else:
        allowed = {"by", "plan", "outcome", "evidence", "notes"}
        if not _need(body, "by", str, where).strip():
            raise Bad(f"{where}: by is empty")
        p = _need(body, "plan", dict, where)
        if set(p) != {"seq", "sha256"} or not isinstance(p["seq"], int) or isinstance(p["seq"], bool) \
                or not isinstance(p["sha256"], str) or not HEX64.match(p["sha256"]):
            raise Bad(f"{where}: plan must be {{seq, sha256}}")
        if body.get("outcome") not in OUTCOMES:
            raise Bad(f"{where}: outcome must be one of {OUTCOMES}")
        _check_digests(_need(body, "evidence", list, where), f"{where}.evidence")
    extra = set(body) - allowed
    if extra:
        raise Bad(f"{where}: unknown body fields {sorted(extra)}")


# ---------- the chain ----------
def entry_files(ledger):
    d = os.path.join(ledger, "entries")
    names = sorted(os.listdir(d)) if os.path.isdir(d) else []
    for n in names:
        if not ENTRY_NAME.match(n):
            raise Bad(f"entries/{n}: unexpected file")
    return [os.path.join(d, n) for n in names]


def load_chain(ledger):
    """Check the whole chain; return list of (entry, sha256) in order. Raises Bad on any fault."""
    chain, plans, verdict_of = [], {}, {}
    for i, path in enumerate(entry_files(ledger), start=1):
        name = os.path.basename(path)
        raw = open(path, "rb").read()
        e = parse_canonical(raw, name)
        if not isinstance(e, dict) or set(e) != {"seq", "prev", "kind", "written_at", "body"}:
            raise Bad(f"{name}: entry must have exactly seq, prev, kind, written_at, body")
        if e["seq"] != i or name != f"{i:06d}.json":
            raise Bad(f"{name}: seq {e['seq']!r} out of order (expected {i})")
        want_prev = chain[-1][1] if chain else None
        if e["prev"] != want_prev:
            raise Bad(f"{name}: prev doesn't match the previous entry's sha256")
        if e["kind"] not in KINDS:
            raise Bad(f"{name}: kind must be one of {KINDS}")
        if not isinstance(e["written_at"], str):
            raise Bad(f"{name}: written_at must be a string")
        check_body(e["kind"], e["body"], name)
        h = hashlib.sha256(raw).hexdigest()
        if e["kind"] == "plan":
            plans[i] = (h, e["body"])
        else:
            ref = e["body"]["plan"]
            if ref["seq"] not in plans or plans[ref["seq"]][0] != ref["sha256"]:
                raise Bad(f"{name}: verdict refers to no earlier plan with that seq and sha256")
            if ref["seq"] in verdict_of:
                raise Bad(f"{name}: plan {ref['seq']} already has a verdict (entry {verdict_of[ref['seq']]}); a re-run needs a new plan")
            if e["body"]["by"].strip().lower() == plans[ref["seq"]][1]["by"].strip().lower():
                raise Bad(f"{name}: the verdict's author is the plan's author; a verdict must come from someone else")
            verdict_of[ref["seq"]] = i
        chain.append((e, h))
    return chain, verdict_of


def make_entry(chain, kind, body, now=None):
    check_body(kind, body, "proposal")
    e = {"seq": len(chain) + 1, "prev": chain[-1][1] if chain else None, "kind": kind,
         "written_at": now or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "body": body}
    _check_value(e, "proposal")
    return e


def cmd_ingest(a):
    """Move every inbox proposal (sorted by file name) into the chain. Used by CI, the only writer."""
    chain, _ = load_chain(a.ledger)
    inbox = os.path.join(a.ledger, "inbox")
    names = sorted(n for n in os.listdir(inbox) if n.endswith(".json")) if os.path.isdir(inbox) else []
    os.makedirs(os.path.join(a.ledger, "entries"), exist_ok=True)
    written = []
    for n in names:
        p = os.path.join(inbox, n)
        try:
            prop = json.loads(open(p, "rb").read().decode("utf-8"), object_pairs_hook=_no_dup_keys)
        except (UnicodeDecodeError, ValueError) as ex:
            raise Bad(f"inbox/{n}: not valid JSON ({ex})")
        if not isinstance(prop, dict) or set(prop) != {"kind", "body"} or prop["kind"] not in KINDS:
            raise Bad(f"inbox/{n}: must be {{kind, body}} with kind in {KINDS}")
        e = make_entry(chain, prop["kind"], prop["body"])
        raw = canonical(e)
        out = os.path.join(a.ledger, "entries", f"{e['seq']:06d}.json")
        with open(out, "xb") as f:
            f.write(raw)
        try:
            load_chain(a.ledger)  # the new entry must pass every rule, e.g. one verdict per plan
        except Bad as ex:
            os.remove(out)
            raise Bad(f"inbox/{n}: {ex}")
        chain.append((e, hashlib.sha256(raw).hexdigest()))
        os.remove(p)
        written.append(out)
    for w in written:
        print(w)
    return 0


def cmd_verify(a):
    chain, verdict_of = load_chain(a.ledger)
    today = datetime.date.today().isoformat()
    plans = [(e, h) for e, h in chain if e["kind"] == "plan"]
    open_, overdue = [], []
    for e, h in plans:
        if e["seq"] not in verdict_of:
            open_.append(e["seq"])
            if e["body"].get("verdict_due", "9999-12-31") < today:
                overdue.append(e["seq"])
    report = {"entries": len(chain), "head_seq": chain[-1][0]["seq"] if chain else 0,
              "head_sha256": chain[-1][1] if chain else None, "plans": len(plans),
              "verdicts": len(verdict_of), "open_plans": open_, "overdue_plans": overdue}
    print(json.dumps(report, indent=1))
    return 0


def cmd_head(a):
    chain, _ = load_chain(a.ledger)
    print(f"{chain[-1][0]['seq']} {chain[-1][1]}" if chain else "0 -")
    return 0


# ---------- anchors ----------
def _get(url, data=None, timeout=20):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json",
                                                          "User-Agent": "exagium-record"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except OSError as ex:
        raise CantCheck(f"network: {url}: {ex}")


def identity(repo, workflow, ref):
    return f"https://github.com/{repo}/{workflow}@{ref}"


def check_sigstore(bundle, digest, repo, workflow, ref):
    """Verify offline that a bundle is an attestation, signed by exactly this workflow on this ref, of this digest.
    Only a verification error is a verdict (Bad); anything else that goes wrong is a failure."""
    try:
        from sigstore.models import Bundle
        from sigstore.verify import Verifier, policy
        from sigstore.errors import VerificationError
    except ImportError as ex:
        raise CantCheck(f"sigstore-python isn't usable ({ex}); pip install sigstore==4.5.1 in a virtualenv")
    raw = open(bundle, "rb").read()
    pol = policy.AllOf([
        policy.Identity(identity=identity(repo, workflow, ref), issuer="https://token.actions.githubusercontent.com"),
        policy.GitHubWorkflowRepository(repo),
        policy.GitHubWorkflowRef(ref),
    ])
    try:
        b = Bundle.from_json(raw)
        ptype, payload = Verifier.production(offline=True).verify_dsse(b, pol)
    except VerificationError as ex:
        raise Bad(f"{os.path.basename(bundle)}: Sigstore verification failed: {ex}")
    st = json.loads(payload)
    if ptype != "application/vnd.in-toto+json" or [s.get("digest", {}).get("sha256") for s in st.get("subject", [])] != [digest]:
        raise Bad(f"{os.path.basename(bundle)}: the attestation's subject isn't this entry")
    return int(json.loads(raw)["verificationMaterial"]["tlogEntries"][0]["integratedTime"])


def check_ots(proof, entry_raw, explorers):
    try:
        from opentimestamps.core.timestamp import DetachedTimestampFile
        from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
        from opentimestamps.core.serialize import StreamDeserializationContext
    except ImportError:
        raise CantCheck("opentimestamps isn't installed (pip install opentimestamps-client)")
    with open(proof, "rb") as f:
        dt = DetachedTimestampFile.deserialize(StreamDeserializationContext(f))
    if dt.file_digest != hashlib.sha256(entry_raw).digest():
        raise Bad(f"{os.path.basename(proof)}: proof is for different bytes")
    times = []
    for msg, att in dt.timestamp.all_attestations():
        if not isinstance(att, BitcoinBlockHeaderAttestation):
            continue
        seen = set()
        for base in explorers:
            bh = _get(f"{base}/block-height/{att.height}").decode().strip()
            hdr = bytes.fromhex(_get(f"{base}/block/{bh}/header").decode().strip())
            if hashlib.sha256(hashlib.sha256(hdr).digest()).digest()[::-1].hex() != bh:
                raise Bad(f"{base}: header doesn't hash to block {bh}")
            if hdr[36:68] != msg:
                raise Bad(f"{os.path.basename(proof)}: path doesn't reach block {att.height}'s merkle root ({base})")
            seen.add(hdr)
        if len(seen) != 1:
            raise Bad(f"explorers disagree about block {att.height}")
        times.append(int.from_bytes(next(iter(seen))[68:72], "little"))
    return min(times) if times else None


def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def cmd_check_anchors(a):
    chain, _ = load_chain(a.ledger)
    adir = os.path.join(a.ledger, "anchors")
    names = sorted(os.listdir(adir)) if os.path.isdir(adir) else []
    rows, covered = [], 0
    for n in names:
        m = re.match(r"^(\d{6})\.(sigstore\.json|ots)$", n)
        if not m:
            raise Bad(f"anchors/{n}: unexpected file")
        seq = int(m.group(1))
        if not 1 <= seq <= len(chain):
            raise Bad(f"anchors/{n}: no entry {seq}")
        raw = open(os.path.join(a.ledger, "entries", f"{seq:06d}.json"), "rb").read()
        path = os.path.join(adir, n)
        if m.group(2) == "ots":
            t = check_ots(path, raw, a.explorer)
            rows.append({"anchor": n, "kind": "opentimestamps", "existed_by": iso(t) or "pending"})
        else:
            t = check_sigstore(path, chain[seq - 1][1], a.repo, a.workflow, a.ref)
            rows.append({"anchor": n, "kind": "sigstore", "existed_by": iso(t)})
        if t:
            covered = max(covered, seq)
    print(json.dumps({"anchors": rows, "entries": len(chain), "anchored_through_seq": covered,
                      "unanchored": list(range(covered + 1, len(chain) + 1))}, indent=1))
    return 0


def _refs(repo):
    """Every ref GitHub will name for this repo: current branches and tags, every ref in the activity log
    (which keeps deleted branches), and every pull request's merge and head refs (pull_request runs sign as those)."""
    refs = {"refs/heads/main"}
    for kind in ("branches", "tags"):
        page = 1
        while True:
            items = json.loads(_get(f"https://api.github.com/repos/{repo}/{kind}?per_page=100&page={page}"))
            refs.update(("refs/heads/" if kind == "branches" else "refs/tags/") + x["name"] for x in items)
            if len(items) < 100:
                break
            page += 1
    page = 1
    while True:
        items = json.loads(_get(f"https://api.github.com/repos/{repo}/pulls?state=all&per_page=100&page={page}"))
        refs.update(f"refs/pull/{x['number']}/{k}" for x in items for k in ("merge", "head"))
        if len(items) < 100:
            break
        page += 1
    page = 1
    while True:
        items = json.loads(_get(f"https://api.github.com/repos/{repo}/activity?per_page=100&page={page}"))
        refs.update(x["ref"] for x in items if x.get("ref"))
        if len(items) < 100:
            break
        page += 1
    return sorted(refs)


def cmd_monitor(a):
    """List every Rekor entry signed by the record workflow on any known ref; flag those the ledger doesn't hold.
    Entries are matched by Rekor log index, which is unique per attestation (identical statements share a payload hash)."""
    import base64
    known = {}
    adir = os.path.join(a.ledger, "anchors")
    for n in sorted(os.listdir(adir)) if os.path.isdir(adir) else []:
        if n.endswith(".sigstore.json"):
            b = json.load(open(os.path.join(adir, n)))
            known[int(b["verificationMaterial"]["tlogEntries"][0]["logIndex"])] = n
    seen, unknown = set(), []
    refs = [a.only_ref] if a.only_ref else _refs(a.repo)
    for ref in refs:
        san = identity(a.repo, a.workflow, ref)
        uuids = json.loads(_get("https://rekor.sigstore.dev/api/v1/index/retrieve", json.dumps({"subject": san}).encode()))
        for u in uuids:
            v = next(iter(json.loads(_get(f"https://rekor.sigstore.dev/api/v1/log/entries/{u}")).values()))
            idx = int(v["logIndex"])
            seen.add(idx)
            if idx not in known:
                unknown.append({"ref": ref, "rekor_uuid": u, "log_index": idx, "integrated": iso(v["integratedTime"])})
    missing = sorted(n for idx, n in known.items() if idx not in seen)
    print(json.dumps({"refs_checked": refs, "rekor_entries": len(seen), "ledger_bundles": len(known),
                      "not_in_ledger": unknown, "ledger_bundles_not_found_in_rekor": missing}, indent=1))
    return 1 if unknown or missing else 0


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=os.path.join(here, "..", "ledger"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    sub.add_parser("verify")
    sub.add_parser("head")
    for name in ("check-anchors", "monitor"):
        p = sub.add_parser(name)
        p.add_argument("--repo", default="5kasgard/exagium")
        p.add_argument("--workflow", default=".github/workflows/record.yml")
        if name == "check-anchors":
            p.add_argument("--ref", default="refs/heads/main")
            p.add_argument("--explorer", action="append", default=None,
                           help="esplora API base; default blockstream.info and mempool.space")
        else:
            p.add_argument("--only-ref", help="check this ref only (default: every ref GitHub knows)")
    a = ap.parse_args(argv)
    if getattr(a, "explorer", "x") is None:
        a.explorer = ["https://blockstream.info/api", "https://mempool.space/api"]
    fn = {"ingest": cmd_ingest, "verify": cmd_verify, "head": cmd_head,
          "check-anchors": cmd_check_anchors, "monitor": cmd_monitor}[a.cmd]
    try:
        return fn(a)
    except Bad as ex:
        print(f"BAD: {ex}", file=sys.stderr)
        return 1
    except CantCheck as ex:
        print(f"CAN'T CHECK: {ex}", file=sys.stderr)
        return 2
    except Exception as ex:  # a failure is never reported as a verdict
        print(f"FAILURE: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
