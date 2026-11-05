#!/usr/bin/env python3
"""exagium record: an append-only, hash-chained ledger of plans and verdicts, with outside anchors.

Layout (under --ledger, default ../ledger):
  entries/000001.json ...   one entry per file; CI (.github/workflows/record.yml) is the only writer
  anchors/NNNNNN.sigstore.json   Sigstore attestation by the writer of the range of entries it added, ending at NNNNNN
  anchors/NNNNNN.ots             OpenTimestamps proof of entry NNNNNN's bytes
  inbox/*.json                   proposals {"kind": ..., "body": ...}; CI moves them into entries/

An entry's identity is the sha256 of its stored bytes, which must be exactly its RFC 8785 (JCS) form
(no trailing newline). The rules that make Python's json.dumps output equal JCS are enforced: ASCII
keys, no floats, integers within +-(2**53-1), no lone surrogates.

Commands: ingest | verify | head | range | check-anchors | monitor.
Exit codes: 0 ok; 1 the ledger, an anchor or the monitor's finding is wrong; 2 can't check (missing
library or network); 3 unexpected failure. A failure is never reported as a verdict.
"""
import argparse, base64, datetime, hashlib, json, os, re, subprocess, sys, urllib.error, urllib.request

MAX_INT = 2**53 - 1
KINDS = ("plan", "verdict", "orphan")
OUTCOMES = ("pass", "fail", "inconclusive")
HEX64 = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")  # who: a plain ASCII handle, so "same author" can't hide behind look-alikes
ENTRY_NAME = re.compile(r"([0-9]{6})\.json")
ANCHOR_NAME = re.compile(r"([0-9]{6})\.(sigstore\.json|ots)")
ISSUER = "https://token.actions.githubusercontent.com"


class Bad(Exception):
    """The data is wrong: exit 1."""


class CantCheck(Exception):
    """A check couldn't run (missing library, network): exit 2."""


# ---------- canonical form ----------
def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _check_value(v, where):
    if isinstance(v, bool) or v is None:
        return
    if isinstance(v, str):
        try:
            v.encode("utf-8")
        except UnicodeEncodeError:
            raise Bad(f"{where}: string holds a lone surrogate")
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


def _no_constants(c):
    raise Bad(f"{c} isn't JSON")


def parse_json(raw, where):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dup_keys, parse_constant=_no_constants)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise Bad(f"{where}: not valid UTF-8 JSON ({e})")


def parse_canonical(raw, where):
    """Parse stored bytes; they must be exactly the canonical form of what they contain."""
    obj = parse_json(raw, where)
    _check_value(obj, where)
    if canonical(obj) != raw:
        raise Bad(f"{where}: bytes aren't in canonical form")
    return obj


# ---------- schema ----------
def _need(d, key, typ, where):
    if key not in d or not isinstance(d[key], typ) or (typ is int and isinstance(d[key], bool)):
        raise Bad(f"{where}: needs {key} ({typ.__name__})")
    return d[key]


def _name(d, where):
    v = _need(d, "by", str, where)
    if not NAME.fullmatch(v):
        raise Bad(f"{where}: by must be a lowercase ASCII handle matching {NAME.pattern}")
    return v


def _check_digests(lst, where):
    if not isinstance(lst, list):
        raise Bad(f"{where}: must be a list")
    for i, x in enumerate(lst):
        if not isinstance(x, dict) or set(x) != {"name", "sha256"} or not isinstance(x["name"], str) \
                or not isinstance(x["sha256"], str) or not HEX64.fullmatch(x["sha256"]):
            raise Bad(f"{where}[{i}]: must be {{name, sha256}} with 64 lowercase hex")


def check_body(kind, body, where):
    if not isinstance(body, dict):
        raise Bad(f"{where}: body must be an object")
    _name(body, where)
    if kind == "plan":
        allowed = {"by", "claim", "test", "decision_rule", "baseline", "inputs", "verdict_due", "notes"}
        for k in ("claim", "test", "decision_rule"):
            if not _need(body, k, str, where).strip():
                raise Bad(f"{where}: {k} is empty")
        if "inputs" in body:
            _check_digests(body["inputs"], f"{where}.inputs")
        if "verdict_due" in body:
            try:
                ok = isinstance(body["verdict_due"], str) and len(body["verdict_due"]) == 10 and \
                    datetime.date.fromisoformat(body["verdict_due"]).isoformat() == body["verdict_due"]
            except ValueError:
                ok = False
            if not ok:
                raise Bad(f"{where}: verdict_due must be a real date, YYYY-MM-DD")
    elif kind == "verdict":
        allowed = {"by", "plan", "outcome", "evidence", "notes"}
        p = _need(body, "plan", dict, where)
        if set(p) != {"seq", "sha256"} or not isinstance(p["seq"], int) or isinstance(p["seq"], bool) \
                or not isinstance(p["sha256"], str) or not HEX64.fullmatch(p["sha256"]):
            raise Bad(f"{where}: plan must be {{seq, sha256}}")
        if body.get("outcome") not in OUTCOMES:
            raise Bad(f"{where}: outcome must be one of {OUTCOMES}")
        _check_digests(_need(body, "evidence", list, where), f"{where}.evidence")
    else:  # orphan: explains a Rekor attestation by the writer that the ledger doesn't hold
        allowed = {"by", "log_index", "reason", "statement_b64", "entries_b64"}
        if _need(body, "log_index", int, where) < 0 or not _need(body, "reason", str, where).strip():
            raise Bad(f"{where}: orphan needs log_index >= 0 and a reason")
        try:  # the orphaned attestation's exact signed statement; monitor checks it against Rekor's payload hash
            base64.b64decode(_need(body, "statement_b64", str, where), validate=True)
            ents = _need(body, "entries_b64", list, where)
            if not ents:
                raise ValueError
            for x in ents:  # the orphaned run's entries, so what it attested is published, not just explained
                base64.b64decode(x if isinstance(x, str) else None, validate=True)
        except (ValueError, TypeError):
            raise Bad(f"{where}: statement_b64 must be base64, and entries_b64 a non-empty list of base64 entries")
    extra = set(body) - allowed
    if extra:
        raise Bad(f"{where}: unknown body fields {sorted(extra)}")


# ---------- the chain ----------
def _listdir(ledger, sub, pattern, what):
    d = os.path.join(ledger, sub)
    if not os.path.exists(d):
        return []
    if os.path.islink(d) or not os.path.isdir(d):
        raise Bad(f"{sub}/ must be a plain directory")
    out = []
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        if not pattern.fullmatch(n) or os.path.islink(p) or not os.path.isfile(p):
            raise Bad(f"{sub}/{n!r}: unexpected {what}")
        out.append(p)
    return out


def load_chain(ledger):
    """Check the whole chain; return ([(entry, sha256)], {plan seq: verdict seq}). Raises Bad on any fault."""
    chain, plans, verdict_of = [], {}, {}
    for i, path in enumerate(_listdir(ledger, "entries", ENTRY_NAME, "file in entries"), start=1):
        name = os.path.basename(path)
        raw = open(path, "rb").read()
        e = parse_canonical(raw, name)
        if not isinstance(e, dict) or set(e) != {"seq", "prev", "kind", "written_at", "body"}:
            raise Bad(f"{name}: entry must have exactly seq, prev, kind, written_at, body")
        if e["seq"] != i or name != f"{i:06d}.json":
            raise Bad(f"{name}: seq {e['seq']!r} out of order (expected {i})")
        if e["prev"] != (chain[-1][1] if chain else None):
            raise Bad(f"{name}: prev doesn't match the previous entry's sha256")
        if e["kind"] not in KINDS:
            raise Bad(f"{name}: kind must be one of {KINDS}")
        if not isinstance(e["written_at"], str):
            raise Bad(f"{name}: written_at must be a string")
        check_body(e["kind"], e["body"], name)
        h = hashlib.sha256(raw).hexdigest()
        if e["kind"] == "plan":
            plans[i] = (h, e["body"]["by"])
        elif e["kind"] == "verdict":
            ref = e["body"]["plan"]
            if ref["seq"] not in plans or plans[ref["seq"]][0] != ref["sha256"]:
                raise Bad(f"{name}: verdict refers to no earlier plan with that seq and sha256")
            if ref["seq"] in verdict_of:
                raise Bad(f"{name}: plan {ref['seq']} already has a verdict (entry {verdict_of[ref['seq']]}); a re-run needs a new plan")
            if e["body"]["by"] == plans[ref["seq"]][1]:
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
    """Move every inbox proposal (in file-name order) into the chain. Used by CI, the only writer.
    Prints one line per entry as it's written; any bad proposal stops with exit 1 and nothing after it is written."""
    chain, _ = load_chain(a.ledger)
    os.makedirs(os.path.join(a.ledger, "entries"), exist_ok=True)
    for p in _listdir(a.ledger, "inbox", re.compile(r"[A-Za-z0-9._-]+\.json"), "file in inbox (proposals are NAME.json)"):
        n = os.path.basename(p)
        prop = parse_json(open(p, "rb").read(), f"inbox/{n}")
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
        print(out, flush=True)
    return 0


def cmd_verify(a):
    chain, verdict_of = load_chain(a.ledger)
    today = datetime.date.today().isoformat()
    plans = [e for e, _ in chain if e["kind"] == "plan"]
    open_ = [e["seq"] for e in plans if e["seq"] not in verdict_of]
    overdue = [e["seq"] for e in plans if e["seq"] not in verdict_of and e["body"].get("verdict_due", "9999-12-31") < today]
    print(json.dumps({"entries": len(chain), "head_seq": len(chain), "head_sha256": chain[-1][1] if chain else None,
                      "plans": len(plans), "verdicts": len(verdict_of),
                      "orphans_explained": sum(e["kind"] == "orphan" for e, _ in chain),
                      "open_plans": open_, "overdue_plans": overdue}, indent=1))
    return 0


def cmd_head(a):
    chain, _ = load_chain(a.ledger)
    print(f"{len(chain)} {chain[-1][1] if chain else '-'}")
    return 0


# ---------- anchors ----------
def _get(url, data=None, timeout=20, headers=False):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json",
                                                          "User-Agent": "exagium-record"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return (body, r.headers) if headers else body
    except urllib.error.HTTPError as ex:
        if ex.code == 404:
            return (None, None) if headers else None
        raise CantCheck(f"network: {url}: HTTP {ex.code}")
    except OSError as ex:
        raise CantCheck(f"network: {url}: {ex}")


def identity(repo, workflow, ref):
    return f"https://github.com/{repo}/{workflow}@{ref}"


def range_predicate(chain, from_seq, to_seq):
    """What the writer attests: the range of entries one run added, tied to the chain on both ends."""
    return {"from_seq": from_seq, "to_seq": to_seq,
            "prev_sha256": chain[from_seq - 2][1] if from_seq > 1 else None, "head_sha256": chain[to_seq - 1][1]}


def check_sigstore(bundle, digest, repo, workflow, ref):
    """Verify offline that a bundle is an attestation of this digest, signed by exactly this workflow on this ref.
    Returns (integrated_time, predicate). Wrong data is Bad; a missing library is CantCheck."""
    try:
        from sigstore.models import Bundle
        from sigstore.verify import Verifier, policy
        from sigstore.errors import Error as SigstoreError
    except ImportError as ex:
        raise CantCheck(f"sigstore-python isn't usable ({ex}); pip install sigstore==4.5.1 in a virtualenv")
    name = os.path.basename(bundle)
    raw = open(bundle, "rb").read()
    pol = policy.AllOf([policy.Identity(identity=identity(repo, workflow, ref), issuer=ISSUER),
                        policy.GitHubWorkflowRepository(repo), policy.GitHubWorkflowRef(ref)])
    try:
        b = Bundle.from_json(raw)
        ptype, payload = Verifier.production(offline=True).verify_dsse(b, pol)
    except SigstoreError as ex:   # verification errors and malformed bundles alike: the anchor is wrong
        raise Bad(f"{name}: Sigstore verification failed: {ex}")
    except (ValueError, KeyError, TypeError) as ex:
        raise Bad(f"{name}: malformed bundle ({type(ex).__name__}: {ex})")
    st = parse_json(payload, f"{name} payload")
    if ptype != "application/vnd.in-toto+json" or not isinstance(st, dict) or \
            [s.get("digest", {}).get("sha256") for s in st.get("subject", [])] != [digest]:
        raise Bad(f"{name}: the attestation's subject isn't this entry")
    try:  # the same bytes the verifier just checked; their signed entry timestamp covers integratedTime
        t = int(json.loads(raw)["verificationMaterial"]["tlogEntries"][0]["integratedTime"])
        from cryptography.x509 import ObjectIdentifier
        commit = b.signing_certificate.extensions.get_extension_for_oid(
            ObjectIdentifier("1.3.6.1.4.1.57264.1.3")).value.value.decode("ascii")  # the commit the run was given
    except Exception as ex:
        raise Bad(f"{name}: malformed bundle or certificate ({type(ex).__name__}: {ex})")
    return t, st.get("predicate"), commit


def check_ots(proof, entry_raw, explorers):
    """Returns the earliest Bitcoin block time the proof reaches, or None if it's still pending."""
    try:
        from opentimestamps.core.timestamp import DetachedTimestampFile
        from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
        from opentimestamps.core.serialize import StreamDeserializationContext, DeserializationError
    except ImportError:
        raise CantCheck("opentimestamps isn't installed (pip install opentimestamps-client==0.8.0)")
    name = os.path.basename(proof)
    try:
        with open(proof, "rb") as f:
            dt = DetachedTimestampFile.deserialize(StreamDeserializationContext(f))
        atts = list(dt.timestamp.all_attestations())
    except (DeserializationError, ValueError, TypeError, EOFError) as ex:
        raise Bad(f"{name}: malformed proof ({ex})")
    if dt.file_digest != hashlib.sha256(entry_raw).digest():
        raise Bad(f"{name}: proof is for different bytes")
    times = []
    for msg, att in atts:
        if not isinstance(att, BitcoinBlockHeaderAttestation):
            continue
        headers = set()
        for base in explorers:
            bh = _get(f"{base}/block-height/{att.height}")
            if bh is None:  # 404: either the block doesn't exist (a forged height) or the explorer is behind
                tip = _get(f"{base}/blocks/tip/height")
                if tip is not None and att.height > int(tip):
                    raise Bad(f"{name}: claims block {att.height}, beyond the chain tip {int(tip)}")
                raise CantCheck(f"{base}: no block at height {att.height}")
            bh = bh.decode().strip()
            hdr = _get(f"{base}/block/{bh}/header")
            if hdr is None:
                raise CantCheck(f"{base}: no header for block {bh}")
            hdr = bytes.fromhex(hdr.decode().strip())
            if hashlib.sha256(hashlib.sha256(hdr).digest()).digest()[::-1].hex() != bh:
                raise CantCheck(f"{base}: header doesn't hash to block {bh} (explorer fault)")
            headers.add(hdr)
        if len(headers) != 1:
            raise CantCheck(f"explorers disagree about block {att.height}")
        hdr = headers.pop()
        if hdr[36:68] != msg:
            raise Bad(f"{name}: path doesn't reach block {att.height}'s merkle root")
        times.append(int.from_bytes(hdr[68:72], "little"))
    return min(times) if times else None


def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def cmd_check_anchors(a):
    """Every anchor must be right, and the writer's Sigstore ranges must cover entries 1..head with no gap or overlap.
    Checks every anchor even after a failure; exit 1 if anything was wrong, else 2 if anything couldn't be checked."""
    chain, _ = load_chain(a.ledger)
    rows, bad, cant, ranges = [], [], [], []
    for path in _listdir(a.ledger, "anchors", ANCHOR_NAME, "file in anchors"):
        n = os.path.basename(path)
        m = ANCHOR_NAME.fullmatch(n)
        seq = int(m.group(1))
        try:
            if not 1 <= seq <= len(chain):
                raise Bad(f"anchors/{n}: no entry {seq}")
            raw = open(os.path.join(a.ledger, "entries", f"{seq:06d}.json"), "rb").read()
            if m.group(2) == "ots":
                t = check_ots(path, raw, a.explorer)
                rows.append({"anchor": n, "kind": "opentimestamps", "existed_by": iso(t) or "pending"})
            else:
                t, pred, commit = check_sigstore(path, chain[seq - 1][1], a.repo, a.workflow, a.ref)
                if not isinstance(pred, dict) or not isinstance(pred.get("from_seq"), int) or pred.get("to_seq") != seq \
                        or not 1 <= pred["from_seq"] <= seq or pred != range_predicate(chain, pred["from_seq"], seq):
                    raise Bad(f"anchors/{n}: attested range {pred!r} doesn't match this chain")
                ranges.append((pred["from_seq"], seq, commit))
                rows.append({"anchor": n, "kind": "sigstore", "entries": f"{pred['from_seq']}-{seq}", "existed_by": iso(t),
                             "writer_commit": commit})
        except Bad as ex:
            bad.append(str(ex))
        except CantCheck as ex:
            cant.append(f"anchors/{n}: {ex}")
    covered, changes, prev_commit = 0, [], None
    for lo, hi, commit in sorted(ranges):
        if lo != covered + 1:
            bad.append(f"the writer's ranges {'overlap' if lo <= covered else 'leave a gap'} at entry {min(lo, covered + 1)}")
        covered = max(covered, hi)
        if prev_commit:  # did the writer's code change between these two runs? (information: a stranger reads the diff)
            changes.append({"entries": f"{lo}-{hi}", "from_commit": prev_commit, "to_commit": commit,
                            "writer_code_changed": _code_changed(a, prev_commit, commit)})
        prev_commit = commit
    if covered < len(chain):
        bad.append(f"entries {covered + 1}-{len(chain)} were never attested by the writer")
    print(json.dumps({"anchors": rows, "entries": len(chain), "writer_attested_through": covered,
                      "writer_runs": changes, "wrong": bad, "could_not_check": cant}, indent=1))
    for x in bad:
        print(f"BAD: {x}", file=sys.stderr)
    for x in cant:
        print(f"CAN'T CHECK: {x}", file=sys.stderr)
    return 1 if bad else 2 if cant else 0


def _git(a, *args):
    root = os.path.dirname(os.path.abspath(a.ledger))
    r = subprocess.run(["git", "-C", root, *args], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _code_changed(a, c1, c2):
    """True/False if anything under record/ or .github/ differs between two commits; "unknown" without that history."""
    if _git(a, "cat-file", "-e", f"{c1}^{{commit}}") is None or _git(a, "cat-file", "-e", f"{c2}^{{commit}}") is None:
        return "unknown (commit not in this clone)"
    root = os.path.dirname(os.path.abspath(a.ledger))
    r = subprocess.run(["git", "-C", root, "diff", "--quiet", c1, c2, "--", "record", ".github"])
    return {0: False, 1: True}.get(r.returncode, "unknown")


def _orphan_entries_problem(o, pred):
    """The orphan must publish the hidden entries: canonical, consecutive from from_seq, chained from prev_sha256,
    ending at head_sha256. Returns a description of the first problem, or None."""
    try:
        raws = [base64.b64decode(x) for x in o["body"]["entries_b64"]]
        if not isinstance(pred.get("from_seq"), int) or len(raws) != pred.get("to_seq", -1) - pred["from_seq"] + 1:
            return "its entries don't match the attested range"
        prev = pred.get("prev_sha256")
        for i, raw in enumerate(raws):
            e = parse_canonical(raw, f"hidden entry {i + 1}")
            if not isinstance(e, dict) or e.get("seq") != pred["from_seq"] + i or e.get("prev") != prev:
                return "its entries don't chain from the attested prev_sha256"
            if e.get("kind") not in KINDS:
                return f"hidden entry {e.get('seq')} has an unknown kind"
            check_body(e["kind"], e.get("body"), f"hidden entry {e['seq']}")
            prev = hashlib.sha256(raw).hexdigest()
        if prev != pred.get("head_sha256"):
            return "its entries don't hash to the attested head"
    except Bad as ex:
        return f"its entries are malformed ({ex})"
    return None


def _published_entry_hashes(a, rev):
    """sha256 of every version of every entry file ever committed in rev's history."""
    if (_git(a, "rev-parse", "--is-shallow-repository") or b"").strip() != b"false":
        raise CantCheck("this clone is shallow or not a git repository; orphan checks need the ledger branch's full history")
    raw = _git(a, "log", rev, "--format=", "--raw", "--no-abbrev", "--", "ledger/entries")
    if raw is None:
        raise CantCheck(f"no git history for {rev} next to the ledger (run monitor in a full clone)")
    blobs = {ln.split()[3] for ln in raw.decode().splitlines() if ln.startswith(":") and ln.split()[3].strip("0")}
    out = set()
    for b in blobs:
        data = _git(a, "cat-file", "blob", b)
        if data is not None:
            out.add(hashlib.sha256(data).hexdigest())
    return out


def _pages(url):
    """Follow GitHub's Link: rel="next" (works for page= and the cursor-paged activity API). The next link may name
    the repo by numeric id; keep the original /repos/owner/name path and take only the query from it."""
    import urllib.parse
    base = url.split("?")[0]
    out = []
    while url:
        body, hdrs = _get(url, headers=True)
        if body is None:
            raise CantCheck(f"404 from {url}")
        out.extend(json.loads(body))
        m = re.search(r'<([^>]+)>;\s*rel="next"', (hdrs.get("Link") if hdrs else None) or "")
        url = f"{base}?{urllib.parse.urlsplit(m.group(1)).query}" if m else None
    return out


def _refs(repo):
    """Every ref GitHub will name for this repo: branches, tags, every pull request's merge and head refs,
    and every ref in the activity log (which keeps deleted branches while the repo exists)."""
    api = f"https://api.github.com/repos/{repo}"
    refs = {"refs/heads/main"}
    refs.update("refs/heads/" + x["name"] for x in _pages(f"{api}/branches?per_page=100"))
    refs.update("refs/tags/" + x["name"] for x in _pages(f"{api}/tags?per_page=100"))
    refs.update(f"refs/pull/{x['number']}/{k}" for x in _pages(f"{api}/pulls?state=all&per_page=100") for k in ("merge", "head"))
    activity = _pages(f"{api}/activity?per_page=100")
    refs.update(x["ref"] for x in activity if x.get("ref"))
    return sorted(refs), activity


def cmd_monitor(a):
    """List every Rekor entry signed by the writer workflow on any ref GitHub names. Each must be an anchor in the
    ledger, or be explained by an orphan entry whose statement matches Rekor's payload hash and whose entries were
    never published on the ledger branch (so an orphan can't launder a deleted verdict). Any force push to the
    ledger branch is wrong. Entries are matched by Rekor log index, unique per attestation."""
    chain, _ = load_chain(a.ledger)
    current = {h for _, h in chain}
    known, bad = {}, []
    for path in _listdir(a.ledger, "anchors", ANCHOR_NAME, "file in anchors"):
        if path.endswith(".sigstore.json"):
            try:
                known[int(json.load(open(path))["verificationMaterial"]["tlogEntries"][0]["logIndex"])] = os.path.basename(path)
            except (ValueError, KeyError, IndexError, TypeError) as ex:
                raise Bad(f"{os.path.basename(path)}: malformed bundle ({ex})")
    explained = {e["body"]["log_index"]: e for e, _ in chain if e["kind"] == "orphan"}
    if a.only_ref:
        refs, activity = [a.only_ref], []
    else:
        refs, activity = _refs(a.repo)
        for x in activity:
            if x.get("activity_type") in ("force_push", "branch_deletion") and x.get("ref") == a.ledger_ref:
                bad.append(f"{x['activity_type']} on {a.ledger_ref} at {x.get('timestamp')} (rewrites the ledger's history)")
    published = _published_entry_hashes(a, a.git_rev) if explained else set()
    seen, unknown, orphans = set(), [], []
    for ref in refs:
        uuids = _get("https://rekor.sigstore.dev/api/v1/index/retrieve", json.dumps({"subject": identity(a.repo, a.workflow, ref)}).encode())
        if uuids is None:
            raise CantCheck("Rekor's index returned 404")
        for u in json.loads(uuids):
            got = _get(f"https://rekor.sigstore.dev/api/v1/log/entries/{u}")
            if got is None:
                raise CantCheck(f"Rekor lists {u} but won't return it")
            v = next(iter(json.loads(got).values()))
            idx = int(v["logIndex"])
            seen.add(idx)
            row = {"ref": ref, "log_index": idx, "integrated": iso(v["integratedTime"]), "rekor_uuid": u}
            if idx in known:
                continue
            if idx not in explained:
                unknown.append(row)
                continue
            o = explained[idx]
            body = json.loads(base64.b64decode(v["body"]))
            stmt = base64.b64decode(o["body"]["statement_b64"])
            if body.get("kind") != "dsse" or body["spec"]["payloadHash"]["value"] != hashlib.sha256(stmt).hexdigest():
                bad.append(f"orphan entry {o['seq']}: its statement isn't the one Rekor holds for log index {idx}")
                continue
            pred = (parse_json(stmt, f"orphan entry {o['seq']} statement") or {}).get("predicate") or {}
            head = pred.get("head_sha256")
            problem = _orphan_entries_problem(o, pred)
            if problem:
                bad.append(f"orphan entry {o['seq']}: {problem}")
                continue
            if head in published and head not in current:
                bad.append(f"orphan entry {o['seq']}: the attested head {head} was published on {a.git_rev} and later removed")
                continue
            orphans.append(dict(row, explained_by_entry=o["seq"], attested_range=f"{pred.get('from_seq')}-{pred.get('to_seq')}"))
    missing = sorted(n for idx, n in known.items() if idx not in seen) if not a.only_ref else []
    print(json.dumps({"refs_checked": refs, "rekor_entries": len(seen), "ledger_anchors": len(known),
                      "explained_orphans": orphans, "unexplained": unknown, "wrong": bad,
                      "ledger_anchors_not_found_in_rekor": missing if not a.only_ref else "not checked with --only-ref"}, indent=1))
    for x in bad:
        print(f"BAD: {x}", file=sys.stderr)
    return 1 if unknown or missing or bad else 0


def cmd_range(a):
    chain, _ = load_chain(a.ledger)
    if not 1 <= a.from_seq <= a.to_seq <= len(chain):
        raise Bad(f"no range {a.from_seq}-{a.to_seq} in a chain of {len(chain)}")
    print(json.dumps(range_predicate(chain, a.from_seq, a.to_seq), sort_keys=True, separators=(",", ":")))
    return 0


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=os.path.join(here, "..", "ledger"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    sub.add_parser("verify")
    sub.add_parser("head")
    rp = sub.add_parser("range", help="print the writer's predicate for entries FROM..TO (used by CI)")
    rp.add_argument("from_seq", type=int)
    rp.add_argument("to_seq", type=int)
    for name in ("check-anchors", "monitor"):
        p = sub.add_parser(name)
        p.add_argument("--repo", default="5kasgard/exagium")
        p.add_argument("--workflow", default=".github/workflows/record.yml")
        if name == "check-anchors":
            p.add_argument("--ref", default="refs/heads/main")
            p.add_argument("--explorer", action="append", default=None,
                           help="esplora API base; default blockstream.info and mempool.space")
        else:
            p.add_argument("--only-ref", help="check this ref only (default: every ref GitHub names)")
            p.add_argument("--ledger-ref", default="refs/heads/main", help="the ledger's branch; a force push to it is wrong")
            p.add_argument("--git-rev", default="HEAD", help="the ledger branch's history in this clone (for orphan checks)")
    a = ap.parse_args(argv)
    if getattr(a, "explorer", "x") is None:
        a.explorer = ["https://blockstream.info/api", "https://mempool.space/api"]
    fn = {"ingest": cmd_ingest, "verify": cmd_verify, "head": cmd_head, "range": cmd_range,
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
