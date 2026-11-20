"""Check due facts in facts/register.json by script, no model (queue P30). Usage:
python3 -I facts/check.py REGISTER STATUS_JSON [--keyed]   -> exit 0 all same or not due, 2 if any changed, 3 if a probe could not run.
Facts marked "keyed" need a provider key (model lists, account allowances): checked only with --keyed, in wiring.yml,
which holds the keys and runs no generated code; the worker's idle job checks the rest without keys.
STATUS_JSON keeps each fact's last check; it lives on the worker-runs branch, not in the register (the register is the claim)."""
import datetime as dt, hashlib, json, os, socket, ssl, sys, urllib.parse, urllib.request

TIMEOUT = 60


def get(url, key_env=None):
    h = {"User-Agent": "exagium-facts"}
    if key_env:
        h["Authorization"] = "Bearer " + os.environ[key_env]
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=TIMEOUT) as r:
        return r.read()


def field(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def probe(p):
    k = p["kind"]
    if k == "sha256_url":
        try:
            data = get(p["url"])
        except Exception as e:   # the hash is the fact; transport trust is a separate fact (tls_valid)
            if not isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
                raise
            with urllib.request.urlopen(p["url"], timeout=TIMEOUT, context=ssl._create_unverified_context()) as r:
                data = r.read()
        return hashlib.sha256(data).hexdigest()
    if k == "tls_valid":
        try:
            with socket.create_connection((p["host"], 443), timeout=TIMEOUT) as s, \
                    ssl.create_default_context().wrap_socket(s, server_hostname=p["host"]):
                return True
        except ssl.SSLCertVerificationError:
            return False
    if k == "openrouter_listed":
        # output_modalities=all: the default list leaves out decision and rerank models (2026-11-20: the decider and the
        # reranker read as "gone" while both answered; 741 models listed this way against 477 by default)
        ids = {m["id"] for m in json.loads(get("https://openrouter.ai/api/v1/models?output_modalities=all"))["data"]}
        return p["model"] in ids
    if k == "openrouter_key":   # the account's allowances, as the key endpoint reports them
        return field(json.loads(get("https://openrouter.ai/api/v1/key", "OPENROUTER_API_KEY"))["data"], p["field"])
    if k in ("nim_listed", "groq_listed"):
        url = {"nim_listed": ("https://integrate.api.nvidia.com/v1/models", "NVIDIA_API_KEY"),
               "groq_listed": ("https://api.groq.com/openai/v1/models", "GROQ_API_KEY")}[k]
        return p["model"] in {m["id"] for m in json.loads(get(*url))["data"]}
    if k == "anthropic_listed":   # Claude models on the owner's API key (CLAUDE_API_KEY, added 2026-11-20); the list call is free
        req = urllib.request.Request("https://api.anthropic.com/v1/models?limit=100",
                                     headers={"x-api-key": os.environ["CLAUDE_API_KEY"], "anthropic-version": "2023-06-01",
                                              "User-Agent": "exagium-facts"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return p["model"] in {m["id"] for m in json.loads(r.read())["data"]}
    if k == "cloudflare_listed":
        url = (f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}/ai/models/search?"
               + urllib.parse.urlencode({"search": p["model"].split("/")[-1], "per_page": 100}))
        return p["model"] in {m.get("name") for m in json.loads(get(url, "CLOUDFLARE_API_TOKEN"))["result"]}
    if k == "hf_dataset_license":
        return (json.loads(get(f"https://huggingface.co/api/datasets/{p['dataset']}")).get("cardData") or {}).get("license")
    raise ValueError(f"unknown probe kind {k}")


def run(register, status_path, now=None, keyed=False):
    now = now or dt.datetime.now(dt.timezone.utc)
    reg = json.load(open(register))
    st = json.load(open(status_path)) if os.path.exists(status_path) else {}
    changed, broken = [], []
    for f in reg["facts"]:
        if bool(f.get("keyed")) != keyed:
            continue
        last = st.get(f["id"], {})
        if last.get("at") and now - dt.datetime.fromisoformat(last["at"]) < dt.timedelta(days=f.get("recheck_days", 1)):
            continue
        try:
            v = probe(f["probe"])
        except Exception as e:
            broken.append(f"{f['id']}: probe failed: {str(e)[:160]}")
            st[f["id"]] = dict(last, error=str(e)[:200], error_at=now.isoformat())
            continue
        same = v == f["value"]
        st[f["id"]] = {"at": now.isoformat(), "value": v, "same": same,
                       "dependants": [] if same else (f.get("dependants") or [])}
        if not same:
            changed.append(f"{f['id']}: registered {f['value']!r}, now {v!r}; cited by: {'; '.join(f['cited_by'])}")
    with open(status_path, "w") as fh:
        json.dump(st, fh, indent=1)
    for m in changed:
        print("CHANGED", m)
    for m in broken:
        print("PROBE FAILED", m)
    print(f"facts: {len(changed)} changed, {len(broken)} probe failures, {sum(bool(f.get('keyed')) == keyed for f in reg['facts'])} registered ({'keyed' if keyed else 'keyless'})")
    return 2 if changed else 3 if broken else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2], keyed="--keyed" in sys.argv[3:]))
