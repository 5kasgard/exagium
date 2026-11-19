"""Check due facts in facts/register.json by script, no model (queue P30). Usage:
python3 -I facts/check.py REGISTER STATUS_JSON   -> exit 0 all same or not due, 2 if any changed, 3 if a probe could not run.
STATUS_JSON keeps each fact's last check; it lives on the worker-runs branch, not in the register (the register is the claim)."""
import datetime as dt, hashlib, json, os, socket, ssl, sys, urllib.request

TIMEOUT = 60


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "exagium-facts"}), timeout=TIMEOUT) as r:
        return r.read()


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
        ids = {m["id"] for m in json.loads(get("https://openrouter.ai/api/v1/models"))["data"]}
        return p["model"] in ids
    if k == "hf_dataset_license":
        return (json.loads(get(f"https://huggingface.co/api/datasets/{p['dataset']}")).get("cardData") or {}).get("license")
    raise ValueError(f"unknown probe kind {k}")


def run(register, status_path, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    reg = json.load(open(register))
    st = json.load(open(status_path)) if os.path.exists(status_path) else {}
    changed, broken = [], []
    for f in reg["facts"]:
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
        st[f["id"]] = {"at": now.isoformat(), "value": v, "same": same}
        if not same:
            changed.append(f"{f['id']}: registered {f['value']!r}, now {v!r}; cited by: {'; '.join(f['cited_by'])}")
    with open(status_path, "w") as fh:
        json.dump(st, fh, indent=1)
    for m in changed:
        print("CHANGED", m)
    for m in broken:
        print("PROBE FAILED", m)
    print(f"facts: {len(changed)} changed, {len(broken)} probe failures, {len(reg['facts'])} registered")
    return 2 if changed else 3 if broken else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2]))
