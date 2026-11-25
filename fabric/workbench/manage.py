"""Keep the owner's LobeHub workbench in step with config in the repo, through LobeHub's own CLI (`lh`, signed in
with LOBEHUB_CLI_API_KEY). Owner 2026-11-25: "add automation so you can work on and change lobehub directly".

  python3 -I fabric/workbench/manage.py status                   # what is there: providers, models, connectors, agents
  python3 -I fabric/workbench/manage.py apply CONFIG_JSON        # make connectors and agents match the config

Public logs: prints ids, titles, hosts and counts only, never the account's email, keys or headers. Changes come only
from merged config (fabric/workbench/config/); agents it manages carry the tag forge-managed and are found by it."""
import json, os, subprocess, sys, tempfile
from urllib.parse import urlparse

LH = os.environ.get("LH", "lh")
TAG = "forge-managed"


def lh(*args, quiet=False):
    r = subprocess.run([LH, *args], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["?"]
        if quiet:
            return None
        sys.exit(f"lh {args[0]} {args[1] if len(args) > 1 else ''}: failed: {msg[0][:200]}")
    out = r.stdout.strip()
    try:
        return json.loads(out) if out else None
    except ValueError:
        return out


def rows(x):
    if isinstance(x, dict):
        for k in ("data", "items", "list", "agents", "providers", "models", "connectors"):
            if isinstance(x.get(k), list):
                return x[k]
    return x if isinstance(x, list) else []


def status():
    who = lh("whoami", "--json") or {}
    print("signed in:", "yes" if who else "no")
    provs = rows(lh("provider", "list", "--json"))
    on = [p for p in provs if p.get("enabled")]
    print(f"providers: {len(provs)} known, {len(on)} enabled: {', '.join(sorted(p.get('id', '?') for p in on))}")
    for p in on:
        ms = rows(lh("model", "list", p["id"], "--enabled", "--type", "chat", "--json", quiet=True) or [])
        tools = [m.get("id") for m in ms if (m.get("abilities") or {}).get("functionCall")]
        print(f"  {p['id']}: {len(ms)} chat models enabled, {len(tools)} with tool calling: {', '.join(tools[:12])}")
    cons = rows(lh("connector", "list", "--json"))
    print(f"connectors: {len(cons)}")
    for c in cons:
        url = c.get("url") or (c.get("customParams") or {}).get("mcp", {}).get("url") or ""
        print(f"  {c.get('identifier')}: {urlparse(url).netloc or c.get('type', '')}")
    ags = rows(lh("agent", "list", "-L", "100", "--json"))
    print(f"agents: {len(ags)}")
    for a in ags:
        print(f"  {a.get('title')}: {a.get('provider')}/{a.get('model')} tags={a.get('tags') or []}")


def apply(path):
    cfg = json.load(open(path))
    have = {c.get("identifier") for c in rows(lh("connector", "list", "--json"))}
    for c in cfg.get("connectors", []):
        if c["identifier"] in have:
            print(f"connector {c['identifier']}: present, kept"); continue
        args = ["connector", "add", "--url", c["url"], "--name", c["name"], "--identifier", c["identifier"]]
        if c.get("bearer_env"):
            tok = os.environ.get(c["bearer_env"])
            if not tok:
                print(f"connector {c['identifier']}: {c['bearer_env']} not set, skipped"); continue
            args += ["--auth", "bearer", "--token", tok]
        else:
            args += ["--auth", "none"]
        lh(*args, "--json")
        print(f"connector {c['identifier']}: added")
    for a in cfg.get("agents", []):
        key = f"forge:{a['key']}"
        found = rows(lh("agent", "list", "--tag", key, "--json"))
        extra = {k: a[k] for k in ("plugins", "openingMessage", "openingQuestions", "params", "chatConfig") if k in a}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(extra, f)
        flags = ["-t", a["title"], "-d", a.get("description", ""), "-p", a["provider"], "-m", a["model"], "-s", a["system_role"]]
        if found:
            lh("agent", "edit", found[0]["id"], *flags, "--config-file", f.name, "--json", "id")
            print(f"agent {a['title']}: updated ({a['provider']}/{a['model']})")
        else:
            made = lh("agent", "create", *flags, "--tag", TAG, "--tag", key, "--json", "id")
            aid = (made or {}).get("id") if isinstance(made, dict) else None
            if aid and extra:
                lh("agent", "edit", aid, "--config-file", f.name, "--json", "id")
            print(f"agent {a['title']}: created ({a['provider']}/{a['model']})")
        os.unlink(f.name)


if __name__ == "__main__":
    if sys.argv[1:2] == ["status"]:
        status()
    elif sys.argv[1:2] == ["apply"]:
        apply(sys.argv[2])
    else:
        sys.exit(__doc__)
