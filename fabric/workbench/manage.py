"""Keep the owner's LobeHub workbench in step with config in the repo, through LobeHub's own CLI (`lh`, signed in
with LOBEHUB_CLI_API_KEY). Owner 2026-11-25: "add automation so you can work on and change lobehub directly".

  python3 -I fabric/workbench/manage.py status                   # what is there: providers, models, connectors, agents
  python3 -I fabric/workbench/manage.py apply CONFIG_JSON        # make connectors and agents match the config
  python3 -I fabric/workbench/manage.py run AGENT_KEY PROMPT     # run a forge-managed agent once (headless, free tools)
  python3 -I fabric/workbench/manage.py due SCHEDULES_JSON OUT   # run every schedule entry due this UTC hour; status to OUT
  python3 -I fabric/workbench/manage.py errors [DAYS]            # recent failed replies: when, provider/model, error type and
                                                                 # the provider's message (redacted); never message content

Public logs: prints ids, titles, hosts and counts only, never the account's email, keys or headers. Changes come only
from merged config (fabric/workbench/config/); agents it manages carry the tag forge-managed and are found by it."""
import json, os, subprocess, sys, tempfile
from urllib.parse import urlparse

LH = os.environ.get("LH", "lh")
TAG = "forge-managed"
FREE = ["nvidia", "groq", "cerebras", "opencodezen", "huggingface"]   # providers whose free models status lists in full


def lh(*args, quiet=False):
    r = subprocess.run([LH, *args], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["?"]
        if quiet:
            return None
        sys.exit(f"lh {args[0]} {args[1] if len(args) > 1 else ''}: failed: {msg[0][:200]}")
    out = r.stdout.strip()
    try:
        return json.loads(out) if out else ""
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
    for pid in FREE:
        ms = rows(lh("model", "list", pid, "-L", "300", "--type", "chat", "--json", quiet=True) or [])
        tools = [m.get("id") + ("*" if m.get("enabled") else "") for m in ms if (m.get("abilities") or {}).get("functionCall")]
        print(f"  all {pid} chat models with tool calling (* = enabled): {', '.join(tools)}")
    cons = rows(lh("connector", "list", "--json"))
    print(f"connectors: {len(cons)}")
    for c in cons:
        url = c.get("url") or (c.get("customParams") or {}).get("mcp", {}).get("url") or ""
        print(f"  {c.get('identifier')}: {urlparse(url).netloc or c.get('type', '')}")
    ags = rows(lh("agent", "list", "-L", "100", "--json"))
    print(f"agents: {len(ags)}")
    for a in ags:
        v = lh("agent", "view", a["id"], "--json", quiet=True) or {}
        v = v if isinstance(v, dict) else {}
        print(f"  {v.get('title') or a.get('title')}: {v.get('provider')}/{v.get('model')} plugins={v.get('plugins') or []} tags={v.get('tags') or a.get('tags') or []}")


def apply(path):
    cfg = json.load(open(path))
    for e in cfg.get("enable_models", []):   # one at a time: batch-toggle answered a server error (first apply run)
        for mid in e["ids"]:
            ok = lh("model", "toggle", mid, "--provider", e["provider"], "--enable", quiet=True) is not None
            print(f"model {e['provider']}/{mid}: {'enabled' if ok else 'NOT enabled (lh error; see the workbench logs)'}")
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
    if cfg.get("inbox"):   # the default "Lobe AI" chat: LobeHub ships it on deepseek, which has no key here
        i = cfg["inbox"]
        lh("agent", "edit", "--slug", "inbox", "-p", i["provider"], "-m", i["model"], "--json", "id")
        print(f"default chat (inbox): {i['provider']}/{i['model']}")
    for a in cfg.get("agents", []):
        key = f"forge:{a['key']}"
        found = rows(lh("agent", "list", "--tag", key, "--json"))
        extra = {k: a[k] for k in ("openingMessage", "openingQuestions", "params", "chatConfig") if k in a}
        if "plugins" in a:   # LobeHub's shape: [{"identifier": ..., "mode": "auto"}] (seen in status, 2026-11-25)
            extra["plugins"] = [{"identifier": x, "mode": "auto"} for x in cfg.get("builtin_plugins", []) + a["plugins"]]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(extra, f)
        role = (cfg.get("preamble", "") + "\n\n" + a["system_role"]).strip()   # every agent knows what it can use
        flags = ["-t", a["title"], "-d", a.get("description", ""), "-p", a["provider"], "-m", a["model"], "-s", role]
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


def errors(days):
    """The workbench stores each failed reply with its error (type and the provider's body). Vercel's runtime-log API
    returned 0 records three times, so this reads the failures where LobeHub keeps them."""
    import re
    from datetime import datetime, timedelta, timezone
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    msgs = rows(lh("message", "list", "--role", "assistant", "--start", start, "-L", "200", "--json"))
    bad = [m for m in msgs if m.get("error")]
    print(f"assistant replies since {start}: {len(msgs)}; with an error: {len(bad)}")
    by = {}
    for m in bad:
        e = m["error"] if isinstance(m["error"], dict) else {"type": str(m["error"])}
        body = json.dumps(e.get("body") or e.get("message") or "")[:300]
        body = re.sub(r"[\w.+-]+@[\w-]+(\.[\w-]+)+|https?://\S+|[A-Za-z0-9_-]{32,}", "<redacted>", body)
        key = f"{m.get('provider')}/{m.get('model')} {e.get('type')}"
        by.setdefault(key, []).append(f"{str(m.get('createdAt'))[:19]} {body}")
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(v):3d}  {k}")
        for x in v[:3]:
            print(f"         {x}")


def agent_id(key):
    found = rows(lh("agent", "list", "--tag", f"forge:{key}", "--json"))
    if not found:
        sys.exit(f"no agent tagged forge:{key}")
    return found[0]["id"]


def run(key, prompt, timeout=600):
    """One headless run. Returns (exit code, seconds). The reply stays in LobeHub (a topic the owner can open) and,
    when DISCORD_WEBHOOK_URL is set, is also posted to the owner's private Discord as the agent's persona
    (fabric/discord/post.py). The public log gets only the status, because agent output may quote private material."""
    import time
    t = time.time()
    aid = agent_id(key)
    t_lookup = time.time()
    probe = os.environ.get("FORGE_STREAM_PROBE") == "1"
    args = [LH, "agent", "run", "-a", aid, "-p", prompt, "--timeout", str(timeout)] + (["--json"] if probe else [])
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 60)
    print(f"timing: agent lookup {t_lookup - t:.1f} s, agent run {time.time() - t_lookup:.1f} s")
    if probe:   # learn the event stream's shape for live replies: event types and field names only, never content
        from collections import Counter
        kinds, keys, first = Counter(), Counter(), []
        for line in (r.stdout or "").splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                kinds["(not json)"] += 1; continue
            k = str(ev.get("type") or ev.get("event") or "?") if isinstance(ev, dict) else type(ev).__name__
            kinds[k] += 1
            if isinstance(ev, dict):
                keys.update(f"{k}.{f}" for f in ev)
                if isinstance(ev.get("data"), dict):
                    keys.update(f"{k}.data.{f}" for f in ev["data"])
            if len(first) < 12 and (not first or first[-1] != k):
                first.append(k)
        print("stream event types:", dict(kinds.most_common(20)))
        print("stream first order:", first)
        print("stream fields:", sorted(keys)[:80])
        return r.returncode, round(time.time() - t)
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "discord"))
        import post
        head = {0: "", 1: "(the run failed)\n", 2: "(waiting for your approval in LobeHub)\n", 3: "(still running)\n"}
        codes = post.post("agent", key, head.get(r.returncode, "") + (r.stdout or ""))
        print("discord:", codes)
    return r.returncode, round(time.time() - t)


def due(path, out):
    """Schedule entries: {"key", "agent", "prompt", "hours_utc": [..], "days": ["mon",..] (optional), "enabled"}.
    Due when this UTC hour is listed (and today is listed, if days are given)."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    day = now.strftime("%a").lower()
    state = json.load(open(out)) if os.path.exists(out) else {}
    for e in json.load(open(path)).get("schedules", []):
        if not e.get("enabled", True) or now.hour not in e.get("hours_utc", []) or (e.get("days") and day not in e["days"]):
            continue
        if state.get(e["key"], {}).get("last_hour") == now.strftime("%Y-%m-%dT%H"):
            continue   # already ran this hour (a re-run of the job)
        code, secs = run(e["agent"], e["prompt"], e.get("timeout_s", 600))
        state[e["key"]] = {"last_hour": now.strftime("%Y-%m-%dT%H"), "at": now.isoformat()[:19] + "Z", "agent": e["agent"],
                           "exit": code, "seconds": secs, "meaning": {0: "done", 1: "failed", 2: "waiting for approval", 3: "unconfirmed"}.get(code, "?")}
        print(f"schedule {e['key']}: {state[e['key']]['meaning']} in {secs} s ({e['agent']})")
    json.dump(state, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["status"]:
        status()
    elif sys.argv[1:2] == ["run"]:
        code, secs = run(sys.argv[2], sys.argv[3])
        print(f"run {sys.argv[2]}: exit {code} in {secs} s (0 done, 1 failed, 2 waiting for approval, 3 unconfirmed)")
        sys.exit(code)
    elif sys.argv[1:2] == ["due"]:
        due(sys.argv[2], sys.argv[3])
    elif sys.argv[1:2] == ["errors"]:
        errors(float(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif sys.argv[1:2] == ["apply"]:
        apply(sys.argv[2])
    else:
        sys.exit(__doc__)
