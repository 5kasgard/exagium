"""Build the Forge dashboard's data (fabric/dash): one JSON from the public record, for the owner to see the system.
Standard library only. Runs hourly in .github/workflows/dash-snapshot.yml (free runner, the job's own GITHUB_TOKEN).

  python3 -I fabric/dash/snapshot.py QUEUE_JSON OUT_JSON      # reads GITHUB_TOKEN (optional) for PRs and workflow runs

Every number is computed from a named source (queue.json runs, the GitHub API); nothing is estimated. Records name
their recorder; this file only counts them (audit/fabric/README.md)."""
import json, os, sys, time, urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

REPO = "5kasgard/exagium"
FINISHED = {"merged", "solved", "decide_done", "done"}   # work that landed (a generator "accept" is only a proposal)
OPENED = {"pr"}                                          # a PR opened, not yet merged
FAIL = {"failed", "guard_blocked", "error", "lost_lease", "lease_expired"}
EMPTY = {"no_claims", "unchanged"}                       # ran, kept nothing
SERVICES = {   # what a healthy answer looks like for each live part (no secrets: unauthenticated probes)
    "connector (read)": ("https://connector.exagium.xyz/mcp", "POST", {200}),
    "act (write, needs token)": ("https://act.exagium.xyz/mcp", "POST", {401}),
    "workbench": ("https://workbench.exagium.xyz/signin", "GET", {200}),
    "chat (owner login)": ("https://chat.exagium.xyz/", "GET", {302, 403}),
}


def who(worker):
    """Worker classes, from the worker field's own prefix (free runs are named worker-<run id>)."""
    w = str(worker or "")
    if w.startswith("worker-"): return "free worker"
    if w.startswith("tier-"): return "tier (" + w[5:].split("-20")[0] + ")"
    if w.startswith("generator"): return "generator"
    if w.startswith(("claude", "chat")): return "Claude (chat/shift)"
    if w.startswith("pr-"): return "PR sync"
    return w.split("-")[0] or "unknown"


def ts(s):
    try: return datetime.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError: return None


def queue_part(q, now):
    items = q["items"]
    runs = [dict(r, item=i["id"]) for i in items for r in i.get("runs") or [] if ts(r.get("at"))]
    week = [r for r in runs if ts(r["at"]) >= now - timedelta(days=7)]
    by = defaultdict(Counter)
    for r in week:
        by[who(r.get("worker"))][r.get("outcome") or "none"] += 1
    days = []
    for d in range(13, -1, -1):
        day = (now - timedelta(days=d)).date()
        rs = [r for r in runs if ts(r["at"]).date() == day]
        n = lambda S, free=False: sum(r.get("outcome") in S and (not free or who(r.get("worker")) == "free worker") for r in rs)
        days.append({"day": day.isoformat(), "runs": len(rs), "finished": n(FINISHED), "prs": n(OPENED), "failed": n(FAIL),
                     "empty": n(EMPTY), "free_finished": n(FINISHED, True), "free_prs": n(OPENED, True)})
    recent = sorted(runs, key=lambda r: r["at"], reverse=True)[:40]
    openi = [i for i in items if i["status"] == "open"]
    return {
        "status_counts": dict(Counter(i["status"] for i in items)), "items": len(items),
        "workers_7d": {k: dict(v) for k, v in sorted(by.items(), key=lambda kv: -sum(kv[1].values()))},
        "daily_14d": days,
        "recent_events": [{"at": r["at"], "item": r["item"], "who": who(r.get("worker")), "outcome": r.get("outcome"),
                           "from": r.get("from"), "to": r.get("to"), "note": str(r.get("result") or "")[:160]} for r in recent],
        "open_items": [{"id": i["id"], "title": i.get("title", "")[:120], "kind": i.get("kind"), "value": i.get("value"),
                        "jobs": i.get("work_types") or []} for i in openi][:60],
        "waiting_on_owner": [{"id": i["id"], "title": i.get("title", "")[:120]} for i in items if i["status"] == "waiting on the owner"],
    }


def api(path, token):
    req = urllib.request.Request("https://api.github.com" + path, headers={
        "accept": "application/vnd.github+json", "user-agent": "forge-dash", **({"authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def github_part(token, now):
    out = {}
    try:
        prs = api(f"/repos/{REPO}/pulls?state=open&per_page=50", token)
        out["open_prs"] = [{"n": p["number"], "title": p["title"][:110], "by": p["user"]["login"], "at": p["created_at"]} for p in prs]
        closed = api(f"/repos/{REPO}/pulls?state=closed&sort=updated&direction=desc&per_page=100", token)
        wk = (now - timedelta(days=7)).isoformat()[:19]
        merged = [p for p in closed if (p.get("merged_at") or "") >= wk]
        out["merged_7d"] = len(merged)
        out["merged_7d_by"] = dict(Counter(p["user"]["login"] for p in merged))
    except Exception as e:   # the dashboard still shows the queue if GitHub is unreachable
        out["prs_error"] = str(e)[:160]
    try:
        runs = api(f"/repos/{REPO}/actions/runs?per_page=100", token)["workflow_runs"]
        day = (now - timedelta(days=1)).isoformat()[:19]
        per = defaultdict(Counter)
        for r in runs:
            if r["created_at"] >= day: per[r["name"]][r.get("conclusion") or r["status"]] += 1
        out["workflows_24h"] = {k: dict(v) for k, v in sorted(per.items())}
        out["latest_failures"] = [{"workflow": r["name"], "at": r["created_at"], "url": r["html_url"]}
                                  for r in runs if r.get("conclusion") == "failure"][:10]
    except Exception as e:
        out["runs_error"] = str(e)[:160]
    return out


def probe(url, method):
    req = urllib.request.Request(url, method=method, data=b'{"jsonrpc":"2.0","id":1,"method":"ping"}' if method == "POST" else None,
                                 headers={"content-type": "application/json", "user-agent": "forge-dash"})
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k): return None
    t = time.time()
    try:
        code = urllib.request.build_opener(NoRedirect).open(req, timeout=20).status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception:
        code = 0
    return code, round((time.time() - t) * 1000)


def main(qpath, out):
    now = datetime.now(timezone.utc)
    q = json.load(open(qpath))
    data = {"generated_at": now.isoformat()[:19] + "Z", "sources": {"queue": "audit/2026-11-17/queue.json on pilot/lgn-20261115",
            "github": "GitHub REST API (pulls, actions runs)", "services": "unauthenticated probes"}}
    data["queue"] = queue_part(q, now)
    data["github"] = github_part(os.environ.get("GITHUB_TOKEN"), now)
    data["services"] = {}
    for name, (url, method, ok) in SERVICES.items():
        code, ms = probe(url, method)
        data["services"][name] = {"http": code, "ms": ms, "ok": code in ok}
    models = os.path.join(os.path.dirname(out), "models.json")   # written by the model-ranking job, if it has run
    if os.path.exists(models):
        data["models"] = json.load(open(models))
    json.dump(data, open(out, "w"), indent=1)
    print(f"dash: {data['queue']['items']} items, {sum(d['runs'] for d in data['queue']['daily_14d'])} runs in 14 days, "
          f"services ok {sum(s['ok'] for s in data['services'].values())}/{len(SERVICES)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
