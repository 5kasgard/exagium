"""Daily digest: the automatic handoff (queue P8). Facts by script; a short summary by a free model, checked by script.

python3 worker/digest.py facts MAIN_WT QUEUE_WT RUNS_WT OUT_DIR [--hours 24]   # no keys
python3 worker/digest.py summary OUT_DIR                                         # key: one free-model call
python3 worker/digest.py write RUNS_WT OUT_DIR                                   # no keys: commits digest/DATE.md

The facts section is the record: queue items that changed status, worker runs with their results, commits on main,
budget used per pool, and the open list in the queue's own order. The summary is labelled as model-drafted; any
sentence in it with a number but no evidence pointer (a path, run ID, commit or item ID) is flagged, as P8 asks.
"""
import datetime as dt, json, os, re, subprocess, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import queue_ops as Q  # noqa: E402

SUMMARY_MODEL = ("cloudflare", "@cf/openai/gpt-oss-120b")
POINTER = re.compile(r"(\b[QP]\d+\b|\brun \d{6,}|\b\d{8,}\b|\b[0-9a-f]{7,40}\b|[\w./-]+/[\w.-]+|\w+\.(md|py|json|c|yml)\b)")


def git(wt, *a):
    return subprocess.run(["git", "-C", wt, *a], capture_output=True, text=True).stdout


def queue_at(wt, since):
    """(queue, label): the queue at the last commit before `since`; if the queue file didn't exist then, its oldest
    version inside the window (digest 2026-11-18: the queue was younger than the window, so every item read as new)."""
    sha = git(wt, "rev-list", "-1", f"--before={since}", "HEAD").strip()
    s = git(wt, "show", f"{sha}:{Q.QUEUE}") if sha else ""
    if s:
        return json.loads(s), "the window's start"
    for sha in git(wt, "rev-list", "--reverse", f"--since={since}", "HEAD", "--", Q.QUEUE).split():
        s = git(wt, "show", f"{sha}:{Q.QUEUE}")
        if s:
            return json.loads(s), f"its first version in the window ({sha[:7]})"
    return None, "none found"


def facts(main_wt, queue_wt, runs_wt, out, hours=24):
    t = Q.now()
    since = Q.iso(t - dt.timedelta(hours=hours))
    q = Q.load(queue_wt)
    old, base = queue_at(queue_wt, since)
    old_status = {i["id"]: i.get("status") for i in (old or {}).get("items", [])}
    changed = [(i["id"], old_status.get(i["id"], "new"), i.get("status"), i["title"][:110]) for i in q["items"]
               if old_status.get(i["id"], "new") != i.get("status")]
    runs = [(i["id"], r) for i in q["items"] for r in i.get("runs", []) if r.get("at", "") >= since]
    commits = git(main_wt, "log", f"--since={since}", "--format=%h %s").strip().splitlines()
    ledgers = {}
    day = (t - dt.timedelta(hours=hours)).date()
    while day <= t.date():
        d = day.strftime("%Y-%m-%d")
        p = os.path.join(runs_wt, "ledger", d + ".json")
        if os.path.exists(p):
            ledgers[d] = {k: {x: v.get(x, 0) for x in ("reserved", "calls", "tokens", "neurons", "usd")} for k, v in json.load(open(p)).items()}
        day += dt.timedelta(days=1)
    by = {i["id"]: i for i in q["items"]}
    open_list = [i for i in q["items"] if i.get("status") in ("open", "in progress")]
    lines = [f"# Digest {t.strftime('%Y-%m-%d')} (facts by script, {hours} h to {Q.iso(t)})", "",
             f"## Queue: status changes ({len(changed)}, compared with {base})"]
    lines += [f"- {a}: {b} -> {c}. {d}" for a, b, c, d in changed] or ["- none"]
    lines += ["", f"## Worker runs ({len(runs)})"]
    lines += [f"- {a}: {r.get('result')} by {r.get('worker') or 'a chat'} at {r.get('at')}; {r.get('where', '')} {('(lease lost)' if r.get('lost_lease') else '')}".rstrip()
              for a, r in runs] or ["- none"]
    lines += ["", f"## Commits on main ({len(commits)})"] + ([f"- {c}" for c in commits] or ["- none"])
    lines += ["", "## Budget used (ledger on worker-runs)"]
    lines += [f"- {d} {k}: {v}" for d, l in ledgers.items() for k, v in l.items()] or ["- no worker spend recorded"]
    lines += ["", "## Open list, in the queue's order (first 12)"]
    for i in open_list[:12]:
        waits = [b for b in i.get("blocked_by", []) if by.get(b, {}).get("status") != "done"]
        lines.append(f"- {i['id']} [{i.get('status')}{', claimed by ' + i['claimed_by'] if i.get('claimed_by') else ''}"
                     f"{', waits on ' + ', '.join(waits) if waits else ''}{', worker-ready' if i.get('targets') and not waits else ''}]: {i['title'][:120]}")
    lines += ["", "## Worker measures (queue P9: measured by script every day)"] + metrics(queue_wt, since, q)
    lines += ["", "## Loop measures (owner 2026-11-20: is it getting faster, and who still has to step in?)"] + loop_metrics(q, since, ledgers)
    dpath = os.path.join(out, "diagnose.json")
    if os.path.exists(dpath):
        import diagnose
        lines += ["", "## Failed runs by class (worker/diagnose.py; what the system does with each)"] + diagnose.report(json.load(open(dpath)))
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "facts.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    return 0


def metrics(queue_wt, since, q):
    """From the items' run records: claims, double claims (an item worked by multiple workers),
    releases, expired leases; and results by kind."""
    claims, releases, expired = 0, 0, 0
    double = []
    results = {}
    worker_runs_by_item = defaultdict(list)
    for i in q["items"]:
        for r in i.get("runs", []):
            if r.get("at", "") >= since:
                worker = r.get("worker")
                if worker and str(worker).startswith("worker-"):
                    worker_runs_by_item[i["id"]].append(r)
                results[str(r.get("result"))] = results.get(str(r.get("result")), 0) + 1
    for item_id, runs in worker_runs_by_item.items():
        claims += len(runs)
        workers = set(r.get("worker") for r in runs)
        if len(workers) > 1:
            double.append(f"{item_id}: multiple workers ({', '.join(sorted(workers))})")
        for r in runs:
            if r.get("lost_lease"):
                expired += 1
            else:
                releases += 1
    return [f"- claims {claims}, releases {releases}, double claims {len(double)}{': ' + '; '.join(double) if double else ''}, "
            f"lease-expiry commits {expired}",
            f"- results: {', '.join(f'{k} {v}' for k, v in sorted(results.items())) or 'none'}",
            "- spot-check acceptance (Claude or the owner): recorded per item when checked; not measured by script"]


LANDED = {"merged", "confirm", "solved", "decide_done"}
PAID_POOLS = ("openrouter_paid", "anthropic")


def actor(w):
    w = str(w or "")
    return ("free worker" if w.startswith("worker-") else "Claude API tier" if w.startswith("tier-")
            else "generator" if w.startswith("generator") else "Claude session" if w.startswith("claude") else "owner or chat")


def idle_runs(since):
    """Worker runs in the window that claimed nothing although not stopped for budget (needs GH_TOKEN; else None)."""
    if not os.environ.get("GH_TOKEN"):
        return None
    repo = os.environ.get("GITHUB_REPOSITORY", "5kasgard/exagium")
    ids = subprocess.run(["gh", "api", f"repos/{repo}/actions/workflows/worker.yml/runs?per_page=100&created=>={since[:10]}",
                          "-q", f'.workflow_runs[] | select(.created_at >= "{since}") | .id'], capture_output=True, text=True).stdout.split()
    n = 0
    for rid in ids:
        jobs = subprocess.run(["gh", "api", f"repos/{repo}/actions/runs/{rid}/jobs", "-q",
                               '[.jobs[] | select(.name=="land" or .name=="idle") | "\(.name)=\(.conclusion)"] | join(",")'],
                              capture_output=True, text=True).stdout
        n += "land=skipped" in jobs and "idle=" in jobs and "idle=skipped" not in jobs
    return n, len(ids)


def loop_metrics(q, since, ledgers):
    """Throughput, who acted, outcome classes, spend and cost per landed item, idle runs with budget left. By script."""
    acts, outcomes, landed = {}, {}, 0
    for i in q["items"]:
        for r in i.get("runs", []):
            if r.get("at", "") >= since:
                a = actor(r.get("worker"))
                acts[a] = acts.get(a, 0) + 1
                o = r.get("outcome") or "none"
                outcomes[o] = outcomes.get(o, 0) + 1
                landed += o in LANDED
    usd = sum(float((v or {}).get("usd", 0) or 0) for l in ledgers.values() for k, v in l.items() if k in PAID_POOLS)
    calls = {}
    for l in ledgers.values():
        for k, v in l.items():
            calls[k] = calls.get(k, 0) + int((v or {}).get("calls", 0) or 0)
    idle = idle_runs(since)
    return [f"- items landed (merged, confirmed, solved or decided done): {landed}",
            f"- moves by actor: {', '.join(f'{k} {v}' for k, v in sorted(acts.items())) or 'none'}",
            f"- outcomes: {', '.join(f'{k} {v}' for k, v in sorted(outcomes.items())) or 'none'}",
            f"- paid spend: ${usd:.4f}" + (f" (${usd / landed:.4f} per landed item)" if landed else "")
            + f"; calls by pool: {', '.join(f'{k} {v}' for k, v in sorted(calls.items())) or 'none'}",
            f"- worker runs that found no ready item with budget left: " + (f"{idle[0]} of {idle[1]}" if idle else "not measured (no GH_TOKEN)")]


def flag(summary):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", summary or ""):
        if re.search(r"\d", s) and not POINTER.search(s):
            out.append(s.strip())
    return out


def summary(out):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "router"))
    import engines
    f = open(os.path.join(out, "facts.md")).read()
    prompt = ("Write a short handoff (at most 8 sentences) from these facts only: what changed, what is blocked or stuck, "
              "and what the next three items are. Cite the item ID, run ID, commit or file for every number. Add nothing "
              "the facts don't say.\n\n" + f)
    r = engines.call(*SUMMARY_MODEL, prompt, max_tokens=3000)
    json.dump({k: r.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage", "error", "text")},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    with open(os.path.join(out, "calls.jsonl"), "w") as fh:
        fh.write(json.dumps({k: r.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage")}) + "\n")
    print("summary ok", r["ok"], r["status"])
    return 0


def write(runs_wt, out):
    f = open(os.path.join(out, "facts.md")).read()
    s = {}
    if os.path.exists(os.path.join(out, "summary.json")):
        s = json.load(open(os.path.join(out, "summary.json")))
    text = (s.get("text") or "").strip()
    flagged = flag(text)
    body = f + "\n## Summary (drafted by " + (s.get("model") or "no model") + "; not checked by a person)\n\n" + \
        (text or "_no summary: " + str(s.get("error") or "the summary job did not run") + "_") + "\n"
    if flagged:
        body += "\n**Flagged: a number with no evidence pointer:**\n" + "\n".join(f"- {x}" for x in flagged) + "\n"
    day = Q.now().strftime("%Y-%m-%d")
    for _ in range(Q.TRIES):
        subprocess.run(["git", "-C", runs_wt, "fetch", "-q", "--depth", "5", "origin", "worker-runs"], capture_output=True)
        subprocess.run(["git", "-C", runs_wt, "reset", "-q", "--hard", "FETCH_HEAD"], capture_output=True)
        p = os.path.join(runs_wt, "digest", day + ".md")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(body)
        subprocess.run(["git", "-C", runs_wt, "add", p], check=True)
        subprocess.run(["git", "-C", runs_wt, "-c", "user.name=exagium-worker", "-c", "user.email=worker@exagium.invalid",
                        "commit", "-qm", f"digest {day}"], check=True)
        if subprocess.run(["git", "-C", runs_wt, "push", "-q", "origin", "HEAD:worker-runs"]).returncode == 0:
            print(f"digest/{day}.md written; {len(flagged)} sentence(s) flagged")
            return 0
    return 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "facts":
        h = int(a[a.index("--hours") + 1]) if "--hours" in a else 24
        sys.exit(facts(a[1], a[2], a[3], a[4], h))
    sys.exit({"summary": lambda: summary(a[1]), "write": lambda: write(a[1], a[2])}[a[0]]())
