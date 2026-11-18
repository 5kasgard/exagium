"""Daily digest: the automatic handoff (queue P8). Facts by script; a short summary by a free model, checked by script.

python3 worker/digest.py facts MAIN_WT QUEUE_WT RUNS_WT OUT_DIR [--hours 24]   # no keys
python3 worker/digest.py summary OUT_DIR                                         # key: one free-model call
python3 worker/digest.py write RUNS_WT OUT_DIR                                   # no keys: commits digest/DATE.md

The facts section is the record: queue items that changed status, worker runs with their results, commits on main,
budget used per pool, and the open list in the queue's own order. The summary is labelled as model-drafted; any
sentence in it with a number but no evidence pointer (a path, run ID, commit or item ID) is flagged, as P8 asks.
"""
import datetime as dt, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import queue_ops as Q  # noqa: E402

SUMMARY_MODEL = ("cloudflare", "@cf/openai/gpt-oss-120b")
POINTER = re.compile(r"(\b[QP]\d+\b|\brun \d{6,}|\b\d{8,}\b|\b[0-9a-f]{7,40}\b|[\w./-]+/[\w.-]+|\w+\.(md|py|json|c|yml)\b)")


def git(wt, *a):
    return subprocess.run(["git", "-C", wt, *a], capture_output=True, text=True).stdout


def queue_at(wt, since):
    """The queue as it was at the last commit before `since` (None if none is in the fetched history)."""
    sha = git(wt, "rev-list", "-1", f"--before={since}", "HEAD").strip()
    if not sha:
        return None
    s = git(wt, "show", f"{sha}:{Q.QUEUE}")
    return json.loads(s) if s else None


def facts(main_wt, queue_wt, runs_wt, out, hours=24):
    t = Q.now()
    since = Q.iso(t - dt.timedelta(hours=hours))
    q = Q.load(queue_wt)
    old = queue_at(queue_wt, since)
    old_status = {i["id"]: i.get("status") for i in (old or {}).get("items", [])}
    changed = [(i["id"], old_status.get(i["id"], "new"), i.get("status"), i["title"][:110]) for i in q["items"]
               if old_status.get(i["id"], "new") != i.get("status")]
    runs = [(i["id"], r) for i in q["items"] for r in i.get("runs", []) if r.get("at", "") >= since]
    commits = git(main_wt, "log", f"--since={since}", "--format=%h %s").strip().splitlines()
    ledgers = {}
    for d in sorted({(t - dt.timedelta(hours=h)).strftime("%Y-%m-%d") for h in (0, hours)}):
        p = os.path.join(runs_wt, "ledger", d + ".json")
        if os.path.exists(p):
            ledgers[d] = {k: {x: v[x] for x in ("reserved", "calls", "tokens", "neurons")} for k, v in json.load(open(p)).items()}
    by = {i["id"]: i for i in q["items"]}
    open_list = [i for i in q["items"] if i.get("status") in ("open", "in progress")]
    lines = [f"# Digest {t.strftime('%Y-%m-%d')} (facts by script, {hours} h to {Q.iso(t)})", "",
             f"## Queue: status changes ({'baseline missing: history too shallow' if old is None else len(changed)})"]
    lines += [f"- {a}: {b} -> {c}. {d}" for a, b, c, d in changed] or ["- none"]
    lines += ["", f"## Worker runs ({len(runs)})"]
    lines += [f"- {a}: {r.get('result')} by {r.get('worker')} at {r.get('at')}; {r.get('where', '')} {('(lease lost)' if r.get('lost_lease') else '')}".rstrip()
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
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "facts.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    return 0


def metrics(queue_wt, since, q):
    """From the queue branch's own history: claims, double claims (an item claimed while another worker holds it),
    releases, expired leases; and from the items' run notes: results by kind and job type."""
    subj = git(queue_wt, "log", "--reverse", f"--since={since}", "--format=%s").splitlines()
    holder, claims, double, releases, expired = {}, 0, [], 0, 0
    for s_ in subj:
        m = re.match(r"worker: (worker-\d+) claims (\S+)", s_)
        if m:
            claims += 1
            if holder.get(m.group(2)) not in (None, m.group(1)):
                double.append(f"{m.group(2)} by {m.group(1)} while {holder[m.group(2)]} held it")
            holder[m.group(2)] = m.group(1)
            continue
        m = re.match(r"worker: (worker-\d+) (releases|records \(lease lost\)) (\S+)", s_)
        if m:
            releases += 1
            if holder.get(m.group(3)) == m.group(1):
                holder.pop(m.group(3))
            continue
        if s_.startswith("worker: leases expired"):
            expired += 1
            for x in re.findall(r"'([^']+)'", s_):   # an expired lease frees the item (review: no false double claims)
                holder.pop(x, None)
    results = {}
    for i in q["items"]:
        for r in i.get("runs", []):
            if r.get("at", "") >= since:
                results[str(r.get("result"))] = results.get(str(r.get("result")), 0) + 1
    return [f"- claims {claims}, releases {releases}, double claims {len(double)}{': ' + '; '.join(double) if double else ''}, "
            f"lease-expiry commits {expired}",
            f"- results: {', '.join(f'{k} {v}' for k, v in sorted(results.items())) or 'none'}",
            "- spot-check acceptance (Claude or the owner): recorded per item when checked; not measured by script"]


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
