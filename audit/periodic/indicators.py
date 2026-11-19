"""Whole-system indicators: debt and blind-spot signals for the weekly auditor (queue P10, 2026-11-18).

Read-only. Every indicator is mechanical and says how it is computed; heuristics are labelled as such and list what
they flagged, so a reader can check each one. Thresholds are not fixed in advance: the first run is the baseline, and
later runs report the change against a previous report (--previous). Trip rules come from that history, not guesses.

Not yet measured (each needs a tool installed; queue P10): test coverage, lint warnings, dependency freshness.
Usage: python3 audit/periodic/indicators.py [--repo .] [--queue QUEUE.json] [--queue-git DIR] [--runs RUNS.json]
                                            [--previous OLD.json] [--out NEW.json]
  --queue      the queue file (it lives on the pilot branch: git show origin/pilot/lgn-20261115:audit/2026-11-17/queue.json)
  --queue-git  a clone whose history contains that queue file, to date each item's first appearance
  --runs       output of: gh run list --limit 300 --json workflowName,conclusion,createdAt,event
"""
import argparse, ast, collections, datetime as dt, hashlib, json, os, re, subprocess, sys

CODE = (".py", ".c", ".h", ".sh")
TEXT = CODE + (".md", ".yml", ".yaml", ".json", ".txt", ".toml", ".cfg")
DATA_BRANCHES = {"main", "router-runs", "ci-runs", "HEAD"}


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=True).stdout


def tracked(repo):
    return [f for f in git(repo, "ls-files").splitlines() if os.path.isfile(os.path.join(repo, f))]


def read(repo, f):
    try:
        with open(os.path.join(repo, f), encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def size(repo, files):
    """Lines per top-level folder and kind (code / docs / workflows / data)."""
    out = collections.defaultdict(lambda: collections.Counter())
    for f in files:
        top = f.split("/")[0] if "/" in f else "(root)"
        kind = ("workflows" if f.startswith(".github/") else "code" if f.endswith(CODE) else
                "docs" if f.endswith(".md") else "data")
        if f.endswith(TEXT):
            out[top][kind] += len(read(repo, f).splitlines())
    return {k: dict(v) for k, v in sorted(out.items())}


def duplication(repo, files):
    """Python functions whose bodies are identical after parsing (ast.dump, without the function's name or docstring).
    Method: exact structural duplicates only; near-duplicates are not counted. Share = duplicated function lines / all function lines."""
    seen = collections.defaultdict(list)
    total = 0
    for f in files:
        if not f.endswith(".py"):
            continue
        try:
            tree = ast.parse(read(repo, f))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n = (node.end_lineno or node.lineno) - node.lineno + 1
                total += n
                stmts = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                                          and isinstance(node.body[0].value.value, str)) else node.body
                body = ast.dump(ast.Module(body=stmts, type_ignores=[]), include_attributes=False)
                if n >= 4:
                    seen[hashlib.sha256(body.encode()).hexdigest()].append((f, node.name, n))
    groups = [g for g in seen.values() if len(set(x[0] for x in g)) > 1]
    dup_lines = sum(sum(x[2] for x in g[1:]) for g in groups)
    return {"function_lines": total, "duplicated_lines": dup_lines,
            "share": round(dup_lines / total, 4) if total else 0.0,
            "groups": [[f"{f}:{name}" for f, name, _ in g] for g in sorted(groups, key=lambda g: -g[0][2])][:20]}


def unreferenced(repo, files):
    """Heuristic: code files whose name (with or without extension) appears in no other tracked text file.
    Entry points run by hand can be flagged wrongly; the list is printed so each can be checked."""
    texts = {f: read(repo, f) for f in files if f.endswith(TEXT)}
    out = []
    for f in files:
        if not f.endswith(CODE) or os.path.basename(f).startswith("test_"):
            continue
        base = os.path.basename(f)
        stem = os.path.splitext(base)[0]
        pat = re.compile(r"(?<![\w-])" + re.escape(stem) + r"(?![\w-])")
        if not any((base in t or pat.search(t)) for g, t in texts.items() if g != f):
            out.append(f)
    return {"count": len(out), "files": out}


def cannot_fail(repo, files):
    """Workflow lines that swallow a failure: '|| true' and 'continue-on-error: true'. ('set +e' is not counted: the
    repo uses it deliberately to record an exit code and return it later.)
    Some are deliberate diagnostics; each is listed with its line so a reader can tell which."""
    hits = []
    for f in files:
        if not (f.startswith(".github/workflows/") and f.endswith((".yml", ".yaml"))):
            continue
        for n, line in enumerate(read(repo, f).splitlines(), 1):
            s = line.strip()
            if "|| true" in s or re.match(r"continue-on-error:\s*true", s):
                hits.append(f"{f}:{n}: {s[:100]}")
    return {"count": len(hits), "lines": hits}


def ci(runs, days=7, now=None):
    """Share of completed runs per workflow that did not succeed, over the last `days` days."""
    if runs is None:
        return None
    now = now or dt.datetime.now(dt.timezone.utc)
    per = collections.defaultdict(collections.Counter)
    for r in runs:
        t = dt.datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00"))
        if now - t < dt.timedelta(days=days) and r.get("conclusion"):
            per[r["workflowName"]][r["conclusion"]] += 1
    out = {}
    for w, c in sorted(per.items()):
        n = sum(c.values())
        bad = n - c["success"] - c["skipped"] - c["cancelled"]
        out[w] = {"runs": n, "not_success": bad, "share": round(bad / n, 3), "conclusions": dict(c)}
    return out


def queue_state(queue, qgit=None, qpath="audit/2026-11-17/queue.json", now=None):
    """Counts by status; for open items, hours since the item id first appeared in the queue's history.
    Uses commit timestamps (Unix time), so the commit's own timezone cannot shift the age (a first run in local
    dates read 0 days for items about 12 hours old)."""
    if queue is None:
        return None
    items = queue["items"]
    st = collections.Counter(i["status"] for i in items)
    age = {}
    if qgit:
        now = now or dt.datetime.now(dt.timezone.utc).timestamp()
        for i in items:
            if i["status"] in ("done", "parked"):
                continue
            log = subprocess.run(["git", "-C", qgit, "log", "--all", "--reverse", "--format=%ct",
                                  "-S", f'"id": "{i["id"]}"', "--", qpath], capture_output=True, text=True).stdout.split()
            if log:
                age[i["id"]] = round((now - int(log[0])) / 3600, 1)
    claims = [f"{i['id']} by {i.get('claimed_by')} since {i.get('claimed_at')}" for i in items if i.get("claimed_by")]
    return {"by_status": dict(st), "open_age_hours": age, "claims": claims}


def flow(queue, qgit, qpath="audit/2026-11-17/queue.json", days=7, now=None):
    """Arrivals and closures over the window, from every version of the queue file; median hours from an item's first
    appearance to its first 'done'; Little's law estimate of waiting time (open items / closures per day).
    Arrivals are split by the item's `origin`: follow-up (spawned by finishing or checking work: depth) or new
    (a new direction: breadth); items from before the field existed count as unknown."""
    if queue is None or not qgit:
        return None
    now = now or dt.datetime.now(dt.timezone.utc).timestamp()
    since = now - days * 86400
    log = subprocess.run(["git", "-C", qgit, "log", "--all", "--reverse", "--format=%H %ct", "--", qpath],
                         capture_output=True, text=True).stdout.split("\n")
    first, done = {}, {}
    for line in log:
        if not line.strip():
            continue
        h, t = line.split()
        t = int(t)
        try:
            q = json.loads(subprocess.run(["git", "-C", qgit, "show", f"{h}:{qpath}"], capture_output=True, text=True).stdout)
        except ValueError:
            continue
        for i in q.get("items", []):
            first.setdefault(i["id"], t)
            if i.get("status") == "done":
                done.setdefault(i["id"], t)
    origin = {i["id"]: i.get("origin", "unknown") for i in queue["items"]}
    arrived = [k for k, t in first.items() if t >= since]
    closed = [k for k, t in done.items() if t >= since]
    cycle = sorted(round((done[k] - first[k]) / 3600, 1) for k in closed)
    open_now = sum(1 for i in queue["items"] if i["status"] not in ("done", "parked"))
    per_day = len(closed) / days
    return {"window_days": days, "arrived": len(arrived),
            "arrived_by_origin": dict(collections.Counter(origin.get(k, "unknown") for k in arrived)),
            "closed": len(closed), "closed_per_day": round(per_day, 2),
            "median_hours_to_done": cycle[len(cycle) // 2] if cycle else None,
            "open_now": open_now, "littles_wait_days": round(open_now / per_day, 1) if per_day else None}


def connectors(repo, manifest="audit/periodic/connectors.json"):
    """Which connectors a real part has gone through end to end. The manifest names, per connector, the evidence
    paths; a connector counts as used only if every path exists in the repo and is not empty. Hand-kept list,
    checked by script: a stale or wrong path shows up as unused rather than passing silently."""
    path = os.path.join(repo, manifest)
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    out = {}
    for name, c in m["connectors"].items():
        ev = c.get("evidence", [])
        ok = bool(ev) and all(os.path.exists(os.path.join(repo, e)) and (os.path.isdir(os.path.join(repo, e)) or
                              os.path.getsize(os.path.join(repo, e)) > 0) for e in ev)
        out[name] = "used" if ok else "not yet"
    return {"used": sum(v == "used" for v in out.values()), "total": len(out), "by_connector": out}


def stale_branches(repo, days=14):
    """Remote branches, other than main and the data branches, whose last commit is older than `days`."""
    out = []
    now = dt.datetime.now(dt.timezone.utc)
    for line in git(repo, "for-each-ref", "--format=%(refname:short) %(committerdate:iso-strict)", "refs/remotes/origin").splitlines():
        name, when = line.split(" ", 1)
        b = name.split("/", 1)[1] if "/" in name else name
        if b in DATA_BRANCHES:
            continue
        age = (now - dt.datetime.fromisoformat(when)).days
        if age > days:
            out.append(f"{b} ({age} days)")
    return out


EVIDENCE = re.compile(r"(https?://\S+|\]\([^)]+\)|\b[0-9a-f]{7,40}\b|\b\d{11}\b|[\w./-]+/[\w./-]+|\.(json|md|txt|py)\b|ledger entr|run \d|entry \d)")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*(?:%|×|x\b|ns|ms|µs|s\b)|\b0\.\d{2,}\b")


def claims_without_evidence(repo, path="README.md"):
    """Heuristic: README lines (or table rows) that state a measured number but carry no pointer to evidence
    (a path, a commit, a run ID, a ledger entry). Listed so each can be checked; tables often cite evidence in a column."""
    out = []
    for n, line in enumerate(read(repo, path).splitlines(), 1):
        if NUMBER.search(line) and not EVIDENCE.search(line):
            out.append(f"{path}:{n}: {line.strip()[:110]}")
    return {"count": len(out), "lines": out}


def churn(repo, days=7):
    """Lines added plus deleted per top-level folder over the last `days` days (git log --numstat): where the work went.
    A direction signal: compare it with what the README says the project is for."""
    out = collections.defaultdict(collections.Counter)
    for line in git(repo, "log", f"--since={days}.days", "--numstat", "--format=").splitlines():
        p = line.split("\t")
        if len(p) == 3 and p[0].isdigit() and p[1].isdigit():
            f = p[2]
            kind = ("workflows" if f.startswith(".github/") else "code" if f.endswith(CODE) else
                    "docs" if f.endswith(".md") else "data")
            out[f.split("/")[0] if "/" in f else "(root)"][kind] += int(p[0]) + int(p[1])
    return {k: dict(v) for k, v in sorted(out.items(), key=lambda kv: -kv[1]["code"])}


def delta(new, old):
    """Changes against a previous report, for the scalar indicators."""
    if not old:
        return None
    pick = lambda r: {"duplication_share": r["duplication"]["share"], "unreferenced": r["unreferenced"]["count"],
                      "cannot_fail_lines": r["cannot_fail"]["count"], "claims_without_evidence": r["claims_without_evidence"]["count"],
                      "stale_branches": len(r["stale_branches"]),
                      "code_lines": sum(v.get("code", 0) for v in r["size"].values()),
                      "ci_not_success_runs": sum(v["not_success"] for v in (r["ci_7d"] or {}).values()),
                      "open_items": (r["queue"] or {}).get("by_status", {}).get("open"),
                      "littles_wait_days": (r.get("flow_7d") or {}).get("littles_wait_days"),
                      "connectors_used": (r.get("connectors") or {}).get("used")}
    a, b = pick(old), pick(new)
    return {k: {"was": a.get(k), "now": b[k]} for k in b if a.get(k) != b[k]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--queue")
    ap.add_argument("--queue-git")
    ap.add_argument("--runs")
    ap.add_argument("--previous")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    repo = os.path.abspath(a.repo)
    files = tracked(repo)
    load = lambda p: json.load(open(p)) if p else None
    r = {"tool": "audit/periodic/indicators.py", "commit": git(repo, "rev-parse", "HEAD").strip(),
         "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "tracked_files": len(files),
         "size": size(repo, files), "duplication": duplication(repo, files), "unreferenced": unreferenced(repo, files),
         "cannot_fail": cannot_fail(repo, files), "ci_7d": ci(load(a.runs)), "queue": queue_state(load(a.queue), a.queue_git),
         "stale_branches": stale_branches(repo), "claims_without_evidence": claims_without_evidence(repo),
         "churn_7d": churn(repo), "flow_7d": flow(load(a.queue), a.queue_git), "connectors": connectors(repo)}
    r["change"] = delta(r, load(a.previous))
    if a.out:
        with open(a.out, "x") as fh:
            json.dump(r, fh, indent=1)
    print(json.dumps({k: v for k, v in r.items() if k not in ("size",)}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
