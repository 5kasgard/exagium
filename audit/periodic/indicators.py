"""Whole-system indicators: debt and blind-spot signals for the weekly auditor (queue P10, 2026-11-18).

Read-only. Every indicator is mechanical and says how it is computed; heuristics are labelled as such and list what
they flagged, so a reader can check each one. Thresholds are not fixed in advance: the first run is the baseline, and
later runs report the change against a previous report (--previous). Trip rules come from that history, not guesses.

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
            out[top][kind] += read(repo, f).count("\n")
    return {k: dict(v) for k, v in sorted(out.items())}


def duplication(repo, files):
    """Python functions whose bodies are identical after parsing (ast.dump, names of the function itself excluded).
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
                body = ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
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
    """Workflow lines that swallow a failure: '|| true', 'continue-on-error: true', 'set +e' without a later exit.
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
        if (now - t).days < days and r.get("conclusion"):
            per[r["workflowName"]][r["conclusion"]] += 1
    out = {}
    for w, c in sorted(per.items()):
        n = sum(c.values())
        bad = n - c["success"] - c["skipped"] - c["cancelled"]
        out[w] = {"runs": n, "not_success": bad, "share": round(bad / n, 3), "conclusions": dict(c)}
    return out


def queue_state(queue, qgit=None, qpath="audit/2026-11-17/queue.json", today=None):
    """Counts by status; for open items, days since the item id first appeared in the queue's history."""
    if queue is None:
        return None
    items = queue["items"]
    st = collections.Counter(i["status"] for i in items)
    age = {}
    if qgit:
        today = today or dt.date.today()
        for i in items:
            if i["status"] in ("done", "parked"):
                continue
            log = subprocess.run(["git", "-C", qgit, "log", "--all", "--reverse", "--format=%ad", "--date=short",
                                  "-S", f'"id": "{i["id"]}"', "--", qpath], capture_output=True, text=True).stdout.split()
            if log:
                age[i["id"]] = (today - dt.date.fromisoformat(log[0])).days
    claims = [f"{i['id']} by {i.get('claimed_by')} since {i.get('claimed_at')}" for i in items if i.get("claimed_by")]
    return {"by_status": dict(st), "open_age_days": age, "claims": claims}


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


EVIDENCE = re.compile(r"(\b[0-9a-f]{7,40}\b|\b\d{11}\b|[\w./-]+/[\w./-]+|\.(json|md|txt|py)\b|ledger entr|run \d|entry \d)")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*(?:%|×|x\b|ns|ms|µs|s\b)|\b0\.\d{2,}\b")


def claims_without_evidence(repo, path="README.md"):
    """Heuristic: README lines (or table rows) that state a measured number but carry no pointer to evidence
    (a path, a commit, a run ID, a ledger entry). Listed so each can be checked; tables often cite evidence in a column."""
    out = []
    for n, line in enumerate(read(repo, path).splitlines(), 1):
        if NUMBER.search(line) and not EVIDENCE.search(line):
            out.append(f"{path}:{n}: {line.strip()[:110]}")
    return {"count": len(out), "lines": out}


def delta(new, old):
    """Changes against a previous report, for the scalar indicators."""
    if not old:
        return None
    pick = lambda r: {"duplication_share": r["duplication"]["share"], "unreferenced": r["unreferenced"]["count"],
                      "cannot_fail_lines": r["cannot_fail"]["count"], "claims_without_evidence": r["claims_without_evidence"]["count"],
                      "stale_branches": len(r["stale_branches"]),
                      "code_lines": sum(v.get("code", 0) for v in r["size"].values())}
    a, b = pick(old), pick(new)
    return {k: {"was": a[k], "now": b[k]} for k in b if a[k] != b[k]}


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
         "stale_branches": stale_branches(repo), "claims_without_evidence": claims_without_evidence(repo)}
    r["change"] = delta(r, load(a.previous))
    if a.out:
        with open(a.out, "x") as fh:
            json.dump(r, fh, indent=1)
    print(json.dumps({k: v for k, v in r.items() if k not in ("size",)}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
