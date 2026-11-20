"""Splitter (owner 2026-11-20: big items are not worker-sized). When no item is ready, break up to MAX_PARENTS open
build/fix items a day into at most 3 code sub-items each, every one sized for worker/code_job.py.

python3 -I worker/split.py QUEUE_WT REPO OUT_DIR   (needs CLAUDE_API_KEY and OPENROUTER_API_KEY; writes OUT_DIR/calls.jsonl)

Pipeline, the generator's pattern (worker/generate.py):
1. pick (script): open build/fix items with no job type, not blocked, not the owner's, never split before.
2. draft (Claude Haiku 5.5 on the owner's API credit): sub-items as JSON, given the item and the repository's code files
   with their line counts.
3. check (script): each sub-item names 1-2 existing files the code job may change (code_job.allowed, at most 600 lines)
   and an existing tests file; at most 3 per parent.
4. born "proposed"; the decision model (another family) gates them with two frozen controls, as the generator does.
Nothing the model writes is executed here; a sub-item's code runs only in the code job's no-secrets test job.
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "router"))
import queue_ops as Q
import generate as G
import code_job as CJ

MODEL = "claude-haiku-5-5"
MAX_PARENTS = 2
MAX_SUB = 3
DIRS = ("worker", "router", "runner", "facts", "transfer")

PROMPT = """Split one queue item into at most 3 small code changes, each doable by one model call that edits 1 or 2
existing files and keeps the tests passing. Only steps that move the item's NEXT STEP forward. If no part of the item can
be done this way (it needs a person, money, new infrastructure or data that does not exist), reply {{"subitems": []}}.

ITEM {id}: {title}
NEXT STEP: {next}

CODE FILES (path, lines):
{files}

TEST FILES: {tests}

Reply with JSON only: {{"subitems": [{{"title": "...", "files": ["path.py"], "test": "path/test_x.py",
"next": "exactly what to change and how to tell it is right"}}]}}"""

CONTROL_PARENT = ("Failure classes are listed in two places", "Expose the failure class names from worker/diagnose.py.")
CONTROL_GOOD = "Add classes() to worker/diagnose.py returning the sorted rule names plus 'unknown'"
CONTROL_BAD = "Rewrite the whole queue renderer in Rust and move the queue to a database"


def repo_files(repo):
    code, tests = [], []
    for d in DIRS:
        for root, _, fs in os.walk(os.path.join(repo, d)):
            for f in fs:
                if f.endswith(".py"):
                    p = os.path.relpath(os.path.join(root, f), repo)
                    (tests if re.match(r"test_", f) else code).append(p)
    lines = {p: open(os.path.join(repo, p), encoding="utf-8", errors="replace").read().count("\n") for p in code}
    return lines, sorted(tests)


def check(sub, lines, tests):
    if not isinstance(sub, dict) or not isinstance(sub.get("title"), str) or not isinstance(sub.get("next"), str):
        return None, "not the asked shape"
    files, test = sub.get("files"), sub.get("test")
    if not isinstance(files, list) or not 1 <= len(files) <= 2:
        return None, "1 or 2 files"
    for f in files:
        if f not in lines or not CJ.allowed(f) or lines[f] > CJ.MAX_LINES:
            return None, f"file not allowed, missing or too long: {f}"
    if test not in tests:
        return None, f"no such tests file: {test}"
    if not 10 <= len(sub["title"]) <= 160 or len(sub["next"]) < 20:
        return None, "title or next too short"
    return {"title": " ".join(sub["title"].split()), "files": files, "test": test, "next": sub["next"][:1500]}, None


def candidates(q):
    by = {i["id"]: i for i in q["items"]}
    used = {i.get("split_from") for i in q["items"] if i.get("split_from")}
    out = [i for i in q["items"] if i.get("status") == "open" and i.get("kind") in ("build", "fix") and not Q.job_type(i)
           and i["id"] not in used and i.get("needs") != "the owner"
           and all(by.get(b, {}).get("status") == "done" for b in i.get("blocked_by", []))]
    return sorted(out, key=lambda i: ({"high": 0, "medium": 1, "low": 2}.get(i.get("value"), 3), i["id"]))


def run(wt, repo, out, ask=None, decide_fn=None):
    os.makedirs(out, exist_ok=True)
    calls, log = [], []
    if ask is None:
        import engines

        def ask(prompt):
            r = engines.anthropic_call(MODEL, prompt, max_tokens=2000)
            calls.append({k: r.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage", "error")})
            return r.get("text")
    try:
        Q.sync(wt)
        q = Q.load(wt)
        day = Q.iso(Q.now())[:10]
        done_today = len({i.get("split_from") for i in q["items"] if i.get("split_from") and str(i.get("generated_at", "")).startswith(day)})
        lines, tests = repo_files(repo)
        listing = "\n".join(f"{p} ({n})" for p, n in sorted(lines.items()) if CJ.allowed(p))
        subs = []
        for parent in candidates(q)[:max(0, MAX_PARENTS - done_today)]:
            reply = ask(PROMPT.format(id=parent["id"], title=parent["title"], next=parent.get("next", "")[:3000],
                                      files=listing, tests=", ".join(tests)))
            m = re.search(r"\{.*\}", reply or "", re.S)
            try:
                d = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                d = {}
            for s in (d.get("subitems") or [])[:MAX_SUB]:
                ok, why = check(s, lines, tests)
                log.append({"parent": parent["id"], "ok": bool(ok), "why": why, "sub": s})
                if ok:
                    subs.append((parent, ok))
        if not subs:
            print("split: nothing passed the script check")
            return 0
        pairs = [(CONTROL_GOOD, *CONTROL_PARENT), (CONTROL_BAD, *CONTROL_PARENT)]
        pairs += [(f"{s['title']} (files: {', '.join(s['files'])})", p["title"], p.get("next", "")) for p, s in subs]
        ps, err = (decide_fn or (lambda pr: G.decide(pr, calls)))(pairs)
        outcomes = G.gate_outcomes(ps, len(subs))
        made = []
        for _ in range(Q.TRIES):
            Q.sync(wt)
            q = Q.load(wt)
            n = max([int(m.group(1)) for i in q["items"] for m in [re.fullmatch(r"K(\d+)", str(i["id"]))] if m] or [0]) + 1
            made = []
            for p, s in subs:
                kid = f"K{n}"; n += 1
                q["items"].append({"id": kid, "title": s["title"], "kind": "fix", "status": "proposed", "origin": "generated",
                                   "split_from": p["id"], "generated_at": Q.iso(Q.now()),
                                   "code": {"files": s["files"], "test": s["test"]}, "touches": list(s["files"]),
                                   "next": s["next"], "refs": [f"queue {p['id']}", "worker/split.py"], "severity": "minor",
                                   "needs": "free", "cost": 0, "value": p.get("value", "medium"), "blocked_by": [],
                                   "batch": p.get("batch", "B3")})
                made.append(kid)
            Q.save(wt, q)
            if Q.commit_push(wt, f"splitter: {', '.join(made)} proposed from {', '.join(sorted({p['id'] for p, _ in subs}))}"):
                break
        moved = G.move_all(wt, made, outcomes, ps) if outcomes else []
        log.append({"gate": {"p": ps, "error": err, "controls_ok": outcomes is not None}, "made": made, "moved": moved})
        print(f"split: proposed {made}; gate {'ran' if outcomes else 'withheld'} p={ps}; moved {moved}")
        return 0
    finally:
        json.dump(log, open(os.path.join(out, "split.json"), "w"), indent=1)
        with open(os.path.join(out, "calls.jsonl"), "w") as f:
            for c in calls:
                f.write(json.dumps(c) + "\n")


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2], sys.argv[3]))
