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
MAX_PARENTS = 6   # items asked per run (about $0.0007 each); an item judged not worker-sized is skipped for RETRY_DAYS
RETRY_DAYS = 7
MAX_SUB = 3
DIRS = ("worker", "router", "runner", "facts", "transfer", "experiments")

PROMPT = """Split one queue item into at most 3 small code changes, each doable by one model call that edits 1 or 2
files (existing, or at most one new file of under 80 lines in an existing folder: the merge gate takes 80 changed lines) and keeps the tests passing. Only steps that move the item's NEXT STEP forward. If no part of the item can
be done this way (it needs a person, money, new infrastructure or data that does not exist), reply {{"subitems": []}}.

ITEM {id}: {title}
NEXT STEP: {next}

THE PARTS MOST RELATED TO THIS ITEM (module purpose, functions with their first docstring line, and the tests that
cover them; from worker/repo_index.py):
{parts}

ALL CODE FILES (path, lines):
{files}

TEST FILES: {tests}

Each sub-item edits its 1-2 files and may add tests to its test file. Name up to 2 other files the coder must read to
get data formats and interfaces right ("context"); it sees nothing else. Say in "next" which test to add.

A sub-item that creates a file lists it in "files" and in "new". Building something new is fine when it is small: a new
module of a few functions, with its tests, is one sub-item.

If the item is a measurement or test with a rule (an experiment), one sub-item may write experiments/<name>.py (new) with
PLAN (what is measured, on what, and the pass rule, in words), run() returning a JSON-able dict and rule(result) returning
{{"verdict": "holds"|"fails"|"invalid", "why": "..."}}, test experiments/test_experiments.py; give it "experiment": "<name>"
(lowercase letters, digits, underscores). Once it lands, the experiment runs by itself with no secrets.

Reply with JSON only: {{"subitems": [{{"title": "...", "files": ["path.py"], "new": [], "test": "path/test_x.py", "context": ["path.py"],
"next": "exactly what to change, and the test that shows it is right"}}]}}"""

ASK = ("Pair {k}: is its SUBITEM a correct, self-contained small code change that moves its ITEM's NEXT STEP forward, sized "
       "for one model call that edits the 1 or 2 files named? Vague, off-topic, too large, or not what the item needs: no.")
# controls (2026-11-20): the first live gate used the research generator's question ("would published research ... inform"),
# which does not fit code tasks: the good control scored 0.19 and every sub-item 0.07-0.26
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


def first_json(text):
    """The first JSON object in a reply that parses (a greedy {...} match failed on 4 of 6 replies on 2026-11-21, run
    44820822737: replies with prose or a second brace after the object). {} if none."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            d, _ = dec.raw_decode(text[m.start():])
            if isinstance(d, dict) and "subitems" in d:
                return d
        except json.JSONDecodeError:
            continue
    return {}


def check(sub, lines, tests):
    if not isinstance(sub, dict) or not isinstance(sub.get("title"), str) or not isinstance(sub.get("next"), str):
        return None, "not the asked shape"
    files, test = sub.get("files"), sub.get("test")
    if not isinstance(files, list) or not 1 <= len(files) <= 2:
        return None, "1 or 2 files"
    new = sub.get("new") or []
    if not isinstance(new, list) or len([f for f in new if f != test]) > 1 or any(f not in files and f != test for f in new):
        return None, "at most one new file (plus a new tests file), each in files or the test"
    dirs = {os.path.dirname(p) for p in lines}
    for f in files:
        if f in new:   # owner 2026-11-21: the worker builds, not only edits; a new file in a folder that already exists
            if not isinstance(f, str) or f in lines or not CJ.allowed(f) or not f.endswith(CJ.NEW_EXT) or os.path.dirname(f) not in dirs:
                return None, f"new file not allowed: {f}"
        elif f not in lines or not CJ.allowed(f) or lines[f] > CJ.MAX_LINES:
            return None, f"file not allowed, missing or too long: {f}"
    if test not in tests and not (test in new and isinstance(test, str) and CJ.TEST.fullmatch(test) and CJ.allowed(test)
                                  and os.path.dirname(test) in dirs):
        return None, f"no such tests file: {test}"
    if not 10 <= len(sub["title"]) <= 160 or len(sub["next"]) < 20:
        return None, "title or next too short"
    ctx = [c for c in (sub.get("context") or []) if isinstance(c, str) and c in lines and CJ.allowed(c) and c not in files][:2]
    exp = sub.get("experiment")
    if exp is not None and not (isinstance(exp, str) and re.fullmatch(r"[a-z0-9_]+", exp) and f"experiments/{exp}.py" in new):
        return None, "an experiment sub-item must create experiments/<name>.py"
    return {"title": " ".join(sub["title"].split()), "files": files, "new": new, "test": test, "context": ctx, "next": sub["next"][:1500],
            "experiment": exp}, None


def recently_tried(i):
    import datetime as dt
    t = i.get("split_tried")
    if not t:
        return False
    return dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(t.replace("Z", "+00:00")) < dt.timedelta(days=RETRY_DAYS)


def mark_tried(wt, ids):
    """Record on each item that the splitter found nothing worker-sized in it (not a status change)."""
    if not ids:
        return
    for _ in range(Q.TRIES):
        Q.sync(wt)
        q = Q.load(wt)
        for i in q["items"]:
            if i["id"] in ids:
                i["split_tried"] = Q.iso(Q.now())
        Q.save(wt, q)
        if Q.commit_push(wt, f"splitter: nothing worker-sized in {', '.join(sorted(ids))} (retry after {RETRY_DAYS} days)"):
            return


SPLITTABLE = ("build", "code_change", "experiment")   # worker/classify.py types the splitter can move a rung
NOT_NOW = ("held", "owner_action", "already_done", "sandbox_or_workflow")


def splittable(i):
    """By the classifier's types when it has judged the item (owner 2026-11-21: work types, not three job kinds); by the
    item's kind otherwise."""
    wt = i.get("work_types")
    if wt:
        top = max(wt.items(), key=lambda kv: kv[1])[0]
        return any(wt.get(t, 0) >= 0.5 for t in SPLITTABLE) and top not in NOT_NOW
    return i.get("kind") in ("build", "fix")


def candidates(q):
    by = {i["id"]: i for i in q["items"]}
    kids = {}
    for i in q["items"]:
        if i.get("split_from"):
            kids.setdefault(i["split_from"], []).append(i.get("status"))
    # a parent is split again once every earlier sub-item is finished (done or parked): its next step, not a repeat
    busy = {p for p, ss in kids.items() if any(s not in ("done", "parked") for s in ss)}
    out = [i for i in q["items"] if i.get("status") == "open" and splittable(i) and not Q.job_type(i)
           and not i.get("experiment") and i["id"] not in busy and i.get("needs") != "the owner" and not recently_tried(i)
           and all(by.get(b, {}).get("status") == "done" for b in i.get("blocked_by", []))]
    return sorted(out, key=lambda i: ({"high": 0, "medium": 1, "low": 2}.get(i.get("value"), 3), i["id"]))


def gate_fn(calls):
    return lambda pr: G.decide(pr, calls, ask=ASK, label="SUBITEM")


def regate(wt, decide_fn, calls):
    """Sub-items still 'proposed' (a withheld gate) are gated again."""
    Q.sync(wt)
    q = Q.load(wt)
    by = {i["id"]: i for i in q["items"]}
    follow = []   # a run item follows its code item's gate: accepted with it, rejected with it
    for i in q["items"]:
        if i.get("status") == "proposed" and i.get("experiment") and i.get("blocked_by"):
            st = by.get(i["blocked_by"][0], {}).get("status")
            if st and st != "proposed":
                follow.append((i["id"], "reject" if st == "parked" else "accept"))
    moved = G.move_all(wt, [a for a, _ in follow], [o for _, o in follow], [1.0, 0.0] + [1.0] * len(follow)) if follow else []
    pend = [i for i in q["items"] if i.get("status") == "proposed" and i.get("split_from") in by and i.get("code")][:6]
    if not pend:
        return moved
    pairs = [(CONTROL_GOOD, *CONTROL_PARENT), (CONTROL_BAD, *CONTROL_PARENT)]
    pairs += [(f"{i['title']} (files: {', '.join(i['code']['files'])})", by[i["split_from"]]["title"], by[i["split_from"]].get("next", ""))
              for i in pend]
    ps, err = decide_fn(pairs)
    outcomes = G.gate_outcomes(ps, len(pend))
    print(f"split regate: {[i['id'] for i in pend]} p={ps} err={err}")
    return moved + (G.move_all(wt, [i["id"] for i in pend], outcomes, ps) if outcomes else [])


def run(wt, repo, out, ask=None, decide_fn=None):
    os.makedirs(out, exist_ok=True)
    calls, log = [], []
    if ask is None:
        import engines

        def ask(prompt):
            r = engines.anthropic_call(MODEL, prompt, max_tokens=4000)   # replies with more context ran longer
            calls.append({k: r.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage", "error")})
            return r.get("text")
    try:
        regated = regate(wt, decide_fn or gate_fn(calls), calls)
        if regated:
            print(f"split: regated {regated}")
        Q.sync(wt)
        q = Q.load(wt)
        lines, tests = repo_files(repo)
        import repo_index as RI
        mods = [m for m in RI.build(repo) if CJ.allowed(m["path"])]   # it proposed audit/ files the code job refuses (6 of 7 refusals, run 44821449448)
        listing = "\n".join(f"{p} ({n})" for p, n in sorted(lines.items()) if CJ.allowed(p))
        subs = []
        tried = set()
        for parent in candidates(q)[:MAX_PARENTS]:
            parts = RI.show(RI.rank(mods, parent["title"] + " " + parent.get("next", ""), 8))
            reply = ask(PROMPT.format(id=parent["id"], title=parent["title"], next=parent.get("next", "")[:3000],
                                      files=listing, tests=", ".join(tests), parts=parts or "(none matched)"))
            d = first_json(reply or "")
            if not isinstance(d, dict) or not isinstance(d.get("subitems"), list):
                log.append({"parent": parent["id"], "ok": False, "why": "reply not the asked JSON", "raw": (reply or "")[:2500]})
                d = {}
            elif not d["subitems"]:
                log.append({"parent": parent["id"], "ok": False, "why": "model: nothing worker-sized", "raw": (reply or "")[:600]})
            got = 0
            for s in (d.get("subitems") or [])[:MAX_SUB]:
                ok, why = check(s, lines, tests)
                log.append({"parent": parent["id"], "ok": bool(ok), "why": why, "sub": s})
                if ok:
                    subs.append((parent, ok)); got += 1
            if not got:
                tried.add(parent["id"])
        mark_tried(wt, tried)
        if not subs:
            print("split: nothing passed the script check;", json.dumps(log)[:4000])
            return 0
        pairs = [(CONTROL_GOOD, *CONTROL_PARENT), (CONTROL_BAD, *CONTROL_PARENT)]
        pairs += [(f"{s['title']} (files: {', '.join(s['files'])})", p["title"], p.get("next", "")) for p, s in subs]
        ps, err = (decide_fn or gate_fn(calls))(pairs)
        outcomes = G.gate_outcomes(ps, len(subs))
        made, runs = [], {}
        for _ in range(Q.TRIES):
            Q.sync(wt)
            q = Q.load(wt)
            n = max([int(m.group(1)) for i in q["items"] for m in [re.fullmatch(r"K(\d+)", str(i["id"]))] if m] or [0]) + 1
            made, runs = [], {}
            for p, s in subs:
                kid = f"K{n}"; n += 1
                q["items"].append({"id": kid, "title": s["title"], "kind": "fix", "status": "proposed", "origin": "generated",
                                   "split_from": p["id"], "generated_at": Q.iso(Q.now()),
                                   "code": {"files": s["files"], "test": s["test"], "context": s["context"], "new": s["new"]},
                                   "touches": list(s["files"]) + [s["test"]],
                                   "next": s["next"], "refs": [f"queue {p['id']}", "worker/split.py"], "severity": "minor",
                                   "needs": "free", "cost": 0, "value": p.get("value", "medium"), "blocked_by": [],
                                   "batch": p.get("batch", "B3")})
                made.append(kid)
                if s.get("experiment"):   # the run itself: waits for the code item, then the dispatcher starts it
                    rid = f"K{n}"; n += 1
                    q["items"].append({"id": rid, "title": f"Run experiment {s['experiment']} ({p['id']})", "kind": "test",
                                       "status": "proposed", "origin": "generated", "split_from": p["id"], "generated_at": Q.iso(Q.now()),
                                       "experiment": {"name": s["experiment"]}, "touches": [], "next": f"Runs experiments/{s['experiment']}.py "
                                       f"once {kid} lands; its own rule decides.", "refs": [f"queue {kid}", "experiments/README.md"],
                                       "severity": "minor", "needs": "free", "cost": 0, "value": p.get("value", "medium"),
                                       "blocked_by": [kid], "batch": p.get("batch", "B3")})
                    runs[kid] = rid
            Q.save(wt, q)
            if Q.commit_push(wt, f"splitter: {', '.join(made)} proposed from {', '.join(sorted({p['id'] for p, _ in subs}))}"):
                break
        if outcomes:   # a run item takes its code item's gate outcome
            ids = made + [runs[k] for k in made if k in runs]
            outs = outcomes + [outcomes[made.index(k)] for k in made if k in runs]
            pss = list(ps[:2]) + list(ps[2:]) + [ps[2 + made.index(k)] for k in made if k in runs]
            moved = G.move_all(wt, ids, outs, pss)
        else:
            moved = []
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
