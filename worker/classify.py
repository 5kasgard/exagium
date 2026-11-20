#!/usr/bin/env python3
"""Work types for every live queue item, judged by the free decision model with known-answer controls (owner 2026-11-21:
three job types cannot hold the work; "if it can't break it down into the vast types of work then it isn't working").

    python3 -I worker/classify.py QUEUE_WORKTREE OUT_DIR [MAX_ITEMS]

TYPES is an open list: a type is added when items fit none of them (an item with no type at p >= 0.5 is a 'misfit',
listed in the output so the list can grow). Each item gets `work_types` {type: p} and `work_types_at`; nothing else
about the item changes. Items are re-judged only when their title or next step changed (`work_types_key`).
A call is used only if both controls come out right; else nothing is written."""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate as G  # noqa: E402
import queue_ops as Q  # noqa: E402

TYPES = {   # name: what an item of this type needs done next (the executor, if any, in brackets)
    "code_change": "a small change to existing source code or a new small module, checked by tests (worker code job)",
    "doc_change": "an edit to documentation or an agent-facing text, with exact old and new lines (worker doc job)",
    "literature": "finding and reading published research to answer a question (worker research job)",
    "experiment": "running a measurement or test with a rule fixed before the run, then applying the rule to the result (none yet)",
    "decision": "choosing between options from evidence already gathered (none yet: the tier or the owner)",
    "build": "building a new component or system that needs several steps (splitter, then code jobs)",
    "data": "building, fetching or cleaning a dataset or task (none yet)",
    "owner_action": "something only the owner can do: a key, a payment, a repository setting, deleting branches (the owner)",
    "sandbox_or_workflow": "a change to CI workflows, the jail, seccomp or other paths a person must merge (chat)",
    "review": "checking work already done by someone else, or an audit (none yet)",
    "already_done": "the next step says the work is already built, live or finished, and nothing concrete remains to do (close or narrow it)",
    "held": "the owner has said to keep it in the queue only, with no work yet (wait)",
    # added 2026-11-21 from the first misfits (P23, P24, P5c): answers computed from what is already recorded
    "analysis": "computing an answer from data already recorded (results, logs, the queue, proposals) by script, with no new run (none yet: an experiment whose run() only reads)",
}
CONTROL = ("Fix a typo in worker/README.md line 12: 'recieve' -> 'receive'", "Typo in the worker card",
           "Change the one word on line 12 of worker/README.md; nothing else.")
CONTROL_YES, CONTROL_NO = "doc_change", "experiment"
ASK = ("Pair {k}: does the ITEM need, as its next piece of work, the kind of work its QUESTION describes? Answer for this "
       "item only; most items fit one or two kinds.")
PER_CALL = 3   # items per call: 3 x 13 types + 2 controls = 41 pairs (the decision model judged up to 40 claims per call)


def key(i):
    """Changes when the item's text or the type list changes, so a new type re-judges every item once."""
    return hashlib.sha256((i["title"] + "\n" + i.get("next", "") + "\n" + ",".join(TYPES)).encode()).hexdigest()[:16]


def todo(q, n):
    live = [i for i in q["items"] if i.get("status") not in ("done", "parked")]
    return [i for i in live if i.get("work_types_key") != key(i)][:n]


def judge(items, decide_fn):
    """{item id: {type: p}} for the items, or None if the controls failed or the call did."""
    names = list(TYPES)
    pairs = [(TYPES[CONTROL_YES], CONTROL[1], CONTROL[2]), (TYPES[CONTROL_NO], CONTROL[1], CONTROL[2])]
    for i in items:
        pairs += [(TYPES[t], i["title"], i.get("next", "")) for t in names]
    ps, err = decide_fn(pairs)
    if ps is None or len(ps) != len(pairs) or not (ps[0] >= 0.5 and ps[1] < 0.5):
        print(f"classify: controls failed or no answer (p={ps and ps[:2]}, err={err})")
        return None
    out, k = {}, 2
    for i in items:
        out[i["id"]] = {t: round(ps[k + j], 3) for j, t in enumerate(names)}
        k += len(names)
    return out


def run(wt, out, max_items=12, decide_fn=None):
    os.makedirs(out, exist_ok=True)
    calls, log = [], []
    decide_fn = decide_fn or (lambda pairs: G.decide(pairs, calls, ask=ASK, label="QUESTION"))
    try:
        Q.sync(wt)
        items = todo(Q.load(wt), max_items)
        got = {}
        for s in range(0, len(items), PER_CALL):
            r = judge(items[s:s + PER_CALL], decide_fn)
            if r:
                got.update(r)
            log.append({"items": [i["id"] for i in items[s:s + PER_CALL]], "ok": r is not None})
        if not got:
            print("classify: nothing judged"); return 0
        for _ in range(Q.TRIES):
            Q.sync(wt)
            q = Q.load(wt)
            for i in q["items"]:
                if i["id"] in got:
                    i["work_types"] = {t: p for t, p in sorted(got[i["id"]].items(), key=lambda kv: -kv[1]) if p >= 0.2}
                    i["work_types_key"], i["work_types_at"] = key(i), Q.iso(Q.now())
            Q.save(wt, q)
            if Q.commit_push(wt, f"classify: work types for {', '.join(sorted(got))}"):
                break
        misfit = [k for k, v in got.items() if max(v.values()) < 0.5]
        log.append({"judged": got, "misfit": misfit})
        print(f"classify: {len(got)} judged; misfits (no type at 0.5): {misfit}")
        for k, v in got.items():
            print(f"  {k}: " + ", ".join(f"{t} {p:.2f}" for t, p in sorted(v.items(), key=lambda kv: -kv[1]) if p >= 0.5))
        return 0
    finally:
        json.dump(log, open(os.path.join(out, "classify.json"), "w"), indent=1)
        with open(os.path.join(out, "calls.jsonl"), "w") as f:
            for c in calls:
                f.write(json.dumps(c) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(run(a[0], a[1], int(a[2]) if len(a) > 2 else 12))
