#!/usr/bin/env python3
"""Event spine, test 1 (queue P37): can today's queue be rebuilt from events the Forge already records? Zero model calls.

Two event sources, two checks:
  A. runs[]: each item's own run log. Fold its outcomes through the state machine (worker/transitions.py TABLE) from
     the item's first state and compare the result with the item's status now.
  B. git history of queue.json: every status change between consecutive commits is an event (actor = commit subject's
     first word). Check each against TABLE: a change TABLE does not allow was written around the state machine.

python3 -I audit/spine/replay.py TRANSITIONS_PY QUEUE_JSON [--history N]   (N = commits to walk, default all)
Prints a JSON report. Reads only; writes nothing."""
import collections, importlib.util, json, subprocess, sys

WORKER_OUTCOMES = None


NEXT = None
SKIP_COUNT = ("budget", "retry", "dispatch", "late_result", "defer")


def load_table(path):
    global NEXT
    spec = importlib.util.spec_from_file_location("transitions", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    NEXT = m.next_status
    return m.TABLE, set(m.WORKER)


def fold(item, table, worker):
    """Return (predicted status, list of refused (state, outcome), entries without an outcome). Start: the item's
    baseline (status written before events were enforced, P37b) if it has one, else the first event's 'from', else
    'proposed' for a generated item, else 'open'. An event that carries 'to' is trusted for its target (the writer
    applied the full rule); one without is folded through the table. A worker outcome seen in 'open' implies the
    claim that older logs did not record."""
    runs = item.get("runs", [])
    base = item.get("baseline")
    if base:
        state, runs = base["status"], [r for r in runs if r.get("at", "") > base["at"]]
    elif runs and runs[0].get("from"):
        state = runs[0]["from"]
    else:
        outs = [r.get("outcome") for r in runs]
        state = "proposed" if item.get("origin") == "generated" or (outs and outs[0] in ("accept", "reject")) else "open"
    refused, skipped, fails = [], 0, 0
    for r in runs:
        o = r.get("outcome")
        res = str(r.get("result", ""))
        if o is None:
            skipped += 1
            continue
        if r.get("to"):
            if r.get("from") and r["from"] != state:
                refused.append((state, f"{o} (event says from {r['from']})"))
            state = r["to"]
            continue
        # consecutive failures, as worker/queue_ops.failures counts them: a crash loop climbs a tier
        if o in ("failed", "lost", "lease_expired") or res == "lease expired":
            fails += 1
        elif o not in SKIP_COUNT and o != "claim" and not res.startswith("not started"):
            fails = 0
        if state == "open" and o in worker:
            state = "in progress"
        if (state, o) not in table:
            refused.append((state, o))
            continue
        state = NEXT(state, o, fails) if NEXT else table[(state, o)]
    return state, refused, skipped


def check_runs(q, table, worker):
    rep = collections.Counter()
    mism = collections.Counter()
    examples = {}
    refused_all = collections.Counter()
    for i in q["items"]:
        pred, refused, skipped = fold(i, table, worker)
        rep["items"] += 1
        rep["runs_without_outcome"] += skipped
        for r in refused:
            refused_all[f"{r[0]} + {r[1]}"] += 1
        if not i.get("runs") and not i.get("baseline"):
            rep["no_runs"] += 1
            key = "no events, status " + i["status"]
            if i["status"] != ("proposed" if i.get("origin") == "generated" else "open"):
                mism[key] += 1
                examples.setdefault(key, i["id"])
            else:
                rep["equal"] += 1
            continue
        if pred == i["status"] or (pred == "in progress" and i["status"] == "in progress"):
            rep["equal"] += 1
        else:
            key = f"replay {pred} != now {i['status']}"
            mism[key] += 1
            examples.setdefault(key, i["id"])
    return {"counts": dict(rep), "mismatch": dict(mism.most_common()), "example": examples,
            "refused_transitions": dict(refused_all.most_common(15))}


def check_history(path, table, limit):
    revs = subprocess.run(["git", "log", "--reverse", "--format=%H\t%s", "--", path], capture_output=True, text=True,
                          check=True).stdout.splitlines()
    if limit:
        revs = revs[-limit:]
    prev, rep = None, collections.Counter()
    bad, actors, examples = collections.Counter(), collections.Counter(), {}
    for line in revs:
        sha, subj = line.split("\t", 1)
        raw = subprocess.run(["git", "show", f"{sha}:{path}"], capture_output=True, text=True).stdout
        try:
            cur = {i["id"]: i["status"] for i in json.loads(raw)["items"]}
        except Exception:
            rep["unreadable_snapshot"] += 1
            continue
        if prev is not None:
            actor = subj.split(":", 1)[0] if ":" in subj else subj.split(" ", 1)[0]
            for k, s in cur.items():
                p = prev.get(k)
                if p is None:
                    rep["created"] += 1
                    continue
                if p == s:
                    continue
                rep["changes"] += 1
                ok = any(v == s for (f, _o), v in table.items() if f == p)
                if ok:
                    rep["allowed"] += 1
                else:
                    rep["not_in_table"] += 1
                    key = f"{p} -> {s}"
                    bad[key] += 1
                    actors[actor] += 1
                    examples.setdefault(key, f"{k} @ {sha[:7]} ({subj[:60]})")
        prev = cur
    rep["snapshots"] = len(revs)
    return {"counts": dict(rep), "not_in_table": dict(bad.most_common()), "by_actor": dict(actors.most_common()),
            "example": examples}


def main(a):
    table, worker = load_table(a[0])
    q = json.load(open(a[1]))
    limit = int(a[a.index("--history") + 1]) if "--history" in a else 0
    out = {"A_runs_fold": check_runs(q, table, worker), "B_git_history": check_history(a[1], table, limit)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
