"""Backlog lanes (owner 2026-11-25: no pile of refused or hidden work that is put aside and forgotten as the queue grows).

Every item that is not done gets exactly one lane: who moves it next, and since when it has waited. An item no lane
takes is an ORPHAN; an item older than its lane's limit is STALE. Both are printed first and make the exit code 1, so
a pile cannot grow unseen. Pure on queue.json (no network, no writes).

  python3 -I worker/backlog.py QUEUE_JSON [--json OUT]

Lanes, each with the job that consumes it:
  worker          ready() now: the free worker (worker.yml)
  splitter        split.candidates(): make-work / the idle job (worker/split.py)
  gate            proposed: the generator's decision model (worker/generate.py, split.regate)
  review          review / escalated: the free tier (tier.yml), then the Claude shift
  in progress     claimed by a worker (lease: queue_ops.expire)
  scheduled       an experiment job running (dispatch_experiments.py)
  blocked         waits on its blockers; an orphan if a blocker is parked or itself an orphan
  claude          open work no automatic job takes (the splitter found nothing worker-sized, or no job type fits):
                  the Claude shift, oldest first within value
  owner-only      transitions.OWNER_ONLY categories only (needs a reason starting "owner-only:" or an owner hold)
  parked          closed with a recorded reason; a resplit child whose parent is still moving is not a dead end
Limits (days in lane before STALE) are guesses, labelled so, until a week of lane ages is measured.
"""
import datetime as dt, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import queue_ops as Q  # noqa: E402
import transitions as T  # noqa: E402

LIMIT_DAYS = {"worker": 2, "splitter": 3, "gate": 2, "review": 2, "in progress": 1, "scheduled": 2, "blocked": 14,
              "claude": 7, "owner-only": 14}   # guesses (2026-11-25), to be set from measured lane ages
HOLD = re.compile(r"owner[^.]{0,40}(add to the queue only|do not start until|no work yet)", re.I)


def ts(s):
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def since(i):
    """When the item entered its present state: its last run, else when it was filed."""
    rs = i.get("runs") or []
    # aged_from: set once (2026-11-25) on items with no run and no filed date, so their wait is counted from then, not hidden
    for t in ([rs[-1].get("at")] if rs else []) + [i.get("filed_at"), i.get("generated_at"), i.get("aged_from")]:
        if t and ts(t):
            return ts(t)
    return None


def lanes(q, now=None):
    import split as S
    now = now or Q.now()
    items = q["items"]
    by = {i["id"]: i for i in items}
    ready = {i["id"] for i in Q.ready(q, now)}
    cand = {i["id"] for i in S.candidates(q)}
    out = {}

    def lane(i):
        st, nxt = i["status"], str(i.get("next") or "")
        if st == "done":
            return None, ""
        if st == "waiting on the owner":
            r = (i.get("runs") or [{}])[-1].get("result", "")
            cat = T.owner_only(r) or T.owner_only(nxt.lower()) or ("owner-hold" if HOLD.search(nxt) else None)
            return ("owner-only", cat) if cat else ("orphan", "waiting on the owner with no owner-only category")
        if st in ("review", "escalated"):
            return "review", st
        if st == "proposed":
            return "gate", ""
        if st in ("in progress", "scheduled"):
            return st, ""
        if st == "parked":
            r = (i.get("runs") or [{}])[-1]
            if r.get("outcome") == "resplit" and i.get("split_from"):
                p = by.get(i["split_from"])
                if p and p["status"] != "done":
                    return "parked", f"resplit: parent {p['id']} is {p['status']}"
                return "parked", "resplit: parent finished"
            why = r.get("result") or nxt
            return ("parked", why[:80]) if why.strip() else ("orphan", "parked with no recorded reason")
        if st == "blocked" or (st == "open" and any(by.get(b, {}).get("status") != "done" for b in i.get("blocked_by", []))):
            dead = [b for b in i.get("blocked_by", []) if by.get(b, {}).get("status") in ("parked", None)]
            return ("orphan", f"blocked by parked or missing {dead}") if dead else ("blocked", ",".join(i.get("blocked_by", [])))
        if st == "open":
            n = (i.get("experiment") or {}).get("name", "")
            if re.fullmatch(r"[a-z0-9_]+", n) and os.path.isfile(os.path.join(os.path.dirname(HERE), "experiments", n + ".py")):
                return "worker", "experiment"   # worker/dispatch_experiments.py starts it from the idle job (K89 was shown as claude work)
            if i["id"] in ready:
                return "worker", Q.job_type(i) or ""
            if i["id"] in cand:
                return "splitter", ""
            if HOLD.search(nxt) or T.owner_only(nxt.lower()):
                return "owner-only", T.owner_only(nxt.lower()) or "owner-hold"
            return "claude", "split tried: not worker-sized" if i.get("split_tried") else "no job type fits"
        return "orphan", f"unknown status {st!r}"

    for i in items:
        ln, why = lane(i)
        if ln is None:
            continue
        s = since(i)
        age = round((now - s).total_seconds() / 86400, 1) if s else None
        stale = ln in LIMIT_DAYS and age is not None and age > LIMIT_DAYS[ln]   # undated items are counted apart, not hidden
        out[i["id"]] = {"lane": ln, "why": why, "age_days": age, "stale": stale, "value": i.get("value"),
                        "title": i.get("title", "")[:90]}
    return out


def flow(q, now=None, days=7):
    """Finished vs. created in the last `days`: does the backlog shrink?"""
    now = now or Q.now()
    cut = now - dt.timedelta(days=days)
    made = sum(1 for i in q["items"] if (ts(i.get("filed_at") or i.get("generated_at") or "") or now) >= cut
               and (i.get("filed_at") or i.get("generated_at")))
    # done items whose last event falls in the window (older events carry no 'to': counting r['to'] == 'done' found 38 of 145)
    fin = sum(1 for i in q["items"] if i["status"] == "done" and i.get("runs") and (ts(i["runs"][-1].get("at") or "") or cut) > cut)
    return {"days": days, "created": made, "finished": fin}


def report(q, now=None):
    L = lanes(q, now)
    rows = sorted(L.items(), key=lambda kv: (kv[1]["lane"] != "orphan", not kv[1]["stale"], -(kv[1]["age_days"] or 0)))
    counts = {}
    for v in L.values():
        counts[v["lane"]] = counts.get(v["lane"], 0) + 1
    orphans = [k for k, v in L.items() if v["lane"] == "orphan"]
    stale = [k for k, v in L.items() if v["stale"]]
    undated = [k for k, v in L.items() if v["age_days"] is None and v["lane"] != "parked"]
    f = flow(q, now)
    lines = [f"lanes: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])),
             f"last {f['days']} days: {f['created']} items created, {f['finished']} finished",
             f"orphans {len(orphans)}, stale {len(stale)}, undated (no run, no filed date) {len(undated)}: {' '.join(sorted(undated))}"]
    for k, v in rows:
        if v["lane"] == "orphan" or v["stale"]:
            lines.append(f"  {'ORPHAN' if v['lane'] == 'orphan' else 'STALE '} {k:<6} {v['lane']:<11} {str(v['age_days']):>5}d  "
                         f"{v['why'][:60]} | {v['title'][:60]}")
    return "\n".join(lines), L, orphans, stale


def main(a):
    q = json.load(open(a[0]))
    text, L, orphans, stale = report(q)
    print(text)
    if "--json" in a:
        json.dump(L, open(a[a.index("--json") + 1], "w"), indent=1)
    return 1 if orphans or stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
