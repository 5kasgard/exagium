"""Truth set Q1-Q3: dependency answers computed by script from queue.json's `blocked_by`, not by a model."""
import json, os

PLAN = """Ten fixed dependency questions (answers computed by script from queue.json and the findings register, never by a model):
Q1: items with no blocked_by (queue roots).
Q2: items with at least one blocked_by.
Q3: item ids named as a blocker by at least one other item.
Q4: items whose blockers are all done.
Q5: items whose at least one blocker is done.
Q6: items whose at least one blocker is parked.
Q7: items whose at least one blocker is missing from the queue.
Q8: items reachable from a blocked root (transitive).
Q9: items in a cycle.
Q10: items that are both a blocker and blocked.
Only Q1-Q3 are implemented in run(); the other seven are future work."""


def run(wt=None):
    path = os.path.join(wt, "queue.json") if wt else "queue.json"
    q = json.load(open(path))
    items = q["items"]
    q1 = sorted(i["id"] for i in items if not i.get("blocked_by"))
    q2 = sorted(i["id"] for i in items if i.get("blocked_by"))
    q3 = sorted({b for i in items for b in (i.get("blocked_by") or [])})
    return {"Q1": q1, "Q2": q2, "Q3": q3}


def rule(result):
    for k in ("Q1", "Q2", "Q3"):
        if not isinstance(result.get(k), list):
            return {"verdict": "invalid", "why": f"answer {k} is missing or not a list"}
    return {"verdict": "holds", "why": "3 of 10 answered"}
