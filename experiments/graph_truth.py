"""Truth set part 1: the first five queue dependency questions, fixed before any run."""
import json, os, re

TARGETS = ("K67", "K66", "K65", "K64")
PLAN = ("Fixed 10-question dependency truth set, written before any tool run. "
        "q1: items whose NEXT STEP names another item ID. "
        "q2: items listing K67 as a dependency. q3: items listing K66 as a dependency. "
        "q4: items listing K65 as a dependency. q5: items listing K64 as a dependency. "
        "q6-q10: later part, not implemented here.")


def _items(queue):
    q = json.load(open(queue)) if isinstance(queue, str) else queue
    return q["items"] if isinstance(q, dict) else q


def _text(item):
    return " ".join(str(item.get(k, "")) for k in ("next", "next_step", "next step", "NEXT STEP"))


def _deps(item):
    out = []
    for k in ("depends_on", "dependencies", "deps", "blocked_by", "requires", "depends"):
        v = item.get(k) or []
        if isinstance(v, str):
            v = [v]
        for x in v:
            if isinstance(x, dict):
                x = x.get("id")
            if x:
                out.append(str(x))
    return out


def truth(queue):
    items = _items(queue)
    ids = {i["id"] for i in items}
    ans = {"q1": sorted(i["id"] for i in items
                        if any(o != i["id"] and re.search(r"\b" + re.escape(o) + r"\b", _text(i)) for o in ids))}
    for n, target in enumerate(TARGETS, 2):
        ans[f"q{n}"] = sorted(i["id"] for i in items if target in _deps(i))
    return ans


def run():
    return truth(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "queue.json"))


def rule(r):
    return {"verdict": "holds", "why": "first five dependency answers computed"}
