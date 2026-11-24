"""K67: fixed dependency truth set, part 1 (first five queue-item questions)."""
import json, re

ITEM_RE = re.compile(r"\b[A-Z]{1,3}\d{1,4}\b")
QUESTIONS = (
    ("next_any", "next", None),
    ("next_Q59", "next", "Q59"),
    ("next_K67", "next", "K67"),
    ("deps_Q59", "deps", "Q59"),
    ("deps_K67", "deps", "K67"),
)
PLAN = ("Fixed ten-question dependency truth set, defined before any tool runs. Part 1 is the first five queue-item "
        "questions: next_any lists items whose NEXT STEP names another item ID; next_Q59 and next_K67 list items "
        "whose NEXT STEP names Q59 or K67; deps_Q59 and deps_K67 list items naming Q59 or K67 as a dependency.")

def _items(q):
    if isinstance(q, (str, bytes)) or hasattr(q, "__fspath__"):
        with open(q, encoding="utf-8") as f: q = json.load(f)
    return q.get("items", q) if isinstance(q, dict) else q

def _next(item):
    for k in ("next", "next_step", "NEXT STEP", "next step"):
        if isinstance(item.get(k), str): return item[k]
    return ""

def _deps(item):
    out = []
    for k in ("deps", "depends_on", "dependencies", "dependency", "blocked_by"):
        v = item.get(k)
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(x, str): out += ITEM_RE.findall(x)
    return out

def truth(queue):
    items = _items(queue)
    ans = {}
    for qid, kind, target in QUESTIONS:
        got = []
        for item in items:
            iid = item.get("id")
            if not iid: continue
            if kind == "next":
                ids = [x for x in ITEM_RE.findall(_next(item)) if x != iid]
                ok = bool(ids) if target is None else target in ids
            else:
                ok = target in _deps(item)
            if ok: got.append(iid)
        ans[qid] = sorted(set(got))
    return ans

def run():
    return {"questions": [q[0] for q in QUESTIONS]}

def rule(r):
    ok = isinstance(r, dict) and r.get("questions") == [q[0] for q in QUESTIONS]
    return {"verdict": "holds" if ok else "invalid", "why": "five fixed queue-dependency questions" if ok else "bad question list"}
