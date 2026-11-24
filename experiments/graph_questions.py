"""Ten fixed dependency questions about this repo, with script-computed ground truth (no model text trusted)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PLAN = ("Ten fixed dependency questions about this repo; answers are computed by script from queue.json and the "
        "findings register, so no model text is trusted, and run() returns {question_id: answer}.")
QUESTIONS = ("p21_dependents", "review_items", "open_items", "closed_items", "no_dependency_items",
             "all_item_ids", "findings_citing_target", "findings_citing_queue_json", "findings_count",
             "findings_status_counts")

QUEUE_PATHS = ("queue.json", "worker/queue.json", "queue/queue.json", ".github/queue.json")
REGISTER_PATHS = ("findings.json", "findings/register.json", "audit/findings.json", "FINDINGS.json",
                  "register.json", "FINDINGS.md")

def _load(paths):
    for p in paths:
        full = os.path.join(REPO, p)
        if os.path.exists(full):
            try: return json.load(open(full, encoding="utf-8")), p
            except Exception: pass
    return [], None

def _rows(q, keys):
    if isinstance(q, dict):
        for k in keys:
            if isinstance(q.get(k), list): return q[k]
        if q and all(isinstance(v, dict) for v in q.values()): return list(q.values())
    return q if isinstance(q, list) else []

def _sid(d): return str(d.get("id") or d.get("finding_id") or d.get("item_id") or d.get("name") or d.get("number") or "")
def _status(d): return str(d.get("status") or d.get("state") or "").lower()
def _deps(d):
    for k in ("depends_on", "dependencies", "depends", "blocked_by", "requires"):
        v = d.get(k)
        if v: return [str(x) for x in (v if isinstance(v, list) else [v])]
    return []
def _cites(d):
    for k in ("cites", "files", "paths", "file", "path"):
        v = d.get(k)
        if v: return [str(x) for x in (v if isinstance(v, list) else [v])]
    return []

def run():
    q, _ = _load(QUEUE_PATHS)
    items = _rows(q, ("items", "queue", "issues"))
    r, _ = _load(REGISTER_PATHS)
    reg = _rows(r, ("findings", "items", "register"))
    target = "worker/tier.py"
    return {
        "p21_dependents": sorted({_sid(i) for i in items if "P21" in _deps(i)} - {""}),
        "review_items": sorted({_sid(i) for i in items if _status(i) == "review"} - {""}),
        "open_items": sorted({_sid(i) for i in items if _status(i) in ("open", "todo", "ready")} - {""}),
        "closed_items": sorted({_sid(i) for i in items if _status(i) in ("closed", "done", "merged")} - {""}),
        "no_dependency_items": sorted({_sid(i) for i in items if not _deps(i)} - {""}),
        "all_item_ids": sorted({_sid(i) for i in items} - {""}),
        "findings_citing_target": sorted({_sid(f) for f in reg
                                          if isinstance(f, dict) and target in _cites(f)} - {""}),
        "findings_citing_queue_json": sorted({_sid(f) for f in reg
                                              if isinstance(f, dict) and "queue.json" in _cites(f)} - {""}),
        "findings_count": str(len(reg)),
        "findings_status_counts": ", ".join(f"{s}:{sum(1 for f in reg if isinstance(f, dict) and _status(f) == s)}"
                                            for s in sorted({_status(f) for f in reg if isinstance(f, dict)} - {""})),
    }

def rule(result):
    keys = list(result) if isinstance(result, dict) else []
    if len(keys) != 10 or set(keys) != set(QUESTIONS):
        return {"verdict": "fails", "why": f"expected 10 fixed questions, got {sorted(keys)}"}
    for k in keys:
        v = result[k]
        if v is None or not isinstance(v, (list, str)):
            return {"verdict": "fails", "why": f"answer {k!r} is {type(v).__name__}, not list or str"}
    return {"verdict": "holds", "why": "ten fixed questions, every answer script-computed as a list or string"}
