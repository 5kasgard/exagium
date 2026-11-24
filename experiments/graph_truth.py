"""Fixed dependency questions answered by script from queue.json and the findings register (no model call).

Each question is pure data (a dict), so the set is fixed and reviewable. truth() reads the two JSON files and
answers every question; no model is called. Questions are about dependency edges a coder should know.
"""
import json

QUESTIONS = [
    {"id": "q01", "kind": "queue_file", "file": "worker/split.py"},
    {"id": "q02", "kind": "queue_file", "file": "router/engines.py"},
    {"id": "q03", "kind": "queue_file", "file": "worker/repo_index.py"},
    {"id": "q04", "kind": "blocked_by", "target": "K1"},
    {"id": "q05", "kind": "blocked_by", "target": "K2"},
    {"id": "q06", "kind": "split_from", "target": "K1"},
    {"id": "q07", "kind": "count_items"},
    {"id": "q08", "kind": "register_file", "file": "worker/split.py"},
    {"id": "q09", "kind": "register_file", "file": "router/engines.py"},
    {"id": "q10", "kind": "count_findings"},
]


def _items(queue):
    return queue.get("items") or []


def _findings(register):
    return register if isinstance(register, list) else (register.get("findings") or [])


def _item_files(i):
    out = list(i.get("files") or [])
    out += list((i.get("code") or {}).get("files") or [])
    return [x for x in out if isinstance(x, str)]


def _finding_files(f):
    out = []
    for k in ("files", "cites", "paths"):
        out += list(f.get(k) or [])
    return [x for x in out if isinstance(x, str)]


def truth(queue_path, register_path):
    q = json.load(open(queue_path, encoding="utf-8"))
    r = json.load(open(register_path, encoding="utf-8"))
    items, findings = _items(q), _findings(r)
    out = {}
    for question in QUESTIONS:
        k, qid = question["kind"], question["id"]
        if k == "queue_file":
            f = question["file"]
            out[qid] = sorted(i["id"] for i in items if i.get("id") and f in _item_files(i))
        elif k == "blocked_by":
            t = question["target"]
            out[qid] = sorted(i["id"] for i in items if i.get("id") and t in (i.get("blocked_by") or []))
        elif k == "split_from":
            t = question["target"]
            out[qid] = sorted(i["id"] for i in items if i.get("id") and i.get("split_from") == t)
        elif k == "count_items":
            out[qid] = len(items)
        elif k == "register_file":
            f = question["file"]
            out[qid] = sorted(str(x["id"]) for x in findings if x.get("id") is not None and f in _finding_files(x))
        elif k == "count_findings":
            out[qid] = len(findings)
        else:
            raise ValueError(f"unknown question kind: {k}")
    return out
