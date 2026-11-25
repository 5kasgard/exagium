"""Load scored proposal log records into aligned per-item correctness vectors."""
import json, os, struct


def _read(data):
    out, i = [], 0
    while i < len(data):
        if i + 4 > len(data):
            raise ValueError("truncated length")
        n = struct.unpack_from("<I", data, i)[0]
        i += 4
        if i + n > len(data):
            raise ValueError("truncated record")
        out.append(json.loads(data[i:i + n]))
        i += n
    return out


def _id(e, i):
    return e.get("item", e.get("item_id", e.get("id", i))) if isinstance(e, dict) else i


def _val(e):
    if isinstance(e, dict):
        for k in ("correct", "score", "label", "value"):
            if k in e:
                return e[k]
        return 0
    return e


def _add(scores, order, cand, item, val):
    scores.setdefault(cand, {})
    if item not in scores[cand]:
        scores[cand][item] = 1 if val else 0
    if item not in order:
        order.append(item)


def load(log_dir):
    scores, order = {}, []
    for fn in sorted(os.listdir(log_dir)):
        path = os.path.join(log_dir, fn)
        if not os.path.isfile(path):
            continue
        data = open(path, "rb").read()
        try:
            recs = _read(data)
        except Exception:
            try:
                recs = [json.loads(data)]
            except Exception:
                continue
        for r in recs:
            if not isinstance(r, dict):
                continue
            if isinstance(r.get("candidates"), dict):
                for cand, c in r["candidates"].items():
                    for i, e in enumerate(c.get("items", [])):
                        _add(scores, order, cand, _id(e, i), _val(e))
            elif (r.get("candidate") is not None or r.get("name") is not None) and "items" in r:
                cand = r.get("candidate", r.get("name"))
                items = r["items"]
                if isinstance(items, dict):
                    for item, val in items.items():
                        _add(scores, order, cand, item, val)
                else:
                    for i, e in enumerate(items):
                        _add(scores, order, cand, _id(e, i), _val(e))
            elif r.get("candidate") is not None or r.get("name") is not None:
                cand = r.get("candidate", r.get("name"))
                item = r.get("item", r.get("item_id", r.get("id")))
                if item is not None:
                    _add(scores, order, cand, item, _val(r))
    cands = sorted(scores)
    keep = [it for it in order if all(it in scores[c] for c in cands)]
    return {c: [scores[c][it] for it in keep] for c in cands}
