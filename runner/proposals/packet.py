"""Build the four-key packet from a candidate's dev-fold artifacts (no test split, no secrets)."""
import json, os


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _json(path):
    try:
        return json.loads(_read(path) or "{}")
    except ValueError:
        return {}


def _failed_clusters(items):
    out, run = [], []
    for i, item in enumerate(items or []):
        if item.get("score", 1) == 0:
            run.append(i)
        elif run:
            out.append(run)
            run = []
    if run:
        out.append(run)
    return out


def build(parent_dir, cluster_dir, champion_path):
    """Return exactly stderr_tail, counts, clusters, champion_source; open only runner dev artifacts."""
    res = _json(os.path.join(parent_dir, "result.json"))
    dev = res.get("dev", res)
    cands = dev.get("candidates") or {}
    name = res.get("name")
    if name not in cands:
        name = next(iter(cands)) if len(cands) == 1 else os.path.basename(os.path.normpath(parent_dir))
    c = cands.get(name) or {}
    log = os.path.join(parent_dir, "dev.txt")
    if not os.path.exists(log):
        log = os.path.join(parent_dir, "reproduce.txt")
    cluster = _json(os.path.join(cluster_dir, "dev.json"))
    cluster = cluster.get("dev", cluster)
    items = ((cluster.get("candidates") or {}).get(name) or {}).get("items")
    if items is None:
        items = c.get("items", [])
    return {
        "stderr_tail": _read(log).splitlines()[-40:],
        "counts": {"instructions": c.get("ir_per_item"), "cache": c.get("cache")},
        "clusters": _failed_clusters(items),
        "champion_source": _read(champion_path),
    }
