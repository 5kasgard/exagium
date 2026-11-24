"""Build a dev-fold packet from existing artifacts. No test split, no secrets."""
import json, os


def _read(path):
    if not path:
        return ""
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _first(d, names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return ""


def _pick(src, *keys):
    for k in keys:
        if src.get(k) is not None:
            return src[k]
    return None


def _counts(obj):
    src = obj.get("counts") if isinstance(obj, dict) and isinstance(obj.get("counts"), dict) else obj
    if not isinstance(src, dict):
        return {}
    return {"instructions": _pick(src, "instructions", "instruction_count", "ir_per_item", "ir"),
            "cache": _pick(src, "cache", "cache_count", "cache_misses", "ns_per_item")}


def _clusters(path):
    try:
        v = json.loads(_read(path))
    except ValueError:
        return []
    if isinstance(v, dict) and isinstance(v.get("clusters"), list):
        return v["clusters"]
    return v if isinstance(v, list) else [v]


def build(parent_dir, cluster_dir, champion_path):
    stderr = _read(_first(parent_dir, ("compile_stderr.txt", "compile.txt", "stderr.txt",
                                      "compile_stderr.log", "compile.err")))
    result = _first(parent_dir, ("result.json", "results.json"))
    try:
        counts = _counts(json.loads(_read(result))) if result else {}
    except ValueError:
        counts = {}
    clusters = []
    if os.path.isdir(cluster_dir):
        names = sorted(n for n in os.listdir(cluster_dir)
                       if n.endswith(".json") and "test" not in n.lower())
        if "clusters.json" in names:
            names = ["clusters.json"]
        for n in names:
            clusters.extend(_clusters(os.path.join(cluster_dir, n)))
    elif os.path.isfile(cluster_dir):
        clusters.extend(_clusters(cluster_dir))
    champ = ""
    if champion_path and "test" not in os.path.basename(champion_path).lower():
        champ = _read(champion_path)
    return {"stderr_tail": stderr.splitlines()[-40:], "counts": counts,
            "clusters": clusters, "champion_source": champ}
