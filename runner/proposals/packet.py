"""Build the next proposal's packet from dev-fold artifacts only (queue K68). No secrets, JSON only."""
import json, os


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _json(path, default):
    text = _read(path)
    if text is None:
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default


def build(parent_dir, cluster_dir, champion_path):
    stderr = None
    for name in ("compile.stderr", "compile.txt", "stderr.txt"):
        stderr = _read(os.path.join(parent_dir, name))
        if stderr is not None:
            break
    result = _json(os.path.join(parent_dir, "result.json"), {})
    return {
        "stderr_tail": (stderr or "").splitlines()[-40:],
        "counts": {"instructions": result.get("instructions", result.get("ir_per_item")),
                   "cache": result.get("cache", result.get("ns_per_item"))},
        "clusters": _json(os.path.join(cluster_dir, "clusters.json"), []),
        "champion_source": _read(champion_path) or "",
    }
