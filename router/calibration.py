"""Calibration registry: outcomes per actor and task family, Wilson lower bound as the score.
Standard library only.
"""
import json
import math


Z = 1.96  # 95% confidence
_counts = {}  # (actor, family) -> [successes, trials]


def _key(actor, family):
    return (actor, family)


def record(actor, family, ok):
    """Record one outcome for (actor, family). ok=True counts as success."""
    k = _key(actor, family)
    if k not in _counts:
        _counts[k] = [0, 0]
    _counts[k][1] += 1
    if ok:
        _counts[k][0] += 1


def score(actor, family):
    """Wilson lower bound at z=1.96. Returns 0.0 when there are no trials."""
    s, n = _counts.get(_key(actor, family), (0, 0))
    if n == 0:
        return 0.0
    p = s / n
    z2 = Z * Z
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    margin = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def dump(path):
    """Write counts to JSON file at path."""
    data = {f"{a}|{f}": {"successes": v[0], "trials": v[1]} for (a, f), v in _counts.items()}
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=True)


def load(path):
    """Read counts from JSON file at path, replacing current state."""
    global _counts
    with open(path) as f:
        data = json.load(f)
    _counts = {}
    for k, v in data.items():
        a, fam = k.split("|", 1)
        _counts[(a, fam)] = [v["successes"], v["trials"]]
