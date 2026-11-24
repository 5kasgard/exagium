"""K70: pre-registered packet-pilot rule. Reads runner/results/*.json; no model calls."""
import glob, json, os
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner", "results")
PLAN = ("About 50 proposals per arm: one-call baseline versus packet plus repair. Measured are valid rate and accuracy per call. "
        "Keep the repair loop only if its per-call success exceeds one-third and its valid rate is not below baseline.")

def run():
    paths = glob.glob(os.path.join(RESULTS, "**", "*.json"), recursive=True)
    if not paths:
        return {"status": "no_pilot"}
    arms = defaultdict(lambda: {"proposals": 0, "valid": 0, "calls": 0, "correct": 0})
    found = False
    for p in paths:
        try:
            data = json.load(open(p))
        except (OSError, ValueError):
            continue
        if isinstance(data, list):
            recs = data
        elif isinstance(data, dict) and "arm" in data:
            recs = [data]
        elif isinstance(data, dict):
            recs = data.get("records") or []
            if not recs:
                recs = [dict(x, arm=k) for k, v in data.items() if isinstance(v, list) for x in v if isinstance(x, dict)]
            if not recs:
                recs = [dict(v, arm=k) for k, v in data.items() if isinstance(v, dict) and "valid" in v]
        else:
            recs = []
        for r in recs:
            if not isinstance(r, dict) or not r.get("arm"):
                continue
            a = arms[str(r["arm"])]
            a["proposals"] += 1
            a["valid"] += 1 if r.get("valid") else 0
            a["calls"] += int(r.get("calls") or 0)
            a["correct"] += int(r.get("correct") or 0)
            found = True
    return dict(arms) if found else {"status": "no_pilot"}

def rule(result):
    if not isinstance(result, dict) or result.get("status") == "no_pilot":
        return {"verdict": "invalid", "why": "no pilot results"}
    base = rep = None
    for k, v in result.items():
        if not isinstance(v, dict):
            continue
        s = k.lower()
        if base is None and ("baseline" in s or "one" in s):
            base = v
        if rep is None and ("repair" in s or "packet" in s):
            rep = v
    if base is None or rep is None:
        return {"verdict": "invalid", "why": "missing baseline or repair arm"}
    pc = rep.get("correct", 0) / rep["calls"] if rep.get("calls") else 0.0
    bv = base.get("valid", 0) / base["proposals"] if base.get("proposals") else 0.0
    rv = rep.get("valid", 0) / rep["proposals"] if rep.get("proposals") else 0.0
    ok = pc > 1 / 3 and rv >= bv
    why = f"repair per-call success {pc:.2f} (needs > 1/3); valid rate {rv:.2f} vs baseline {bv:.2f}"
    return {"verdict": "holds" if ok else "fails", "why": why}
