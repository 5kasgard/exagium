import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "runner", "results", "packet-pilot")
PLAN = """About 50 proposals per arm: one-call baseline versus packet plus repair.
Measure valid rate and accuracy per call. Keep the repair loop only if its per-call
success exceeds one-third and its valid rate is not below the baseline's."""
ARMS = ("baseline", "repair")


def _empty():
    return {a: {"proposals": 0, "valid": 0, "calls": 0, "correct": 0} for a in ARMS}


def run(results_dir=None):
    out = _empty()
    seen = False
    for path in sorted(glob.glob(os.path.join(results_dir or RESULTS_DIR, "*.json"))):
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if not isinstance(r, dict) or r.get("arm") not in ARMS:
                continue
            valid, calls, correct = r.get("valid"), r.get("calls"), r.get("correct")
            if (not isinstance(valid, bool) or not isinstance(calls, int) or isinstance(calls, bool)
                    or not isinstance(correct, int) or isinstance(correct, bool)):
                continue
            a = r["arm"]
            out[a]["proposals"] += 1
            out[a]["valid"] += int(valid)
            out[a]["calls"] += calls
            out[a]["correct"] += correct
            seen = True
    return out if seen else {"status": "no_pilot"}


def rule(result):
    if not isinstance(result, dict) or result.get("status") == "no_pilot":
        return {"verdict": "invalid", "why": "no pilot result"}
    try:
        b, r = result["baseline"], result["repair"]
        br = b["valid"] / b["proposals"] if b["proposals"] else 0.0
        rr = r["valid"] / r["proposals"] if r["proposals"] else 0.0
        pc = r["correct"] / r["calls"] if r["calls"] else 0.0
    except (KeyError, TypeError, ZeroDivisionError):
        return {"verdict": "fails", "why": "missing per-arm counts"}
    if pc > 1 / 3 and rr >= br:
        return {"verdict": "holds",
                "why": f"repair per-call success {pc:.3f} > 1/3 and valid rate {rr:.3f} >= baseline {br:.3f}"}
    if pc <= 1 / 3:
        return {"verdict": "fails", "why": f"repair per-call success {pc:.3f} does not exceed 1/3"}
    return {"verdict": "fails", "why": f"repair valid rate {rr:.3f} is below baseline {br:.3f}"}
