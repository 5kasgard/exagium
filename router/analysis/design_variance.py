"""Exploratory: how much would pairing and richer measures sharpen the benches? Not a verdict on any arm.

usage: python3 -I design_variance.py ROUTER_RUNS_DIR
Reads every non-control attempt of benches 2-4 (bench-4 pilots included: this estimates design numbers,
not results). For each bench and measure it prints:
  task_share  share of the measure's variance explained by the task (what pairing by task removes)
  rho         correlation of two different arms on the same task (paired-difference variance = 1 - rho)
  arm_share   share explained by the arm after the task is removed (the signal a bench is looking for)
"""
import glob, json, math, os, sys, itertools, collections, statistics as st

R = sys.argv[1]
rows = []
for p in glob.glob(os.path.join(R, "bench/bench[234]*/**/result.json"), recursive=True):
    d = json.load(open(p))
    if d.get("control") or d.get("arm") in ("Z", "control"):
        continue
    end = d.get("end") or d.get("ending")
    if end in ("provider_error", "seal_failed", "skipped_budget", "control"):
        continue
    bench = p.split("/bench/")[1].split("/")[0].split("-")[0]
    t = d.get("tokens") or {}
    tok = t.get("total") or ((t.get("prompt_tokens") or 0) + (t.get("completion_tokens") or 0)) or None
    calls = d.get("calls") or d.get("llm_calls") or d.get("actions")
    rows.append({"bench": bench, "task": d["task"], "arm": d["arm"], "passed": float(bool(d.get("passed"))),
                 "log_tokens": math.log10(tok) if tok else None, "calls": float(calls) if calls else None})

def shares(rs, m):
    rs = [r for r in rs if r[m] is not None]
    if len(rs) < 4: return None
    y = [r[m] for r in rs]; mu = st.fmean(y); tot = sum((v - mu) ** 2 for v in y)
    if tot == 0: return {"n": len(rs), "note": "no variance"}
    tm = {k: st.fmean([r[m] for r in rs if r["task"] == k]) for k in {r["task"] for r in rs}}
    resid = [r[m] - tm[r["task"]] for r in rs]
    task_ss = tot - sum(v * v for v in resid)
    am = {k: st.fmean([resid[i] for i, r in enumerate(rs) if r["arm"] == k]) for k in {r["arm"] for r in rs}}
    arm_ss = sum(am[r["arm"]] ** 2 for r in rs)
    # rho: correlation over pairs of different arms on the same task (centred on the overall mean)
    num = den = 0.0
    by_task = collections.defaultdict(list)
    for r in rs: by_task[r["task"]].append(r)
    pairs = 0
    for k, g in by_task.items():
        for a, b in itertools.combinations(g, 2):
            if a["arm"] != b["arm"]:
                num += (a[m] - mu) * (b[m] - mu); pairs += 1
    var = tot / len(rs)
    rho = (num / pairs) / var if pairs and var else None
    return {"n": len(rs), "task_share": round(task_ss / tot, 2), "arm_share": round(arm_ss / tot, 2),
            "rho": None if rho is None else round(rho, 2), "pairs": pairs}

out = {}
for b in sorted({r["bench"] for r in rows}):
    rs = [r for r in rows if r["bench"] == b]
    out[b] = {m: shares(rs, m) for m in ("passed", "log_tokens", "calls")}
    out[b]["arms"] = sorted({r["arm"] for r in rs}); out[b]["tasks"] = sorted({r["task"] for r in rs})
print(json.dumps(out, indent=1))
