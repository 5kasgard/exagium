"""bench-3: every pre-registered rule as code. Standard library only.

  decide.py selftest             every ending maps to exactly one label; every rule returns a verdict
  decide.py plan DIR REMAINING   JSON list of the next attempts that fit in REMAINING free requests
  decide.py status DIR           STATUS.json and STATUS.md text (markdown on stdout, JSON to DIR/STATUS.json)

DIR holds */*/result.json, one per attempt (see measure.py). PREREG.md states the same rules in words;
where the two differ, this file is the rule and the difference is a defect to report.
"""
import glob, json, math, os, re, statistics, sys

ARMS = {  # name: (model, vendor)
    "A": ("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "NVIDIA"),
    "B": ("openrouter/recursive/flint-v1.0:free", "Recursive"),
    "D": ("openrouter/cohere/north-mini-code:free", "Cohere"),
    "E": ("openrouter/baidu/ernie-5.5-flash-preview:free", "Baidu"),
    "F": ("openrouter/dots-studio/dots-3-note-preview:free", "Dots Studio"),
}
TASKS, REPS = ["T1", "T2", "T3"], 5
CAP = 80                 # iterations per attempt, and the worst-case requests one attempt can use
MARGIN = 20              # free requests always left unspent
BATCH_MAX = 12           # attempts per batch
FUTILITY_N = 6           # an arm with 0 clean passes in its first 6 counted attempts stops
MAX_TRIES = 3            # a cell cut by the provider this many times is recorded as provider_cut
ENDS = ["control", "skipped_budget", "timeout", "harness_crash", "provider_error",
        "iteration_cap", "stuck", "finished", "unclassified"]
COUNTED = {"timeout", "harness_crash", "iteration_cap", "stuck", "finished", "unclassified"}
CRASH = {"harness_crash", "unclassified"}
PROVIDER = re.compile(r"RateLimit|\b429\b|RESOURCE_EXHAUSTED|quota|LLMAuthenticationError|APIConnectionError"
                      r"|ServiceUnavailable|\b50[234]\b|Timeout|timed out", re.I)


def classify(run, events, control=False):
    """One label per attempt. The first matching line wins; the order is part of the rule."""
    if control:
        return "control"
    if run.get("skipped"):
        return "skipped_budget"
    err = run.get("error") or ""
    errs = [e for e in events if e.get("kind") == "ConversationErrorEvent"]
    codes = [str(e.get("code")) for e in errs]
    if "rc=124" in err or "rc=137" in err:
        return "timeout"
    if err.startswith("harness"):
        return "harness_crash"
    if PROVIDER.search(err + " " + " ".join(c + " " + str(e.get("detail")) for c, e in zip(codes, errs))):
        return "provider_error"
    if "MaxIterationsReached" in codes or "MaxIterationsReached" in err:
        return "iteration_cap"
    status = str(run.get("status", "")).upper()
    if status.endswith("STUCK"):
        return "stuck"
    if status.endswith("FINISHED") and not err and not codes:
        return "finished"
    return "unclassified"


def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def interval(k, n, alpha=0.05):
    """Clopper-Pearson two-sided interval for k of n, by bisection."""
    if n == 0:
        return 0.0, 1.0

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    low = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) < alpha / 2)
    high = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) > alpha / 2)
    return low, high


def share_verdict(k, n, min_n=5):
    """holds at 7/9 or more, fails at 4/9 or less (exact integer comparison, no rounding)."""
    if n < min_n:
        return "inconclusive"
    return "holds" if 9 * k >= 7 * n else "fails" if 9 * k <= 4 * n else "inconclusive"


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*", "*", "result.json"))):
        out.append(json.load(open(f)))
    return out


def cells(results):
    """(task, arm, rep) -> the counted result, or 'provider_cut', or None while still open."""
    done, tries = {}, {}
    for r in results:
        key = (r["task"], r["arm"], r["rep"])
        if r["end"] in COUNTED or r["end"] == "control":
            done.setdefault(key, r)   # the first counted result stands; a rerun never replaces it
        elif r["end"] == "provider_error":
            tries[key] = tries.get(key, 0) + 1
    for key, n in tries.items():
        if key not in done and n >= MAX_TRIES:
            done[key] = "provider_cut"
    return done


def arm_stats(done, arm):
    rs = [v for (t, a, r), v in sorted(done.items()) if a == arm and v != "provider_cut"]
    k = sum(1 for r in rs if r["passed"] and r["clean"])
    return rs, k


def futile(done, arm):
    rs, _ = arm_stats(done, arm)
    return len(rs) >= FUTILITY_N and not any(r["passed"] and r["clean"] for r in rs[:FUTILITY_N])


def plan(results, remaining):
    done = cells(results)
    todo = [{"task": t, "arm": "Z", "model": "none", "rep": 0} for t in TASKS if (t, "Z", 0) not in done]
    for rep in range(REPS):
        for t in TASKS:
            for a in ARMS:
                if (t, a, rep) not in done and not futile(done, a):
                    todo.append({"task": t, "arm": a, "model": ARMS[a][0], "rep": rep})
    n_agents = max(0, min(BATCH_MAX, (remaining - MARGIN) // CAP))
    out, used = [], 0
    for c in todo:
        if c["arm"] == "Z":
            out.append(c)
        elif used < n_agents:
            out.append(c); used += 1
    return out, len([c for c in todo if c["arm"] != "Z"])


def stop_on_green(rs):
    """Would stopping at the first visible-green snapshot have kept the pass, and what would it save?"""
    use = [r for r in rs if r["passed"] and r["clean"] and r.get("green_actions") is not None and r["actions"]]
    kept = sum(1 for r in use if r.get("green_kept"))
    saving = [1 - r["green_actions"] / r["actions"] for r in use]
    med = statistics.median(saving) if saving else None
    if len(use) < 8:
        v = "inconclusive"
    elif kept / len(use) >= 0.90 and med >= 0.20:
        v = "holds"
    elif kept / len(use) < 0.75 or med < 0.10:
        v = "fails"
    else:
        v = "inconclusive"
    return {"n": len(use), "kept": kept, "median_saving": med, "verdict": v}


def status(results):
    done = cells(results)
    counted = [v for v in done.values() if v != "provider_cut" and v["end"] in COUNTED]
    controls = [v for v in done.values() if v != "provider_cut" and v["end"] == "control"]
    _, open_n = plan(results, 0)
    s = {"complete": open_n == 0 and len(controls) == len(TASKS), "open_cells": open_n, "counted": len(counted),
         "provider_cut": sorted(k for k, v in done.items() if v == "provider_cut"),
         "control_ok": len(controls) == len(TASKS) and not any(c["passed"] for c in controls),
         "seal": "holds" if not any(r["outside_paths"] for r in counted) else "broken",
         "egress_drops": sum(r.get("egress_drops") or 0 for r in counted),
         "ends": {e: sum(1 for r in counted if r["end"] == e) for e in ENDS if any(r["end"] == e for r in counted)}}
    ok = sum(1 for r in counted if r["end"] not in CRASH)
    s["harness"] = {"no_crash": ok, "n": len(counted), "verdict": share_verdict(ok, len(counted))}
    s["arms"] = {}
    for a in ARMS:
        rs, k = arm_stats(done, a)
        lo, hi = interval(k, len(rs))
        calls = [r["calls"] for r in rs if r.get("calls") is not None]
        s["arms"][a] = {"model": ARMS[a][0], "clean_passes": k, "n": len(rs), "low": round(lo, 3), "high": round(hi, 3),
                        "usable": "inconclusive" if not rs else "holds" if lo >= 0.5 else "fails" if hi < 0.5 else "inconclusive",
                        "stopped_for_futility": futile(done, a),
                        "median_calls": statistics.median(calls) if calls else None,
                        "calls_per_clean_pass": round(sum(calls) / k, 1) if k and calls else None,
                        "median_wall_s": statistics.median([r["wall_s"] for r in rs if r.get("wall_s")] or [0]),
                        "ends": {e: sum(1 for r in rs if r["end"] == e) for e in ENDS if any(r["end"] == e for r in rs)}}
    A = s["arms"]
    s["ranking"] = [f"{x} above {y}" for x in A for y in A if A[x]["n"] and A[y]["n"] and A[x]["low"] > A[y]["high"]]
    s["stop_on_green"] = stop_on_green(counted)
    calls = [r["calls"] for r in counted if r.get("calls") is not None]
    s["budget"] = {"median_calls": statistics.median(calls) if calls else None, "total_calls": sum(calls)}
    return s


def markdown(s):
    L = ["# bench-3 status (generated by decide.py; not hand-edited)", "",
         f"- Complete: **{s['complete']}** ({s['counted']} counted attempts, {s['open_cells']} cells open)",
         f"- Control fails all tasks, as required: **{s['control_ok']}**",
         f"- Seal: **{s['seal']}**; blocked outbound connections: {s['egress_drops']}",
         f"- Harness (no crash): {s['harness']['no_crash']}/{s['harness']['n']}, **{s['harness']['verdict']}**",
         f"- Endings: {s['ends']}",
         f"- Budget: median {s['budget']['median_calls']} calls per attempt; {s['budget']['total_calls']} calls in all",
         f"- Stop on visible green: {s['stop_on_green']}",
         f"- Ranking claims: {s['ranking'] or 'none'}",
         f"- Provider-cut cells: {s['provider_cut'] or 'none'}", "",
         "| Arm | Clean | 95% | Usable | Calls | Per pass |", "|---|---|---|---|---|---|"]
    for a, v in s["arms"].items():
        L.append(f"| {a} {v['model'].split('/')[-1].replace(':free', '')} | {v['clean_passes']}/{v['n']} | "
                 f"{round(100 * v['low'])}–{round(100 * v['high'])}% | {v['usable']}"
                 f"{' (stopped)' if v['stopped_for_futility'] else ''} | {v['median_calls']} | {v['calls_per_clean_pass']} |")
    if not s["complete"]:
        L += ["", "**Not complete: every verdict above is provisional and may change.**"]
    return "\n".join(L) + "\n"


def selftest():
    # 1. every combination of inputs gets exactly one known label
    seen = set()
    errs = [None, "harness: boom", "harness: no run.json (agent step rc=124)", "harness: no run.json (agent step rc=1)",
            "run: ConversationRunError: Conversation run failed for id=14433486-b5bb-4cda-850c-52dd19003e19: Remote conversation ended with error",
            "run: ConversationRunError: MaxIterationsReached", "run: RateLimitError 429", "run: ValueError: odd"]
    evsets = [[], [{"kind": "ConversationErrorEvent", "code": "MaxIterationsReached", "detail": "limit (80)"}],
              [{"kind": "ConversationErrorEvent", "code": "LLMAuthenticationError", "detail": "x"}],
              [{"kind": "ConversationErrorEvent", "code": "TypeError", "detail": "set is not JSON serializable"}]]
    for control in (False, True):
        for skipped in (False, True):
            for st in ("ConversationExecutionStatus.FINISHED", "ConversationExecutionStatus.STUCK",
                       "ConversationExecutionStatus.ERROR", ""):
                for e in errs:
                    for ev in evsets:
                        lab = classify({"status": st, "error": e, "skipped": skipped}, ev, control)
                        assert lab in ENDS, lab
                        seen.add(lab)
    assert seen == set(ENDS), set(ENDS) - seen
    # 2. bench-2's three kinds of ending get the labels a person gave them
    stuck = errs[4]
    assert classify({"status": "ConversationExecutionStatus.STUCK", "error": stuck}, []) == "stuck"
    assert classify({"status": "ConversationExecutionStatus.ERROR", "error": errs[5]}, evsets[1]) == "iteration_cap"
    assert classify({"status": "ConversationExecutionStatus.FINISHED", "error": None}, []) == "finished"
    assert classify({"status": "ConversationExecutionStatus.ERROR", "error": "run: x"}, evsets[3]) == "unclassified"
    # 3. intervals and share rules at their edges
    assert abs(interval(3, 3)[0] - 0.2924) < 1e-3 and abs(interval(0, 3)[1] - 0.7076) < 1e-3
    assert abs(interval(0, 6)[1] - 0.4593) < 1e-3 and interval(15, 15)[0] > 0.78
    assert [share_verdict(k, 9) for k in (4, 5, 6, 7, 8)] == ["fails", "inconclusive", "inconclusive", "holds", "holds"]
    assert share_verdict(4, 4) == "inconclusive"
    # 4. plan, futility, provider retries and status on made-up results
    def res(t, a, rep, end, p=False, clean=True, **kw):
        return dict({"task": t, "arm": a, "rep": rep, "end": end, "passed": p, "clean": clean, "outside_paths": [],
                     "calls": 40, "actions": 40, "wall_s": 100.0}, **kw)
    p0, n0 = plan([], 1000)
    assert n0 == 75 and len(p0) == 3 + BATCH_MAX and [c["arm"] for c in p0[:3]] == ["Z"] * 3
    assert [c["arm"] for c in plan([], 10)[0]] == ["Z"] * 3 and len(plan([], 260)[0]) == 3 + 3
    rs = [res(t, "Z", 0, "control") for t in TASKS]
    rs += [res(t, "B", r, "stuck") for t in TASKS for r in (0, 1)]
    assert all(c["arm"] != "B" for c in plan(rs, 1000)[0])               # futility stops B
    rs += [res("T1", "A", 0, "provider_error")] * 2
    assert plan(rs, 1000)[0][0] == {"task": "T1", "arm": "A", "model": ARMS["A"][0], "rep": 0}  # retried
    rs += [res("T1", "A", 0, "provider_error")]
    assert cells(rs)[("T1", "A", 0)] == "provider_cut"
    rs += [res(t, "D", r, "iteration_cap", p=True, green_actions=20, green_kept=True) for t in TASKS for r in range(5)]
    rs += [res("T2", "A", 0, "finished", p=True, clean=False)]
    s = status(rs)
    assert s["arms"]["D"]["clean_passes"] == 15 and s["arms"]["D"]["usable"] == "holds"
    assert s["arms"]["B"]["usable"] == "fails" and s["arms"]["B"]["stopped_for_futility"]
    assert s["arms"]["A"]["clean_passes"] == 0 and "D above B" in s["ranking"] and not s["complete"]
    assert s["stop_on_green"] == {"n": 15, "kept": 15, "median_saving": 0.5, "verdict": "holds"}
    assert s["harness"]["verdict"] == "holds" and s["control_ok"] and s["seal"] == "holds"
    markdown(s)
    print("selftest ok:", len(ENDS), "labels reached")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "selftest":
        selftest()
    elif cmd == "plan":
        todo, open_n = plan(load(sys.argv[2]), int(sys.argv[3]))
        print(json.dumps({"n": len(todo), "open": open_n, "include": todo}))
    elif cmd == "status":
        s = status(load(sys.argv[2]))
        json.dump(s, open(os.path.join(sys.argv[2], "STATUS.json"), "w"), indent=1)
        print(markdown(s), end="")
