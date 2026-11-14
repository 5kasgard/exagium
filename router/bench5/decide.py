"""bench-5: NVIDIA NIM as a second agent pool. Every pre-registered rule as code. Standard library only.

  decide.py selftest                 every ending maps to exactly one label; every rule returns a verdict
  decide.py plan DIR                 JSON list of the open cells (no allowance to read: NIM publishes none)
  decide.py status DIR [BENCH3_DIR]  STATUS.json and STATUS.md (markdown on stdout, JSON to DIR/STATUS.json)

The labels, the interval and the cell rules are bench-3's own code (router/bench3/decide.py), imported, not
copied, so arm N is judged exactly as bench-3's arm A is. Added here: one more provider pattern (credit or
payment refusals), the capacity rule C and the comparison rule P. PREREG.md states the same rules in words;
where the two differ, this file is the rule and the difference is a defect to report.
"""
import datetime, glob, importlib.util, json, math, os, re, statistics, sys

_here = os.path.dirname(os.path.abspath(__file__))
_b3 = next((p for p in (os.path.join(_here, "bench3_rules.py"), os.path.join(_here, "..", "bench3", "decide.py"))
            if os.path.exists(p)), None)
_spec = importlib.util.spec_from_file_location("bench3_rules", _b3)
b3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b3)

MODEL = "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b"
ARMS = {"N": (MODEL, "NVIDIA, served by NVIDIA NIM")}
B3_ARM = "A"                       # the same model on OpenRouter, in bench-3, same harness
TASKS, REPS = ["T1", "T2", "T3"], 7
MAX_TRIES = b3.MAX_TRIES
ENDS, COUNTED, CRASH = b3.ENDS, b3.COUNTED, b3.CRASH
INSTRUMENT = b3.INSTRUMENT
# A refusal: the provider declined to serve (rate, quota, credit, payment or permission), as opposed to a server
# fault. Status codes count only in an HTTP context, so "line 403" or a path holding "/403/" do not match.
# One pattern for both uses (outside review, 2026-11-14): every refusal is a provider_error (not counted, retried),
# and C reads the same pattern.
REFUSAL = re.compile(
    r"(?:Error code|status(?:[ _]code)?|HTTP)\W{0,3}4(?:29|02|03)\b|\b4(?:29|02|03)\W{0,3}(?:Too Many|Payment|Forbidden)"
    r"|RateLimitError|\brate[ -]?limit(?:ed|s)?\b|RESOURCE_EXHAUSTED|insufficient_quota|\bquota (?:exceeded|exhausted)"
    r"|exceeded (?:your |the )?(?:current )?quota|PermissionDenied|PaymentRequired|payment (?:method )?required"
    r"|insufficient (?:credits?|balance)|credits? (?:exhausted|exceeded)|out of credits", re.I)
EXTRA_PROVIDER = REFUSAL
# An attempt C cannot see into: killed at the time cap, or no run record. Throttling could hide in either.
BLIND = re.compile(r"no run\.json")
C_MIN = 21                         # attempts that must be served inside one UTC day for C to hold
ALPHA = 0.05


def _errtext(run, events):
    errs = [e for e in events if e.get("kind") == "ConversationErrorEvent"]
    return (run.get("error") or "") + " " + " ".join(f"{e.get('code')} {e.get('detail')}" for e in errs)


def classify(run, events, control=False):
    """bench-3's labels in bench-3's order, with one more provider pattern checked where bench-3 checks its own."""
    lab = b3.classify(run, events, control)
    if lab in ("unclassified", "iteration_cap", "stuck", "finished") and EXTRA_PROVIDER.search(_errtext(run, events)):
        err = run.get("error") or ""
        if not ("rc=124" in err or "rc=137" in err or err.startswith("harness")):
            return "provider_error"
    return lab


reconcile, cells, interval = b3.reconcile, b3.cells, b3.interval


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*", "*", "result.json"))):
        rp = os.path.join(os.path.dirname(f), "run.json")
        try:
            run = json.load(open(rp)) if os.path.exists(rp) else None
        except (ValueError, OSError):
            run = None
        r = reconcile(json.load(open(f)), run)
        # error text from run.json and from the grader's record of the event-log error codes (audit, 2026-11-14)
        r["_err"] = " ".join(filter(None, [_errtext(run or {}, []) if run else "", str(r.get("error") or ""),
                                           " ".join(map(str, r.get("error_codes") or []))]))
        r["_prompt_tokens"] = ((run or {}).get("tokens") or {}).get("prompt_tokens") if isinstance(run, dict) else None
        out.append(r)
    return out


def plan(results):
    done = cells(results)
    todo = [{"task": t, "arm": "Z", "model": "none", "rep": 0} for t in TASKS if (t, "Z", 0) not in done]
    for rep in range(REPS):
        for t in TASKS:
            if (t, "N", rep) not in done:
                todo.append({"task": t, "arm": "N", "model": MODEL, "rep": rep})
    return todo, len([c for c in todo if c["arm"] != "Z"])


def _day(r):
    s = r.get("started_utc")
    try:
        return datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def capacity(results):
    """C, read from the first registered batch that ran agent attempts. Uncounted attempts count here too:
    capacity is about what the provider served, not about what the worker achieved."""
    agents = [r for r in results if r.get("arm") == "N" and r.get("end") != "skipped_budget" and not r.get("pilot")]
    if not agents:
        return {"verdict": "inconclusive", "why": "no registered agent attempt yet"}
    first = min(agents, key=lambda r: (len(str(r.get("run_id"))), str(r.get("run_id"))))["run_id"]   # numeric order, no int()
    b = [r for r in agents if r.get("run_id") == first]
    days = sorted({_day(r) for r in b} - {None})
    refused = [r for r in b if REFUSAL.search(r.get("_err") or "")]
    blind = [r for r in b if r.get("end") == "timeout" or BLIND.search(r.get("_err") or "")]
    served = sum(r.get("calls") or 0 for r in b)
    out = {"batch": first, "attempts": len(b), "calls_served": served, "refused": len(refused), "blind": len(blind), "utc_days": days,
           "parallel": max((r.get("parallel") or 0) for r in b)}
    if refused:
        v, why = "fails", f"NIM refused {len(refused)} of {len(b)} attempts in the batch"
    elif len(b) >= C_MIN and not blind and len(days) == 1 and all(_day(r) for r in b):
        v, why = "holds", f"{len(b)} attempts ({served} calls) served in one UTC day with no refusal"
    else:
        v, why = "inconclusive", ("fewer than 21 attempts, a timeout or missing record (C can't see into those), "
                                  "a missing start time, or the batch crossed a UTC day")
    return dict(out, verdict=v, why=why)


def fisher(k1, n1, k2, n2):
    """Two-sided Fisher exact p for the 2x2 table [[k1, n1-k1], [k2, n2-k2]]."""
    K, N = k1 + k2, n1 + n2
    if n1 == 0 or n2 == 0:
        return 1.0

    def pmf(x):
        return math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1)
    p0 = pmf(k1)
    lo, hi = max(0, K - n2), min(K, n1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p0 * (1 + 1e-9)))


def compare(n_k, n_n, a_k, a_n, a_complete):
    """P. holds: N usable (lower end >= 50%) and not detectably worse than bench-3 arm A (worse = Fisher p < 0.05
    with N's rate lower; a detectably better N still holds).
    fails: N's upper end under 50%, or N detectably worse than A (p < 0.05 and N's rate lower).
    Inconclusive otherwise, and final only when both sides are complete."""
    lo, hi = interval(n_k, n_n)
    p = fisher(n_k, n_n, a_k, a_n)
    worse = a_n > 0 and n_n > 0 and p < ALPHA and n_k * a_n < a_k * n_n
    if n_n == 0:
        v = "inconclusive"
    elif hi < 0.5 or worse:
        v = "fails"
    elif lo >= 0.5 and a_n > 0:
        v = "holds"
    else:
        v = "inconclusive"
    return {"n_clean": n_k, "n": n_n, "low": round(lo, 3), "high": round(hi, 3), "a_clean": a_k, "a_n": a_n,
            "fisher_p": round(p, 4), "verdict": v, "a_complete": a_complete}


def bench3_arm_a(d):
    if not d or not os.path.isdir(d):
        return 0, 0, False, []
    rs = b3.load(d)
    done = b3.cells(rs)
    got = [v for (t, a, r), v in done.items() if a == B3_ARM and not isinstance(v, str) and v["end"] in COUNTED]
    k = sum(1 for r in got if r["passed"] and r["clean"])
    # complete only when every arm-A cell holds a counted result: a cut cell blocks it, as it blocks bench-3 (audit)
    complete = all(not isinstance(done.get((t, B3_ARM, r), "open"), str) for t in b3.TASKS for r in range(b3.REPS))
    return k, len(got), complete, got


def _per_call(rs, key):
    v = [r[key] / r["calls"] for r in rs if r.get(key) and r.get("calls")]
    return round(statistics.median(v), 1) if v else None


def status(results, b3dir=None):
    done = cells(results)
    counted = [v for v in done.values() if not isinstance(v, str) and v["end"] in COUNTED]
    controls = [v for v in done.values() if not isinstance(v, str) and v["end"] == "control"]
    _, open_n = plan(results)
    icut = any(v == "instrument_cut" for v in done.values())
    rs = [v for (t, a, r), v in sorted(done.items(), key=lambda kv: (kv[0][2], kv[0][0])) if a == "N" and not isinstance(v, str)]
    k = sum(1 for r in rs if r["passed"] and r["clean"])
    ak, an, acomp, arows = bench3_arm_a(b3dir)
    s = {"complete": open_n == 0 and len(controls) == len(TASKS) and not icut, "open_cells": open_n, "counted": len(counted),
         "provider_cut": sorted(k_ for k_, v in done.items() if v == "provider_cut"),
         "instrument_cut": sorted(k_ for k_, v in done.items() if v == "instrument_cut"),
         "control_ok": len(controls) == len(TASKS) and not any(c["passed"] for c in controls),
         "void": any(c["passed"] for c in controls),
         "seal": "holds" if not any(r["outside_paths"] for r in counted) else "broken",
         "egress_drops": sum(r.get("egress_drops") or 0 for r in counted),
         "ends": {e: sum(1 for r in counted if r["end"] == e) for e in ENDS if any(r["end"] == e for r in counted)},
         "instrument_faults": sum(1 for r in results if r["end"] == INSTRUMENT),
         "provider_errors": sum(1 for r in results if r["end"] == "provider_error")}
    ok = sum(1 for r in counted if r["end"] not in CRASH)
    s["harness"] = {"no_crash": ok, "n": len(counted), "verdict": b3.share_verdict(ok, len(counted))}
    s["capacity"] = capacity(results)
    s["compare"] = compare(k, len(rs), ak, an, acomp)
    s["final"] = s["complete"] and acomp and not s["void"]
    s["setup_skips"] = sum(1 for r in results if r["end"] == "skipped_budget")
    s["second_pool"] = ("void" if s["void"] else "yes" if s["capacity"]["verdict"] == "holds" and s["compare"]["verdict"] == "holds" else
                        "no" if "fails" in (s["capacity"]["verdict"], s["compare"]["verdict"]) else "undecided")
    s["report"] = {  # no verdict: reported for routing
        "N": {"median_calls": statistics.median([r["calls"] for r in rs if r.get("calls") is not None] or [0]),
              "wall_s_per_call": _per_call(rs, "wall_s"), "prompt_tokens_per_call": _per_call(rs, "_prompt_tokens")},
        "A": {"median_calls": statistics.median([r["calls"] for r in arows if r.get("calls") is not None] or [0]),
              "wall_s_per_call": _per_call(arows, "wall_s"),
              "prompt_tokens_per_call": _per_call([dict(r, _prompt_tokens=(r.get("tokens") or {}).get("prompt_tokens")) for r in arows], "_prompt_tokens")}}
    return s


def markdown(s):
    c, p = s["capacity"], s["compare"]
    L = ["# bench-5 status (generated by decide.py; not hand-edited)", "",
         *(["**VOID: a no-agent control passed. No verdict below stands.**", ""] if s["void"] else []),
         f"- Model: `{MODEL}` (arm N), against bench-3 arm A (the same model on OpenRouter, same harness)",
         f"- Complete: **{s['complete']}** ({s['counted']} counted attempts, {s['open_cells']} cells open); "
         f"bench-3 arm A complete: **{p['a_complete']}**; verdicts final: **{s['final']}**",
         f"- Control fails all tasks, as required: **{s['control_ok']}**",
         f"- Seal: **{s['seal']}**; blocked outbound connections: {s['egress_drops']}",
         f"- Harness (no crash): {s['harness']['no_crash']}/{s['harness']['n']}, **{s['harness']['verdict']}**",
         f"- Endings: {s['ends']}; provider errors (not counted, retried): {s['provider_errors']}; instrument faults: {s['instrument_faults']}; "
         f"setup or seal failures before any model call (not counted, retried): {s['setup_skips']}",
         f"- **C, capacity: {c['verdict']}.** {c.get('why', '')}. "
         f"Batch {c.get('batch')}: {c.get('attempts')} attempts, {c.get('calls_served')} calls, {c.get('refused')} refused, {c.get('blind')} blind, "
         f"UTC days {c.get('utc_days')}, parallel {c.get('parallel')}",
         f"- **P, pass rate vs arm A: {p['verdict']}.** N {p['n_clean']}/{p['n']} clean "
         f"({round(100 * p['low'])}–{round(100 * p['high'])}%); A {p['a_clean']}/{p['a_n']}; Fisher p = {p['fisher_p']}",
         f"- **Second agent pool: {s['second_pool']}**",
         f"- Provider-cut cells: {s['provider_cut'] or 'none'}; instrument-cut cells: {s['instrument_cut'] or 'none'}", "",
         "Reported, no verdict:", "", "| | Median calls | Wall s per call | Prompt tokens per call |", "|---|---|---|---|"]
    for a, v in s["report"].items():
        L.append(f"| {a} | {v['median_calls']} | {v['wall_s_per_call']} | {v['prompt_tokens_per_call']} |")
    if not s["final"]:
        L += ["", "**Not final: every verdict above is provisional and may change.**"]
    return "\n".join(L) + "\n"


def selftest():
    b3.selftest()   # the imported rules still pass their own tests
    # 1. labels: bench-3's, plus credit refusals as provider errors; timeouts and harness crashes keep their meaning
    seen = set()
    for control in (False, True):
        for skipped in (False, True):
            for st in ("ConversationExecutionStatus.FINISHED", "ConversationExecutionStatus.STUCK", "ConversationExecutionStatus.ERROR", ""):
                for e in (None, "harness: boom", "harness: no run.json (agent step rc=124)", "run: MaxIterationsReached",
                          "run: RateLimitError 429", "run: BadRequestError 402 insufficient credits", "run: ValueError: odd"):
                    lab = classify({"status": st, "error": e, "skipped": skipped}, [], control)
                    assert lab in ENDS, lab
                    seen.add(lab)
    assert seen == set(ENDS), set(ENDS) - seen
    assert classify({"status": "x", "error": "run: APIError 402 Payment Required"}, []) == "provider_error"
    assert classify({"status": "x", "error": "run: ValueError: odd"}, []) == "unclassified"
    assert classify({"status": "x", "error": "harness: no run.json (agent step rc=124) 402"}, []) == "timeout"
    assert classify({"status": "ConversationExecutionStatus.FINISHED", "error": None}, []) == "finished"
    ev = [{"kind": "ConversationErrorEvent", "code": "BadRequestError", "detail": "insufficient credits"}]
    assert classify({"status": "ConversationExecutionStatus.ERROR", "error": "run: x"}, ev) == "provider_error"

    def res(t, rep, end, p=False, clean=True, arm="N", **kw):
        return dict({"task": t, "arm": arm, "rep": rep, "end": end, "passed": p, "clean": clean, "outside_paths": [],
                     "calls": 40, "actions": 40, "wall_s": 800.0, "run_id": "100", "pilot": False,
                     "started_utc": "2026-11-14T10:00:00Z", "parallel": 4, "_err": "", "model": MODEL}, **kw)
    # 2. plan: controls first, then repeat-major order, 21 cells
    p0, n0 = plan([])
    assert n0 == 21 and [c["arm"] for c in p0[:3]] == ["Z"] * 3 and p0[3] == {"task": "T1", "arm": "N", "model": MODEL, "rep": 0}
    assert p0[4]["task"] == "T2" and p0[6]["rep"] == 1
    full = [res(t, 0, "control", arm="Z") for t in TASKS] + [res(t, r, "finished", p=True) for t in TASKS for r in range(REPS)]
    assert plan(full)[1] == 0
    # 3. capacity
    assert capacity(full)["verdict"] == "holds" and capacity(full)["calls_served"] == 21 * 40
    one_refused = full[:-1] + [res("T3", 6, "provider_error", _err="run: RateLimitError 429 Too Many Requests")]
    assert capacity(one_refused)["verdict"] == "fails" and capacity(one_refused)["refused"] == 1
    server_fault = full[:-1] + [res("T3", 6, "provider_error", _err="run: ServiceUnavailable 503")]
    assert capacity(server_fault)["verdict"] == "holds"         # a 5xx is not a refusal; the attempt was still served
    crossed = full[:-1] + [res("T3", 6, "finished", p=True, started_utc="2026-11-15T00:05:00Z")]
    assert capacity(crossed)["verdict"] == "inconclusive"
    short = [r for r in full if not (r["arm"] == "N" and r["rep"] == 6)]
    assert capacity(short)["verdict"] == "inconclusive"
    later = full + [res("T1", 0, "provider_error", run_id="200", _err="429")]
    assert capacity(later)["verdict"] == "holds" and capacity(later)["batch"] == "100"   # only the first batch counts
    pilots = [res("T1", 0, "provider_error", run_id="50", pilot=True, _err="429")] + full
    assert capacity(pilots)["batch"] == "100"                                             # pilots never count
    assert capacity([])["verdict"] == "inconclusive"
    nostart = full[:-1] + [res("T3", 6, "finished", p=True, started_utc=None)]
    assert capacity(nostart)["verdict"] == "inconclusive"
    # 4. Fisher exact against known values
    assert abs(fisher(3, 3, 0, 3) - 0.1) < 1e-9 and fisher(5, 5, 5, 5) == 1.0
    assert abs(fisher(1, 10, 8, 10) - 0.005477) < 1e-5
    assert abs(fisher(10, 21, 15, 15) - 0.000694308) < 1e-8   # scipy.stats.fisher_exact
    assert abs(fisher(13, 21, 15, 15) - 0.011403044) < 1e-8 and abs(fisher(15, 15, 5, 15) - 0.000199900) < 1e-8
    # 5. comparison rule at its edges
    assert compare(21, 21, 15, 15, True)["verdict"] == "holds"
    assert compare(10, 21, 15, 15, True)["verdict"] == "fails"           # detectably worse than A
    assert compare(3, 21, 0, 15, True)["verdict"] == "fails"             # usable fails (upper end under 50%)
    assert compare(13, 21, 15, 15, True)["verdict"] != "holds"
    assert compare(21, 21, 0, 0, False)["verdict"] == "inconclusive"     # no arm-A data: never holds
    assert compare(0, 0, 15, 15, True)["verdict"] == "inconclusive"
    assert compare(15, 15, 5, 15, True)["verdict"] == "holds"            # better than A is not "fails"
    # 6. status and markdown, including void and the provisional flag
    s = status(full)
    assert s["complete"] and not s["final"] and s["second_pool"] == "undecided" and s["capacity"]["verdict"] == "holds"
    assert "Not final" in markdown(s) and s["control_ok"] and s["seal"] == "holds"
    v = status([res("T1", 0, "control", p=True, arm="Z")])
    assert v["void"] and "VOID" in markdown(v)
    s2 = status(one_refused)
    assert s2["second_pool"] == "no" and s2["provider_errors"] == 1 and s2["open_cells"] == 1
    # 7. refusal pattern: matches refusals, not look-alikes (audit and outside review, 2026-11-14)
    for t in ("litellm.RateLimitError: x", "Error code: 429 - {...}", "429 Too Many Requests", "Rate limit exceeded",
              "payment method required", "402 Payment Required", "insufficient credits", "LLMPermissionDeniedError: 403 Forbidden",
              '{"status":429,"title":"Too Many Requests"}', "HTTP 403", "You exceeded your current quota", "RESOURCE_EXHAUSTED"):
        assert REFUSAL.search(t), t
    for t in ("accredited", "credit_limit", "line 403", "/tmp/x/403/y", "disk quota", "forbidden_paths", "ServiceUnavailable 503",
              "Timeout", "run: ValueError: odd", "a1b2403c", "MaxIterationsReached"):
        assert not REFUSAL.search(t), t
    assert classify({"status": "x", "error": "run: LLMPermissionDeniedError: 403 Forbidden"}, []) == "provider_error"
    # 8. C can't see into timeouts or missing records; refusals in the grader's error codes count; ids order numerically
    tmo = full[:-1] + [res("T3", 6, "timeout", _err="harness: no run.json (agent step rc=124)")]
    assert capacity(tmo)["verdict"] == "inconclusive" and capacity(tmo)["blind"] == 1
    ev_only = [dict(r, _err=r["_err"] + " RateLimitError") if r["task"] == "T1" and r["rep"] == 3 and r["arm"] == "N" else r for r in full]
    assert capacity(ev_only)["verdict"] == "fails"
    ids = full + [res("T1", 0, "finished", p=True, run_id="99")]
    assert capacity(ids)["batch"] == "99"                                    # "99" < "100" numerically, not as text
    skip = full[:-1] + [res("T3", 6, "skipped_budget", calls=None)]
    assert capacity(skip)["attempts"] == 20 and capacity(skip)["verdict"] == "inconclusive"
    assert status(skip)["setup_skips"] == 1 and status(skip)["open_cells"] == 1
    # 9. a void run never reports a pool decision
    assert status([res("T1", 0, "control", p=True, arm="Z")])["second_pool"] == "void"
    # 10. instrument faults use bench-3's amended rule
    assert reconcile(res("T1", 0, "unclassified"), {"model": "$B_MODEL"})["end"] == INSTRUMENT
    print("selftest ok: bench-5 rules, on bench-3's labels")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "selftest":
        selftest()
    elif cmd == "plan":
        todo, open_n = plan(load(sys.argv[2]))
        print(json.dumps({"n": len(todo), "open": open_n, "include": todo}))
    elif cmd == "status":
        s = status(load(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
        json.dump(s, open(os.path.join(sys.argv[2], "STATUS.json"), "w"), indent=1, default=str)
        print(markdown(s), end="")
