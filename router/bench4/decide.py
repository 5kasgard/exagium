"""bench-4: every pre-registered rule as code. Standard library only.

  decide.py selftest             every ending maps to exactly one label; every rule returns a verdict
  decide.py plan DIR REMAINING   JSON list of the next attempts that fit in REMAINING free requests
  decide.py status DIR           STATUS.json (to DIR) and STATUS.md text (on stdout)

The model is fixed; the arms are harnesses. Where PREREG.md and this file differ, this file is the rule.
"""
import glob, json, math, os, random, statistics, sys

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
BASE = "openhands"                       # the current harness, the one every other arm is compared with
ARMS = ["openhands", "mini", "goose", "pi", "opencode"]
TASKS, REPS = ["T1", "T2", "T3"], 3
CAP, MARGIN, BATCH_MAX = 80, 20, 10
FUTILITY_N, MAX_TRIES = 6, 3
SEED, B = 20261113, 10000
ENDS = ["control", "skipped_budget", "seal_failed", "harness_crash", "provider_error", "call_cap", "timeout", "exited"]
COUNTED = {"harness_crash", "call_cap", "timeout", "exited"}


def classify(run, calls, passed, control=False):
    """One label per attempt; the first matching line wins."""
    if control:
        return "control"
    if run.get("skipped"):
        return "skipped_budget"
    if run.get("ending") == "seal_failed":
        return "seal_failed"
    if not run or run.get("ending") in (None, "start_failed"):
        return "harness_crash"
    # only the provider's own trouble (rate limit, server error, a stream that broke or carried an error) is
    # retried; any other refusal of a harness's request (a 4xx it caused) is that harness's failure
    upstream_bad = any(isinstance(c.get("status"), int) and (c["status"] == 429 or c["status"] >= 500 or c.get("stream_error")
                                                              or c.get("truncated")) for c in calls)
    if upstream_bad and not passed:
        return "provider_error"
    return {"call_cap": "call_cap", "timeout": "timeout", "exited": "exited"}.get(run["ending"], "harness_crash")


def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def interval(k, n, alpha=0.05):
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


def fisher_less(a, n1, b, n2):
    """One-sided Fisher exact p that arm 1's rate (a/n1) is below arm 2's (b/n2)."""
    K, N = a + b, n1 + n2
    def hyp(x):
        return math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
    return sum(hyp(x) for x in range(max(0, K - n2), a + 1))


def load(d):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*", "*", "result.json")))]


def cells(results):
    done, tries = {}, {}
    for r in results:
        key = (r["task"], r["arm"], r["rep"])
        if r["end"] in COUNTED or r["end"] == "control":
            done.setdefault(key, r)
        elif r["end"] == "provider_error":
            tries[key] = tries.get(key, 0) + 1
    for key, n in tries.items():
        if key not in done and n >= MAX_TRIES:
            done[key] = "provider_cut"
    return done


def arm_rows(done, arm):
    return [v for (t, a, r), v in sorted(done.items(), key=lambda kv: (kv[0][2], kv[0][0])) if a == arm and v != "provider_cut"]


def futile(done, arm):
    rs = arm_rows(done, arm)
    return arm != BASE and len(rs) >= FUTILITY_N and not any(r["passed"] for r in rs[:FUTILITY_N])


def plan(results, remaining):
    done = cells(results)
    todo = [{"task": t, "arm": "control", "rep": 0} for t in TASKS if (t, "control", 0) not in done]
    for rep in range(REPS):
        for t in TASKS:
            for a in ARMS:
                if (t, a, rep) not in done and not futile(done, a):
                    todo.append({"task": t, "arm": a, "rep": rep})
    n_agents = max(0, min(BATCH_MAX, (remaining - MARGIN) // CAP))
    out, used = [], 0
    for c in todo:
        if c["arm"] == "control":
            out.append(c)
        elif used < n_agents:
            out.append(c); used += 1
    return out, len([c for c in todo if c["arm"] != "control"])


def per_pass(rows, key):
    passes = sum(r["passed"] for r in rows)
    total = sum(key(r) for r in rows)
    return total / passes if passes else math.inf


def boot_ratio(rx, rb, key):
    """Percentile 95% interval of (cost per pass of X) / (cost per pass of BASE), resampling attempts within each task."""
    rng = random.Random(SEED)
    by = lambda rows: {t: [r for r in rows if r["task"] == t] for t in TASKS}
    bx, bb = by(rx), by(rb)
    vals = []
    for _ in range(B):
        sx = [rng.choice(v) for t, v in bx.items() for _ in v]
        sb = [rng.choice(v) for t, v in bb.items() for _ in v]
        a, b = per_pass(sx, key), per_pass(sb, key)
        vals.append(math.inf if math.isinf(a) else (0.0 if math.isinf(b) else a / b))
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def tokens_known(rows):
    calls = sum(r["calls_ok"] for r in rows)
    missing = sum(r["tokens"]["calls_missing_usage"] for r in rows)
    return calls > 0 and missing <= 0.10 * calls


def status(results):
    done = cells(results)
    counted = [v for v in done.values() if v != "provider_cut" and v["end"] in COUNTED]
    controls = [v for v in done.values() if v != "provider_cut" and v["end"] == "control"]
    _, open_n = plan(results, 0)
    s = {"model": MODEL, "complete": open_n == 0 and len(controls) == len(TASKS), "open_cells": open_n, "counted": len(counted),
         "void": any(c["passed"] for c in controls),
         "control_ok": len(controls) == len(TASKS) and not any(c["passed"] for c in controls),
         "provider_cut": sorted(k for k, v in done.items() if v == "provider_cut"),
         "seal_checks_failed": sum(1 for r in results if r["end"] == "seal_failed"),
         "container_connections_refused": sum((r.get("egress") or {}).get("drops") or 0 for r in counted)}
    rb = arm_rows(done, BASE)
    tok = lambda r: r["tokens"]["total"]
    cal = lambda r: r["calls"]
    s["arms"], s["switch_candidates"] = {}, []
    for a in ARMS:
        rs = arm_rows(done, a)
        k = sum(r["passed"] for r in rs)
        lo, hi = interval(k, len(rs))
        v = {"passes": k, "n": len(rs), "low": round(lo, 3), "high": round(hi, 3),
             "tokens_known": tokens_known(rs) if rs else False,
             "tokens_per_pass": None if not k else round(per_pass(rs, tok)),
             "calls_per_pass": None if not k else round(per_pass(rs, cal), 1),
             "median_tokens": statistics.median([tok(r) for r in rs]) if rs else None,
             "median_calls": statistics.median([cal(r) for r in rs]) if rs else None,
             "median_wall_s": statistics.median([r["wall_s"] or 0 for r in rs]) if rs else None,
             "stopped_for_futility": futile(done, a),
             "ends": {e: sum(1 for r in rs if r["end"] == e) for e in ENDS if any(r["end"] == e for r in rs)}}
        if a != BASE and rs and rb:
            v["worse_p"] = round(fisher_less(k, len(rs), sum(r["passed"] for r in rb), len(rb)), 4)
            v["worse"] = v["worse_p"] < 0.05
            if not sum(r["passed"] for r in rb):
                v["token_ratio"], v["token_ratio_95"], v["cheaper"] = None, None, "undefined: the base never passed"
            elif v["tokens_known"] and tokens_known(rb):
                tl, th = boot_ratio(rs, rb, tok)
                pb = per_pass(rb, tok)
                v["token_ratio"] = None if math.isinf(pb) or not k else round(per_pass(rs, tok) / pb, 3)
                v["token_ratio_95"] = [tl, th]
                v["cheaper"] = th < 1.0
            else:
                v["token_ratio"], v["token_ratio_95"], v["cheaper"] = None, None, "unmeasured"
            cl, ch = boot_ratio(rs, rb, cal)
            pb = per_pass(rb, cal)
            v["call_ratio"] = None if math.isinf(pb) or not k else round(per_pass(rs, cal) / pb, 3)
            v["call_ratio_95"] = [cl, ch]
            if v["cheaper"] is True and not v["worse"] and v["token_ratio"] is not None and v["token_ratio"] <= 1 / 3:
                s["switch_candidates"].append(a)
        s["arms"][a] = v
    s["switch_to"] = (min(s["switch_candidates"], key=lambda a: s["arms"][a]["token_ratio"])
                      if s["complete"] and not s["void"] and s["switch_candidates"] else None)
    return s


def markdown(s):
    L = ["# bench-4 status (generated by decide.py; not hand-edited)", "",
         *(["**VOID: a no-agent control passed. No verdict below stands.**", ""] if s["void"] else []),
         f"- Model, the same for every arm: `{s['model']}`",
         f"- Complete: **{s['complete']}** ({s['counted']} counted attempts, {s['open_cells']} cells open)",
         f"- Control fails all tasks, as required: **{s['control_ok']}**",
         f"- Seal checks failed: {s['seal_checks_failed']}; container connections refused: {s['container_connections_refused']}",
         f"- Provider-cut cells: {s['provider_cut'] or 'none'}",
         f"- Switch candidates (cheaper by tokens per pass at 95%, not worse, at most 1/3 the tokens): {s['switch_candidates'] or 'none'}",
         f"- **Switch the default harness to: {s['switch_to'] or 'no switch'}**", "",
         "| Harness | Passes | 95% | Tokens/pass | Calls/pass | Token ratio (95%) |", "|---|---|---|---|---|---|"]
    for a, v in s["arms"].items():
        tr = "base" if a == BASE else (f"{v.get('token_ratio')} ({', '.join(str(round(x, 2)) if x != math.inf else 'inf' for x in v['token_ratio_95'])})"
                                         if v.get("token_ratio_95") else str(v.get("cheaper", "")))
        L.append(f"| {a}{' (stopped)' if v['stopped_for_futility'] else ''} | {v['passes']}/{v['n']} | {round(100 * v['low'])}–{round(100 * v['high'])}% | "
                 f"{v['tokens_per_pass']} | {v['calls_per_pass']} | {tr} |")
    if not s["complete"]:
        L += ["", "**Not complete: every verdict above is provisional and may change.**"]
    return "\n".join(L) + "\n"


def selftest():
    calls_ok = [{"status": 200}]
    seen = set()
    for control in (False, True):
        for run in ({}, {"skipped": True}, {"ending": "seal_failed"}, {"ending": "start_failed"}, {"ending": "exited"}, {"ending": "call_cap"},
                    {"ending": "timeout"}, {"ending": "weird"}):
            for calls in ([], calls_ok, [{"status": 429}], [{"status": "refused_cap"}]):
                for passed in (False, True):
                    lab = classify(run, calls, passed, control)
                    assert lab in ENDS, lab
                    seen.add(lab)
    assert seen == set(ENDS), set(ENDS) - seen
    assert classify({"ending": "exited"}, [{"status": 502}], True) == "exited"          # a pass despite a hiccup counts
    assert classify({"ending": "exited"}, [{"status": 502}], False) == "provider_error"
    assert classify({"ending": "call_cap"}, [{"status": "refused_cap"}], False) == "call_cap"
    assert classify({"ending": "exited"}, [{"status": 400}], False) == "exited"                 # a 4xx it caused is its own
    assert classify({"ending": "exited"}, [{"status": 200, "truncated": True}], False) == "provider_error"
    assert abs(fisher_less(0, 9, 9, 9) - 1 / math.comb(18, 9)) < 1e-12 and abs(fisher_less(9, 9, 0, 9) - 1.0) < 1e-9
    assert fisher_less(5, 9, 9, 9) < 0.05 < fisher_less(7, 9, 9, 9)

    def res(t, a, rep, end, p=False, tok=100000, calls=40, miss=0):
        return {"task": t, "arm": a, "rep": rep, "end": end, "passed": p, "calls": calls, "calls_ok": calls, "wall_s": 100.0,
                "tokens": {"total": tok, "calls_missing_usage": miss}, "egress": {"check_ok": True, "drops": 0}}
    p0, n0 = plan([], 1000)
    assert n0 == 45 and len(p0) == 3 + BATCH_MAX and [c["arm"] for c in p0[:3]] == ["control"] * 3
    assert len(plan([], 99)[0]) == 3 and len(plan([], 100)[0]) == 4
    rs = [res(t, "control", 0, "control") for t in TASKS]
    rs += [res(t, "openhands", r, "exited", p=True, tok=800000, calls=40) for t in TASKS for r in range(REPS)]
    rs += [res(t, "goose", r, "exited", p=True, tok=40000, calls=20) for t in TASKS for r in range(REPS)]
    rs += [res(t, "opencode", r, "timeout", p=(r == 0), tok=1500000) for t in TASKS for r in range(REPS)]
    rs += [res(t, "mini", r, "exited", p=True, tok=600000, miss=40) for t in TASKS for r in range(REPS)]
    rs += [res(t, "pi", r, "call_cap") for t in TASKS for r in (0, 1)]
    s = status(rs)
    g = s["arms"]["goose"]
    assert g["token_ratio"] == 0.05 and g["cheaper"] is True and not g["worse"] and "goose" in s["switch_candidates"]
    assert s["arms"]["opencode"]["worse"] and s["arms"]["opencode"]["cheaper"] is False
    assert s["arms"]["mini"]["cheaper"] == "unmeasured" and "mini" not in s["switch_candidates"]
    assert s["arms"]["pi"]["stopped_for_futility"] and s["complete"] and s["switch_to"] == "goose"  # a futile arm's cells close
    assert status(rs + [res("T1", "control", 0, "control", p=True)])["void"] is False             # first control stands
    assert status([res("T1", "control", 0, "control", p=True)])["void"]
    full = rs[:-6] + [res(t, "pi", r, "exited", p=True, tok=30000, calls=30) for t in TASKS for r in range(REPS)]
    s2 = status(full)
    assert s2["complete"] and s2["switch_to"] == "pi" and set(s2["switch_candidates"]) == {"goose", "pi"}
    markdown(s); markdown(s2)

    # Added 2026-11-13 after a mutation check (60 of 163 mutants killed). Tests only: no rule above changed.
    # Each line names the break it catches. Expected values come from scipy or are derived by hand, not from this file.
    assert classify({"ending": "exited"}, [{"status": 500}], False) == "provider_error"     # 5xx boundary: >= 500, not > 500
    assert classify({"ending": "exited"}, [{"status": 499}], False) == "exited"             # a 4xx stays the harness's own
    for (k, n), (lo, hi) in {(0, 9): (0.0, 0.3363), (5, 9): (0.2120, 0.8630), (9, 9): (0.6637, 1.0)}.items():
        got = interval(k, n)                                                               # Clopper-Pearson, scipy reference
        assert abs(got[0] - lo) < 5e-4 and abs(got[1] - hi) < 5e-4, (k, n, got)
    assert abs(fisher_less(3, 9, 9, 9) - 0.0045249) < 1e-6                                 # scipy fisher_exact, alternative=less
    pe = lambda t, a, rep: res(t, a, rep, "provider_error")
    d2 = cells([pe("T1", "goose", 0)] * 2)
    assert ("T1", "goose", 0) not in d2                                                    # 2 provider errors: retried, not cut
    d3 = cells([pe("T1", "goose", 0)] * 3)
    assert d3[("T1", "goose", 0)] == "provider_cut"                                        # the 3rd cuts the cell
    assert cells([pe("T1", "goose", 0)] * 3 + [res("T1", "goose", 0, "exited", p=True)])[("T1", "goose", 0)] != "provider_cut"  # a counted attempt wins
    s3 = status(rs + [pe("T2", "goose", 0)] * 3 + [res("T3", "goose", 0, "seal_failed")])
    assert s3["provider_cut"] == [] and s3["seal_checks_failed"] == 1                      # goose cells were already counted; seal fails are tallied
    late = [res(t, "pi", r, "exited", p=(t == "T1" and r == 2)) for t in TASKS for r in range(REPS)]
    assert futile(cells(late), "pi")                                                       # futility reads reps 0-1 first; a pass in rep 2 is too late
    early = [res(t, "pi", r, "exited", p=(t == "T1" and r == 1)) for t in TASKS for r in range(REPS)]
    assert not futile(cells(early), "pi")
    assert not futile(cells([res(t, "openhands", r, "exited") for t in TASKS for r in range(REPS)]), "openhands")  # the base is never stopped

    def arm(a, tok, passes):
        return [res(t, a, r, "exited", p=(r < passes), tok=tok) for t in TASKS for r in range(REPS)]
    base = [res(t, "control", 0, "control") for t in TASKS] + arm("openhands", 800000, 3)
    rest = lambda **kw: [x for a in ("mini", "goose", "pi", "opencode") for x in arm(a, kw.get(a, 800000), kw.get(a + "_p", 3))]
    s4 = status(base + rest(goose=240000))                                                 # ratio 0.30: at most 1/3, a candidate
    assert s4["arms"]["goose"]["token_ratio"] == 0.3 and s4["switch_to"] == "goose"
    s5 = status(base + rest(goose=272000))                                                 # ratio 0.34: above 1/3, not a candidate
    assert s5["arms"]["goose"]["cheaper"] is True and s5["switch_to"] is None
    s6 = status(base + rest(goose=24000, goose_p=1))                                       # 3/9 vs 9/9: cheaper per pass but worse
    g6 = s6["arms"]["goose"]
    assert g6["worse"] and g6["cheaper"] is True and g6["token_ratio"] == 0.09 and s6["switch_to"] is None
    s7 = status(base + rest(goose=1200000))                                                # ratio 1.5 everywhere: interval [1.5, 1.5]
    assert s7["arms"]["goose"]["cheaper"] is False                                         # "cheaper" needs the upper end below 1
    voided = [res("T1", "control", 0, "control", p=True)] + base[1:] + rest(goose=240000)
    sv = status(voided)
    assert sv["void"] and not sv["control_ok"] and sv["switch_to"] is None                 # a passing control voids any switch
    assert not status(base[1:] + rest(goose=240000))["control_ok"]                         # a missing control is not "control fails all"
    part = status(base + arm("goose", 240000, 3))
    assert not part["complete"] and part["switch_to"] is None                              # open cells: no verdict yet
    assert [c["arm"] for c in plan(base, 1000)[0]].count("control") == 0                   # controls already run are not re-run
    for miss, known in ((4, True), (5, False)):                                            # usage missing on at most 10% of calls
        assert tokens_known([res("T1", "mini", 0, "exited", calls=40, miss=miss)]) is known
    assert tokens_known([res("T1", "mini", 0, "exited", calls=0)]) is False                # no calls: tokens unknown, not "all known"
    sc = status(rs + [pe("T1", "pi", 2)] * 3)
    assert sc["provider_cut"] == [("T1", "pi", 2)] and sc["counted"] == 42      # 9+9+9+9+6 agent attempts; a cut cell is not counted
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
        json.dump(s, open(os.path.join(sys.argv[2], "STATUS.json"), "w"), indent=1, default=str)
        print(markdown(s), end="")
