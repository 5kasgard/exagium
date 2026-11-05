#!/usr/bin/env python3
"""forge-probe: measure a machine's invariants and write a machine profile.

Runs kernels.c measurements repeatedly, in a shuffled interleaved order, and records
median, standard error and spread for each. Detects which meter tiers this machine
offers by trying each one, not by assuming. Every number in the output says where it
came from: "measured" (timed here), "reported" (what the OS/CPU claims), or "derived"
(arithmetic on measured numbers).

    python3 probe.py                 # full run (a few minutes)
    python3 probe.py --quick         # ~30 s smoke run, fewer sizes and reps
    python3 probe.py --out p.json
Stdlib only. Needs a C compiler (cc/gcc/clang).
"""
import argparse, glob, hashlib, json, math, os, platform, random, shutil, statistics
import subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "forge.machine_profile/0"


# ---------------------------------------------------------------- stats
def se_median_bootstrap(xs, resamples=1000):
    """Bootstrap standard error of the median. Fixed seed, so --reanalyse reproduces it.
    (Plateaus are fitted to medians; the SE of the mean understates their noise.)"""
    n = len(xs)
    if n < 2:
        return None
    r = random.Random(0)
    meds = [statistics.median(r.choices(xs, k=n)) for _ in range(resamples)]
    return statistics.stdev(meds)


def summarise(xs):
    """Median (+ its bootstrap SE), mean, sample SD, SE of the mean, CV. n>=1.
    Undefined spreads (one rep) are null, never NaN or a guess."""
    n = len(xs)
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else None
    se = sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "median": statistics.median(xs), "se_median": se_median_bootstrap(xs),
            "mean": mean, "sd": sd, "se": se,
            "cv": sd / mean if n > 1 and mean else None, "min": min(xs), "max": max(xs)}


# ---------------------------------------------------------------- level detection
# How memory levels are read off the latency curve. All of it is recorded in the profile.
#  - Physics first: latency cannot fall as the working set grows, so a point significantly slower
#    than the next kept size is interference and is excluded (and listed). If more than
#    MAX_EXCLUDED of the curve breaks this, no levels are fitted at all.
#  - Piecewise-linear fit of log latency on log size, BIC, each point weighted by its own noise
#    (bootstrap SE of the median), widened to Student-t tails because it comes from few reps.
#    Real plateaus drift a little (associativity, TLB), so a plateau is one sloped segment.
#  - A level changes little compared with the jump at each end: its fitted rise across its span
#    must be significantly less than LEVEL_RISE_RATIO of each edge jump (owner's 2-SE rule,
#    Bonferroni-corrected because breakpoints are chosen from the data), and so must the plain
#    spread of its medians, which the weighting cannot hide; and per unit of size its slope must
#    be under the same fraction of the edge's slope (so a long ramp is never an "edge"). A lone segment must
#    show no detectable rise. Neighbours that would pass as one level are merged first (a precise
#    plateau split at a small step). No cache sizes are assumed.
#  - Explicit, local choices (recorded): MIN_LEVEL_SPAN, LEVEL_RISE_RATIO, MAX_EXCLUDED.
#  - Every doubt resolves toward "transition", never toward "level"; a capacity whose far end
#    was not found is reported open (null), never as "end of sweep".
MIN_LEVEL_SPAN = 2.0
LEVEL_RISE_RATIO = 0.5
LEVEL_METHOD = ("points significantly slower than the next kept size excluded as interference (latency cannot fall "
                "with working set); piecewise-linear BIC fit of log(median latency) on log(size), weights 1/SE^2 with "
                "SE = bootstrap SE of the median widened by t(df=reps-1)/z; a level spans >= "
                f"{MIN_LEVEL_SPAN:g}x size and its fitted rise is significantly less than {LEVEL_RISE_RATIO:g}x the "
                "jump at each edge (z = 2-sided 5% Bonferroni over n-1 breakpoints), as are the plain spread of its interior medians "
                "and its slope relative to the edge's slope; the edge must step up toward larger sizes; a lone segment must show no "
                "detectable rise; neighbours that would pass as one merged level are merged; capacity bracket starts at the last size where no rep already reached the next "
                "level and ends at the first size where no rep was still at this level (null if no next level); a bracket "
                "covers what these reps showed, it is not a confidence interval for the true capacity")


def t_cdf(t, df):
    """Student-t CDF by Simpson integration of the density (stdlib only)."""
    if t < 0:
        return 1 - t_cdf(-t, df)
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
    n = 2000; h = t / n
    s = f(0) + f(t) + sum((4 if k % 2 else 2) * f(k * h) for k in range(1, n))
    return 0.5 + s * h / 3


_tq = {}


def t_quantile(p, df):
    """Inverse Student-t CDF by bisection; df=None means normal."""
    if df is None:
        from statistics import NormalDist
        return NormalDist().inv_cdf(p)
    key = (round(p, 12), df)
    if key not in _tq:
        lo, hi = 0.0, 1.0
        while t_cdf(hi, df) < p:
            hi *= 2
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
        _tq[key] = (lo + hi) / 2
    return _tq[key]


def merge_z(n, df=None):
    """Threshold for 'these two neighbouring segments differ' (Bonferroni over n-1 breakpoints)."""
    return t_quantile(1 - 0.025 / max(n - 1, 1), df)


def segment_lines(x, y, w, penalty_n):
    """Piecewise-linear fit of y on x (log latency on log size), weights w, segment count by
    known-variance BIC: chi-square + (3k-1) ln n (a line per segment plus k-1 breakpoints).
    A drifting plateau is one segment, not a staircase. Returns [(start, end_inclusive)]."""
    n = len(y)
    memo = {}

    def chi2(i, j):
        """Weighted residual of the best line through points i..j, computed directly (centred,
        two-pass) so weights that differ by many orders of magnitude cannot cancel to zero."""
        if j - i < 2:
            return 0.0  # one or two points: a line fits exactly
        if (i, j) not in memo:
            ws, xs, ys = w[i:j + 1], x[i:j + 1], y[i:j + 1]
            W = sum(ws); xm = sum(a * b for a, b in zip(ws, xs)) / W; ym = sum(a * b for a, b in zip(ws, ys)) / W
            sxx = sum(a * (b - xm) ** 2 for a, b in zip(ws, xs))
            sl = sum(a * (b - xm) * (c - ym) for a, b, c in zip(ws, xs, ys)) / sxx if sxx > 0 else 0.0
            memo[(i, j)] = sum(a * (c - ym - sl * (b - xm)) ** 2 for a, b, c in zip(ws, xs, ys))
        return memo[(i, j)]

    kmax = min(n, 16)
    INF = float("inf")
    dp = [[INF] * n for _ in range(kmax + 1)]
    back = [[-1] * n for _ in range(kmax + 1)]
    for j in range(n):
        dp[1][j] = chi2(0, j)
    for k in range(2, kmax + 1):
        for j in range(k - 1, n):
            for i in range(k - 1, j + 1):
                c = dp[k - 1][i - 1] + chi2(i, j)
                if c < dp[k][j]:
                    dp[k][j], back[k][j] = c, i
    best_k, best_bic = 1, INF
    for k in range(1, kmax + 1):
        bic = dp[k][n - 1] + (3 * k - 1) * math.log(penalty_n)
        if bic < best_bic:
            best_k, best_bic = k, bic
    segs, j, k = [], n - 1, best_k
    while k >= 1:
        i = back[k][j] if k > 1 else 0
        segs.append((i, j)); j, k = i - 1, k - 1
    return segs[::-1]


MAX_EXCLUDED = 0.2


def nonmonotone_points(med, se, df=None):
    """Latency cannot fall as the working set grows (more data never makes a dependent load
    faster). Walking back from the largest size, a point significantly slower than the next kept
    point is interference (a co-tenant burst, say), not structure. Returns excluded indices.
    Same corrected threshold as the level tests, so only clear violations are excluded."""
    n = len(med)
    infl = t_quantile(0.975, df) / t_quantile(0.975, None)
    rel = [infl * s / v for s, v in zip(se, med)]
    zm = merge_z(n, None)
    out, nxt = [], n - 1
    for k in range(n - 2, -1, -1):
        if math.log(med[k]) - math.log(med[nxt]) > zm * math.hypot(rel[k], rel[nxt]):
            out.append(k)
        else:
            nxt = k
    return sorted(out)


def describe_levels(sizes, med, se, df=None, rep_max=None, rep_min=None):
    """Levels and transitions from the latency curve, or None when they cannot be read honestly:
    noise unknown, or more than MAX_EXCLUDED of the curve breaks monotonicity.
    df: degrees of freedom of the noise estimates (reps-1). rep_max / rep_min: per-point slowest /
    fastest rep, used so a capacity bracket never starts after, or ends before, a mixed point.
    Points excluded as non-monotone are left out of the fit (see nonmonotone_points)."""
    if not med or any(s is None or s <= 0 for s in se) or any(m <= 0 for m in med):
        return None
    drop = set(nonmonotone_points(med, se, df))
    if len(drop) > MAX_EXCLUDED * len(med):
        return None
    keep = [k for k in range(len(med)) if k not in drop]
    sizes, med, se = [sizes[k] for k in keep], [med[k] for k in keep], [se[k] for k in keep]
    rep_max = [rep_max[k] for k in keep] if rep_max else None
    rep_min = [rep_min[k] for k in keep] if rep_min else None
    n = len(med)
    x = [math.log(v) for v in sizes]
    y = [math.log(v) for v in med]
    # few reps make each SE an optimistic estimate; widen it to the t distribution's 5% tails so
    # chi-square and BIC are not fooled into extra breakpoints
    infl = t_quantile(0.975, df) / t_quantile(0.975, None)
    rel = [infl * s / v for s, v in zip(se, med)]
    w = [1 / r ** 2 for r in rel]
    zm = merge_z(n, None)  # noise already widened to t tails above; do not apply t twice

    def wmean(i, j):  # weighted mean of log latency and its SE
        W = sum(w[i:j + 1])
        return sum(w[k] * y[k] for k in range(i, j + 1)) / W, 1 / math.sqrt(W)

    def fit(idx):  # weighted LS line of log latency on log size: (slope, slope_se, xm, ym, W, sxx)
        idx = list(idx)
        ws, xs, ys = [w[k] for k in idx], [x[k] for k in idx], [y[k] for k in idx]
        W = sum(ws); xm = sum(a * b for a, b in zip(ws, xs)) / W; ym = sum(a * b for a, b in zip(ws, ys)) / W
        sxx = sum(a * (b - xm) ** 2 for a, b in zip(ws, xs))
        slope = sum(a * (b - xm) * (c - ym) for a, b, c in zip(ws, xs, ys)) / sxx if sxx > 0 else 0.0
        return slope, (1 / math.sqrt(sxx) if sxx > 0 else float("inf")), xm, ym, W, sxx

    def at(idx, x0):  # the line through idx evaluated at x0: (value, SE); one point is itself
        idx = list(idx)
        if len(idx) == 1:
            return y[idx[0]], 1 / math.sqrt(w[idx[0]])
        slope, _, xm, ym, W, sxx = fit(idx)
        return ym + slope * (x0 - xm), math.sqrt(1 / W + (x0 - xm) ** 2 / sxx)

    segs = [list(g) for g in segment_lines(x, y, w, n)]
    cand = [sizes[j] / sizes[i] >= MIN_LEVEL_SPAN * 0.999 for i, j in segs]

    def plateau(t):
        """A level changes little across its span compared with the jump at each end: its fitted
        rise must be significantly less than LEVEL_RISE_RATIO of each edge jump (each segment's
        line at its own edge point; neighbour = next candidate, or the short segments to the curve
        end pooled). On a ramp the jump at a boundary is one step of the ramp while the rise is
        many, so ramps fail; at a real cache boundary the jump dwarfs the drift."""
        i, j = segs[t]
        sl, sl_se = fit(range(i, j + 1))[:2]
        span = x[j] - x[i]
        rise, rise_se = abs(sl) * span, sl_se * span
        # plain spread over interior points: a segment's end points are where reps mix at a
        # boundary (some still in this level, some already in the next) and the fit down-weights
        # them; the guard is for noisy points *inside* a precise level, which this catches
        inner = y[i + 1:j]
        spread = (max(inner) - min(inner)) if inner else 0.0
        if len(segs) == 1:  # nothing to compare with: no detectable rise (uncorrected 5%)
            return rise <= t_quantile(0.975, df) * rise_se
        for step in (-1, 1):
            q = t + step
            if not 0 <= q < len(segs):
                continue
            first = q
            while not cand[q] and 0 <= q + step < len(segs):
                q += step
            if cand[q]:
                lo_, hi_ = segs[q]
            else:
                lo_, hi_ = (segs[q][0], segs[first][1]) if step < 0 else (segs[first][0], segs[q][1])
            mine_x, theirs_x = (x[i], x[hi_]) if step < 0 else (x[j], x[lo_])
            v0, s0 = at(range(i, j + 1), mine_x)
            v1, s1 = at(range(lo_, hi_ + 1), theirs_x)
            # going to larger sizes the neighbour must be slower (and to smaller sizes faster):
            # a real boundary is a step up. A neighbour's line that lands on the wrong side
            # (a curved ramp extrapolated) is no edge at all.
            jump, jump_se = (v1 - v0) * step, math.hypot(s0, s1)
            if jump <= 0:
                return False
            if not (LEVEL_RISE_RATIO * jump - rise > zm * math.hypot(LEVEL_RISE_RATIO * jump_se, rise_se)):
                return False
            # ...and per unit of size the edge must also be steeper than the level: a "jump" that is
            # really a long ramp accumulated over many octaves is shallower than a plateau's own
            # drift and is no edge (fifth review: curved ramps otherwise produced phantom levels)
            dx = abs(theirs_x - mine_x)
            if not (LEVEL_RISE_RATIO * jump / dx - abs(sl) > zm * math.hypot(sl_se, LEVEL_RISE_RATIO * jump_se / dx)):
                return False
            # the weighted fit can be dominated by a few very precise points, so the plain spread of
            # the segment's medians must also stay under the same fraction of the jump
            if spread > LEVEL_RISE_RATIO * jump:
                return False
        return True

    # A precise plateau can be split at a small step. Two neighbours are one level if the step
    # between them is under LEVEL_RISE_RATIO of the merged segment's outer jumps and, merged,
    # they still pass as a level (same rule, no new threshold).
    def step_between(a, b):  # each segment's own line at its own edge; positive = b is slower
        return at(range(b[0], b[1] + 1), x[b[0]])[0] - at(range(a[0], a[1] + 1), x[a[1]])[0]

    merged = True
    while merged and len(segs) > 1:
        merged = False
        for t in range(len(segs) - 1):
            a_, b_ = segs[t], segs[t + 1]
            inner = abs(step_between(a_, b_))
            # outer jumps from each piece's own line to its own neighbour (never from the merged
            # line, which a few very precise points can drag far from the other piece)
            outer = ([step_between(segs[t - 1], a_)] if t > 0 else []) + \
                    ([step_between(b_, segs[t + 2])] if t + 2 < len(segs) else [])
            if not outer or min(outer) <= 0 or inner >= LEVEL_RISE_RATIO * min(outer):
                continue
            keep_segs, keep_cand = [list(g) for g in segs], list(cand)
            segs[t:t + 2] = [[a_[0], b_[1]]]
            cand[t:t + 2] = [sizes[segs[t][1]] / sizes[segs[t][0]] >= MIN_LEVEL_SPAN * 0.999]
            if cand[t] and plateau(t):
                merged = True
                break
            segs[:], cand[:] = keep_segs, keep_cand

    groups = []
    for t, (i, j) in enumerate(segs):
        kind = "level" if cand[t] and plateau(t) else "transition"
        if kind == "transition" and groups and groups[-1][2] == "transition":
            groups[-1][1] = j
        else:
            groups.append([i, j, kind])

    out, k = [], 0
    for idx, (i, j, kind) in enumerate(groups):
        m, s = wmean(i, j)
        lvl = {"kind": kind, "points": j - i + 1, "from_bytes": sizes[i], "to_bytes": sizes[j],
               "latency_ns_range": [min(med[i:j + 1]), max(med[i:j + 1])], "label": "derived"}
        if kind == "level":
            k += 1
            sl = fit(range(i, j + 1))[0] if j > i else 0.0
            lvl.update(level=k, latency_ns=math.exp(m), latency_ns_se=math.exp(m) * s,
                       rise_across_span=math.exp(sl * (x[j] - x[i])))
            if idx + 1 < len(groups):
                nxt_level = next((g for g in groups[idx + 1:] if g[2] == "level"), None)
                lo, hi_i = j, (nxt_level[0] if nxt_level else None)
                if rep_min and nxt_level:
                    # the next level starts at its first point where no rep still ran at this speed
                    mid_m = math.sqrt(math.exp(m) * math.exp(wmean(*nxt_level[:2])[0]))
                    ok = [q for q in range(nxt_level[0], nxt_level[1] + 1) if rep_min[q] > mid_m]
                    hi_i = ok[0] if ok else nxt_level[1]
                if rep_max:
                    # a point where some reps already ran at the next level's speed is past capacity
                    # for those reps: the bracket starts at the last point where none did
                    far = math.exp(wmean(*nxt_level[:2])[0]) if nxt_level else max(med[groups[idx + 1][0]:groups[idx + 1][1] + 1])
                    mid = math.sqrt(math.exp(m) * far)
                    okl = [q for q in range(i, j + 1) if rep_max[q] < mid]
                    lo = okl[-1] if okl else i
                # no next level found: the upper end is unknown, not "end of sweep"
                lvl["capacity_bytes_between"] = [sizes[lo], sizes[hi_i] if hi_i is not None else None]
        out.append(lvl)
    return out


# ---------------------------------------------------------------- environment / reported
def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def parse_size(s):
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(s[:-1]) * mult[s[-1]] if s and s[-1] in mult else int(s)


def reported_caches(root="/"):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "sys/devices/system/cpu/cpu0/cache/index*"))):
        size = read(os.path.join(d, "size"))
        if not size:
            continue
        out.append({"level": int(read(os.path.join(d, "level"), "0")), "type": read(os.path.join(d, "type")),
                    "size_bytes": parse_size(size), "line_bytes": int(read(os.path.join(d, "coherency_line_size"), "0") or 0),
                    "shared_cpu_list": read(os.path.join(d, "shared_cpu_list")), "label": "reported"})
    return out


def cpu_info(root="/"):
    txt = read(os.path.join(root, "proc/cpuinfo"), "") or ""
    model, flags = None, []
    for line in txt.splitlines():
        if line.startswith("model name") and model is None:
            model = line.split(":", 1)[1].strip()
        if line.startswith("flags") and not flags:
            flags = line.split(":", 1)[1].split()
    return model, flags


def mem_available(root="/"):
    for line in (read(os.path.join(root, "proc/meminfo"), "") or "").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def environment(root="/"):
    model, flags = cpu_info(root)
    simd = [f for f in ("sse2", "avx", "avx2", "fma", "avx512f", "amx_tile", "neon", "asimd", "sve") if f in flags]
    return {"hostname_hash": hashlib.sha256(platform.node().encode()).hexdigest()[:12],
            "kernel": platform.release(), "arch": platform.machine(), "cpu_model": model,
            "logical_cpus": os.cpu_count(), "hypervisor": "hypervisor" in flags,
            "simd_flags": simd, "pmu_flag_arch_perfmon": "arch_perfmon" in flags,
            "thp": read(os.path.join(root, "sys/kernel/mm/transparent_hugepage/enabled")),
            "mem_available_bytes": mem_available(root), "label": "reported"}


# ---------------------------------------------------------------- meter tiers
def cpu_ticks(root="/"):
    """(total, steal) jiffies from /proc/stat; steal = time the hypervisor gave our vCPUs to others."""
    f = (read(os.path.join(root, "proc/stat"), "") or "").splitlines()
    v = [int(x) for x in f[0].split()[1:]] if f and f[0].startswith("cpu ") else []
    return (sum(v[:8]), v[7]) if len(v) >= 8 else (None, None)


def run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, "", str(e)


def tier_rapl(root="/"):
    files = sorted(glob.glob(os.path.join(root, "sys/class/powercap/intel-rapl*/energy_uj")))
    if not files:
        return {"available": False, "evidence": "no sys/class/powercap/intel-rapl*/energy_uj"}
    ok = [f for f in files if read(f) is not None and read(f).isdigit()]
    if not ok:
        return {"available": False, "evidence": f"{len(files)} RAPL zones present but not readable (permissions)"}
    return {"available": True, "evidence": f"{len(ok)} readable zone(s)", "zones": [os.path.relpath(f, root) for f in ok]}


def tier_perf(root="/"):
    para = read(os.path.join(root, "proc/sys/kernel/perf_event_paranoid"))
    devs = os.path.join(root, "sys/bus/event_source/devices")
    pmus = sorted(d for d in (os.listdir(devs) if os.path.isdir(devs) else [])
                  if d == "cpu" or d.startswith(("cpu_", "armv8", "arm_", "apple_", "riscv_")))
    hw = {"hardware_pmu": pmus or None}
    exe = shutil.which("perf")
    if not pmus:
        return {"available": False, **hw, "evidence": f"no CPU PMU in sys/bus/event_source/devices; paranoid={para}"}
    if not exe:
        return {"available": False, **hw, "evidence": f"CPU PMU {pmus} present but perf not installed; paranoid={para}"}
    rc, out, err = run([exe, "stat", "-x,", "-e", "instructions,cycles", "true"], timeout=20)
    txt = out + err
    if rc != 0 or "not supported" in txt or "<not counted>" in txt or "Permission" in txt:
        return {"available": False, **hw, "evidence": f"perf present but counters unusable (paranoid={para}): {txt.strip()[:200]}"}
    return {"available": True, **hw, "evidence": f"perf stat counted instructions,cycles (paranoid={para})"}


def nvml_lib_power():
    """Read GPU power through libnvidia-ml directly; None if the library is absent."""
    import ctypes
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError:
        return None
    if lib.nvmlInit_v2() != 0:
        return {"available": False, "evidence": "libnvidia-ml present but nvmlInit failed"}
    try:
        n = ctypes.c_uint(0)
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) != 0 or n.value == 0:
            return {"available": False, "evidence": "libnvidia-ml present but no GPUs"}
        mw = []
        for i in range(n.value):
            h, p = ctypes.c_void_p(), ctypes.c_uint(0)
            if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) == 0 and lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(p)) == 0:
                mw.append(p.value)
        if not mw:
            return {"available": False, "evidence": f"{n.value} GPU(s) but power usage not readable via NVML"}
        return {"available": True, "evidence": f"NVML library read power on {len(mw)} GPU(s)"}
    finally:
        lib.nvmlShutdown()


def tier_nvml(root="/"):
    gpu_dev = bool(glob.glob(os.path.join(root, "dev/nvidia[0-9]*")))
    lib = nvml_lib_power()
    if lib is not None:
        return {**lib, "gpu_device_nodes": gpu_dev}
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "gpu_device_nodes": gpu_dev,
                "evidence": "no libnvidia-ml and no nvidia-smi" + ("; /dev/nvidia* present (driver libs missing?)" if gpu_dev else "")}
    rc, out, err = run([exe, "--query-gpu=name,power.draw", "--format=csv,noheader,nounits"], timeout=20)
    rows = [r for r in out.strip().splitlines() if r]
    try:
        watts = [float(r.split(",")[-1]) for r in rows]
    except ValueError:
        return {"available": False, "evidence": f"nvidia-smi ran but power.draw not numeric: {out.strip()[:120]}"}
    if rc != 0 or not watts:
        return {"available": False, "evidence": f"nvidia-smi failed: {err.strip()[:120]}"}
    return {"available": True, "evidence": f"{len(watts)} GPU(s) report power.draw", "gpus": [r.split(",")[0] for r in rows]}


def tier_cachegrind():
    exe = shutil.which("valgrind")
    if not exe:
        return {"available": False, "evidence": "valgrind not installed"}
    with tempfile.TemporaryDirectory() as td:
        outf = os.path.join(td, "cg.out")
        rc, out, err = run([exe, "--tool=cachegrind", "--cache-sim=yes", f"--cachegrind-out-file={outf}", "/bin/true"], timeout=60)
    if rc == 0 and "I" in err and "refs" in err:
        ver = run([exe, "--version"])[1].strip()
        return {"available": True, "evidence": f"{ver} ran cachegrind with cache simulation on /bin/true"}
    return {"available": False, "evidence": f"valgrind present but cachegrind failed: {err.strip()[-160:]}"}


def tiers(root="/", calibration=None):
    return [
        {"tier": 1, "name": "rapl_energy", "kind": "measured_energy", **tier_rapl(root)},
        {"tier": 1, "name": "nvml_gpu_power", "kind": "measured_energy", **tier_nvml(root)},
        {"tier": 1, "name": "perf_counters", "kind": "measured_counts", **tier_perf(root)},
        {"tier": 2, "name": "timing", "kind": "measured_time", "available": True, "evidence": "CLOCK_MONOTONIC (see timer)"},
        {"tier": 2, "name": "cachegrind", "kind": "simulated_counts", **tier_cachegrind()},
        {"tier": 3, "name": "calibrated_estimate", "kind": "calibrated",
         "available": bool(calibration and os.path.exists(calibration)),
         "evidence": f"calibration file {calibration}" if calibration and os.path.exists(calibration)
         else "no real-hardware calibration yet (open item 5)"},
    ]


# ---------------------------------------------------------------- build + plan
CFLAGS = ["-O2", "-march=native", "-pthread"]


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build(td):
    src = os.path.join(HERE, "kernels.c")
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if not cc:
        sys.exit("no C compiler found")
    exe = os.path.join(td, "kernels")
    rc, out, err = run([cc, *CFLAGS, "-o", exe, src], timeout=120)
    if rc != 0:
        # -march=native is not understood everywhere (e.g. some clang/ARM); fall back, and say so
        rc, out, err = run([cc, "-O2", "-pthread", "-o", exe, src], timeout=120)
        flags = ["-O2", "-pthread"]
        if rc != 0:
            sys.exit("compile failed:\n" + err)
    else:
        flags = CFLAGS
    ver = run([cc, "--version"])[1].splitlines()[0]
    rc, out, _ = run([exe, "isa"])
    isa = json.loads(out)["vector_isa"] if rc == 0 else "unknown"
    return exe, {"compiler": ver, "flags": flags, "kernels_sha256": sha256_file(src),
                 "probe_sha256": sha256_file(__file__), "vector_isa": isa}


def chase_sizes(max_bytes, min_bytes=4 << 10):
    """Geometric sweep, factor sqrt(2), rounded to 4 KiB. No cache sizes assumed."""
    out, s = [], float(min_bytes)
    while s <= max_bytes * 1.0001:
        v = int(round(s / 4096)) * 4096
        if not out or v != out[-1]:
            out.append(v)
        s *= math.sqrt(2)
    return out


def plan(env, caches, quick, min_ms, max_bytes=None):
    ncpu = env["logical_cpus"] or 1
    llc = max((c["size_bytes"] for c in caches), default=32 << 20)
    avail = env["mem_available_bytes"] or (2 << 30)
    # Reported LLC is only a hint for how far to sweep (to pass it into DRAM). It is a VM's
    # claim and is often the whole host's cache, so we sweep 4x past it, capped by RAM.
    cap = max_bytes or min(max(4 * llc, 256 << 20), avail // 4)
    if quick:
        cap = min(cap, 256 << 20)
    sizes = chase_sizes(cap)
    if quick:
        sizes = sizes[::2] + ([sizes[-1]] if len(sizes) % 2 == 0 else [])
    line = next((c["line_bytes"] for c in caches if c["line_bytes"]), 64)
    m = []
    for s in sizes:
        m.append((f"chase:{s}", ["chase", str(s), str(line), str(min_ms), "1"]))
    stream_big = max(cap, 64 << 20)
    for th in sorted({1, ncpu}):
        m.append((f"stream:dram:{th}", ["stream", str(stream_big), str(th), str(min_ms * 3), "1"]))
        m.append((f"stream:l1:{th}", ["stream", str((16 << 10) * th), str(th), str(min_ms), "1"]))
        for op in ("scalar_f64", "vec_f64", "vec_f32"):
            m.append((f"flops:{op}:{th}", ["flops", op, str(th), str(min_ms * 3)]))
    m.append(("timer", ["timer"]))
    return m, {"sweep_max_bytes": cap, "line_bytes": line, "llc_reported_bytes": llc}


def chase_seed(seed, rep, size):
    return int(hashlib.sha256(f"{seed}:{rep}:{size}".encode()).hexdigest()[:15], 16) | 1


# ---------------------------------------------------------------- consistency
def check_raw(raw, settings):
    """Internal consistency of raw records: rep counts match settings, and each rate equals
    its own work divided by its own time. This is NOT proof the raw data is genuine;
    that needs the anchored record (open item 3)."""
    bad = []
    for name, recs in raw.items():
        if len(recs) != settings["reps"]:
            bad.append(f"{name}: {len(recs)} reps, settings say {settings['reps']}")
        for i, x in enumerate(recs):
            k, t = x["kind"], x.get("seconds")
            if k == "timer":
                continue
            if not isinstance(t, (int, float)) or t <= 0:
                bad.append(f"{name} rep {i}: missing or non-positive seconds, rate cannot be checked")
                continue
            if k == "chase":
                want, got = t / x["steps"] * 1e9, x["ns_per_load"]
            elif k == "stream":
                want, got = x["bytes"] * x["passes"] / t / 1e9, x["GBps"]
            else:
                want, got = 2 * x["lanes"] * x["chains"] * x["iters"] * x["threads"] / t / 1e9, x["GFLOPs"]
            if abs(want - got) > 1e-3 * abs(want) + 1e-3:
                bad.append(f"{name} rep {i}: rate {got} but work/time gives {want:.4f}")
            floor = settings["min_ms"] * (1 if k == "chase" or name.startswith("stream:l1") else 3) / 1e3
            if t < floor * 0.999:
                bad.append(f"{name} rep {i}: timed window {t:.4f}s is shorter than the {floor:.4f}s minimum")
            if k == "chase" and x.get("seed") != chase_seed(settings["seed"], i, x["bytes"]):
                bad.append(f"{name} rep {i}: chase seed does not match the recorded run seed")
    return bad


# ---------------------------------------------------------------- digests
DIGEST_SCHEME = ("sha256 of canonical JSON: keys sorted, no whitespace, UTF-8, NaN/Infinity forbidden, "
                 "floats as Python repr (shortest round-trip). Not yet RFC 8785 (JCS): another language must "
                 "format floats the same way to reproduce it")


def canonical(obj):
    """One byte string per value, so a digest names exactly one profile. Fields are never joined
    by a separator (the hash-chain review found a '|'-join collision in an older log)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def add_digests(profile):
    """measurement_sha256 covers what was measured and how (raw reps, settings, build, environment,
    meter tiers, time); profile_sha256 covers everything else as well. Either is what an outside
    record anchors: once anchored, any later edit to the profile shows as a different digest.
    A digest proves the profile was not changed after it was anchored, never that it is genuine."""
    body = {k: v for k, v in profile.items() if k != "digests"}
    meas = {k: profile[k] for k in ("raw", "settings", "build", "environment", "meter_tiers", "created_utc", "elapsed_s")}
    profile["digests"] = {"measurement_sha256": hashlib.sha256(canonical(meas)).hexdigest(),
                          "profile_sha256": hashlib.sha256(canonical(body)).hexdigest(),
                          "scheme": DIGEST_SCHEME, "label": "derived"}
    return profile


# ---------------------------------------------------------------- analysis
def analyse(raw, env, caches, tier_list, build_info, settings, elapsed):
    """Everything in the profile except raw/env/tiers is computed here from raw reps,
    so `--reanalyse` can regenerate (and check) any stored profile."""
    def stat(name, key):
        return summarise([x[key] for x in raw[name]])

    sizes = sorted(int(n.split(":")[1]) for n in raw if n.startswith("chase:"))
    lat = [stat(f"chase:{s}", "ns_per_load") for s in sizes]
    meds = [x["median"] for x in lat]
    df = min(x["n"] for x in lat) - 1 if lat else None
    ses = [x["se_median"] for x in lat]
    noise_known = bool(lat) and all(s is not None and s > 0 for s in ses)
    excluded = [sizes[k] for k in nonmonotone_points(meds, ses, df)] if noise_known else []
    levels = describe_levels(sizes, meds, ses, df=df,
                             rep_max=[x["max"] for x in lat], rep_min=[x["min"] for x in lat])
    if not noise_known:
        level_note = "not computed: needs >= 2 reps per size for noise estimates"
    elif levels is None:
        level_note = (f"not computed: {len(excluded)} of {len(sizes)} sizes are significantly slower than a larger "
                      f"size (more than {MAX_EXCLUDED:.0%}), so the curve is too disturbed to read levels from")
    else:
        level_note = LEVEL_METHOD
    llc = settings.get("llc_reported_bytes") or 0
    reach = sizes[-1] / llc if llc and sizes else None

    ncpu = env["logical_cpus"] or 1
    thr = sorted({1, ncpu})
    bw = {f"{k}_{th}t": {**stat(f"stream:{k}:{th}", "GBps"), "unit": "GB/s", "label": "measured",
                         "bytes": raw[f"stream:{k}:{th}"][0]["bytes"]}
          for k in ("dram", "l1") for th in thr}
    fl = {f"{op}_{th}t": {**stat(f"flops:{op}:{th}", "GFLOPs"), "unit": "GFLOP/s", "label": "measured"}
          for op in ("scalar_f64", "vec_f64", "vec_f32") for th in thr}
    timer = {"min_step_ns": stat("timer", "min_step_ns"), "read_overhead_ns": stat("timer", "read_overhead_ns"),
             "getres_ns_reported": raw["timer"][0]["getres_ns"], "label": "measured",
             "meaning": "min_step_ns is the smallest gap two reads can show (bounded by read cost), "
                        "so timed windows should be >> it; getres_ns_reported is the kernel's claim"}
    lv = [l for l in levels or [] if l["kind"] == "level"]
    ratio = ({"value": lv[-1]["latency_ns"] / lv[0]["latency_ns"], "label": "derived",
              "from": ["memory_levels first level", "memory_levels last level"],
              "meaning": "cost of a load from the farthest level found vs the nearest"}
             if len(lv) >= 2 else None)
    peak_f64 = fl[f"vec_f64_{ncpu}t"]["median"]
    dram_bw = bw[f"dram_{ncpu}t"]["median"]
    profile = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": elapsed,
        "settings": settings,
        "build": build_info,
        "environment": env,
        "meter_tiers": tier_list,
        "best_tier_for_energy": next((t["name"] for t in tier_list if t["available"] and t["kind"] in ("measured_energy",)), None),
        "reported_caches": caches,
        "latency_curve": [{"bytes": s, "ns_per_load": x, "label": "measured"} for s, x in zip(sizes, lat)],
        "memory_levels": levels,  # fitted from the curve: label "derived"
        "memory_levels_excluded_bytes": excluded,
        "memory_levels_choices": {"min_level_span": MIN_LEVEL_SPAN, "level_rise_ratio": LEVEL_RISE_RATIO,
                                  "max_excluded_fraction": MAX_EXCLUDED, "label": "explicit choice",
                                  "meaning": "a level spans >= min_level_span x in size; its rise must be under "
                                             "level_rise_ratio of each edge jump; if more than max_excluded_fraction "
                                             "of sizes break monotonicity, no levels are fitted"},
        "sweep_reach": {"largest_bytes": sizes[-1] if sizes else None, "reported_llc_bytes": llc or None,
                        "ratio": reach, "label": "derived",
                        "warning": None if reach is None or reach >= 2 else
                        "sweep ended below 2x the reported last-level cache (RAM-limited), so DRAM may never be "
                        "reached: a final level here may not be DRAM, and a missing one may just be out of range"},
        "bandwidth": bw,
        "compute": fl,
        "timer": timer,
        "derived": {
            "ridge_flops_per_byte_f64": {"value": peak_f64 / dram_bw, "label": "derived",
                                         "from": [f"compute.vec_f64_{ncpu}t", f"bandwidth.dram_{ncpu}t"],
                                         "meaning": "arithmetic intensity above which all-core f64 work is compute-bound (roofline)"},
            "far_over_near_latency": ratio,
        },
        "memory_levels_method": level_note,
        "consistency_problems": check_raw(raw, settings),
        "caveats": [
            "Tier 2 only unless a tier-1 meter is listed as available: these are times, not energy.",
            "Latency includes TLB effects at large sizes; huge pages were requested (madvise) and THP mode is recorded.",
            "In a VM, reported caches and core counts are the hypervisor's claim; measured levels may differ.",
            "memory_levels are a fit to the measured curve (derived), not direct measurements; capacities are brackets.",
            "Noise is within this run on this machine; it says nothing about other runs or other hosts.",
            "Vector ceilings are for the instruction set recorded in build.vector_isa; elsewhere they mean that ISA.",
            "--reanalyse checks consistency, never authenticity: anyone can fabricate a self-consistent raw section "
            "with any numbers. Only an outside-anchored record (open item 3) can stop that.",
        ],
        "raw": raw,
    }
    return profile


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=None, help="repetitions per measurement (default 7, quick 3)")
    ap.add_argument("--min-ms", type=float, default=None, help="minimum timed window per measurement (default 30, quick 10)")
    ap.add_argument("--max-bytes", type=int, default=None, help="largest working set for the latency sweep")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed (recorded)")
    ap.add_argument("--calibration", default=None, help="tier-3 calibration file, if any")
    ap.add_argument("--out", default="profile.json")
    ap.add_argument("--reanalyse", metavar="PROFILE", help="recompute all statistics from a profile's raw reps; "
                    "exit 1 if stored numbers differ or raw is internally inconsistent; exit 2 if consistent but "
                    "run on another host, so meter claims are unverified (consistency, never authenticity)")
    a = ap.parse_args(argv)
    if a.reanalyse:
        return reanalyse(a.reanalyse, a.out if a.out != "profile.json" else None)
    reps = a.reps or (3 if a.quick else 7)
    min_ms = a.min_ms or (10.0 if a.quick else 30.0)
    seed = a.seed if a.seed is not None else int(time.time())

    t_start = time.time()
    env, caches = environment(), reported_caches()
    tier_list = tiers(calibration=a.calibration)
    with tempfile.TemporaryDirectory() as td:
        exe, build_info = build(td)
        meas, plan_info = plan(env, caches, a.quick, min_ms, a.max_bytes)
        rnd = random.Random(seed)
        raw = {name: [] for name, _ in meas}
        steal = []
        for r in range(reps):
            t0, s0 = cpu_ticks()
            order = meas[:]
            rnd.shuffle(order)  # interleave: drift spreads over every measurement, not one
            for name, args in order:
                if args[0] == "chase":  # a fresh random permutation each rep, so its effect shows up as noise
                    args = [*args, str(chase_seed(seed, r, int(args[1])))]
                rc, out, err = run([exe, *args], timeout=600)
                if rc != 0:
                    sys.exit(f"{name} failed: {err}")
                rec = json.loads(out)
                raw[name].append(rec)
            t1, s1 = cpu_ticks()
            steal.append(round((s1 - s0) / (t1 - t0), 4) if t0 is not None and t1 > t0 else None)
            print(f"rep {r + 1}/{reps} done ({time.time() - t_start:.0f}s, steal {steal[-1]})", file=sys.stderr)

    profile = analyse(raw, env, caches, tier_list, build_info,
                      {"reps": reps, "min_ms": min_ms, "seed": seed, "quick": a.quick,
                       "order": "shuffled per rep, all measurements interleaved",
                       "cpu_steal_fraction_per_rep": steal, "calibration": a.calibration, **plan_info},
                      round(time.time() - t_start, 1))
    add_digests(profile)
    with open(a.out, "w") as f:
        json.dump(profile, f, indent=1, allow_nan=False)
    print(summary_text(profile))
    print(f"\nprofile digest (anchor this): {profile['digests']['profile_sha256']}")
    print(f"\nwrote {a.out}", file=sys.stderr)


def reanalyse(path, out=None):
    with open(path) as f:
        old = json.load(f)
    new = analyse(old["raw"], old["environment"], old["reported_caches"], old["meter_tiers"],
                  old["build"], old["settings"], old["elapsed_s"])
    new["created_utc"] = old["created_utc"]
    add_digests(new)
    same = json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)
    print(f"profile digest recomputed: {new['digests']['profile_sha256']} "
          f"({'matches' if new['digests'] == old.get('digests') else 'DOES NOT match'} the stored one)", file=sys.stderr)
    print(summary_text(new))
    if out:
        with open(out, "w") as f:
            json.dump(new, f, indent=1)
        print(f"\nwrote {out}", file=sys.stderr)
    probs = list(new["consistency_problems"])
    # Meter claims are facts about a machine, not about the raw reps, so they can only be
    # re-checked on the same machine; elsewhere we say they were not re-checked.
    here = environment()["hostname_hash"] == old["environment"]["hostname_hash"]
    if here:
        now = {t["name"]: t["available"] for t in tiers(calibration=old["settings"].get("calibration"))}
        for t in old["meter_tiers"]:
            if t["name"] in now and now[t["name"]] != t["available"]:
                probs.append(f"meter {t['name']}: profile says available={t['available']}, re-detected {now[t['name']]}")
    print(f"meter tiers {'re-checked on this host' if here else 'NOT re-checked (different host)'}", file=sys.stderr)
    for p in probs:
        print(f"inconsistent raw: {p}", file=sys.stderr)
    print(f"\nstored numbers match recompute from raw: {same}; raw internally consistent: {not probs}"
          "\n(consistency only: this cannot show the raw data is genuine; that needs the anchored record)",
          file=sys.stderr)
    if not (same and not probs):
        return 1
    if not here:
        print("exit 2: consistent, but meter claims could not be re-checked off-host", file=sys.stderr)
        return 2
    return 0


def fmt_bytes(b):
    for u, s in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if b >= s:
            return f"{b / s:.3g} {u}"
    return f"{b} B"


def noise(v):
    return f" ± {v['se']:.2f} SE (CV {v['cv']:.1%})" if v["se"] is not None else " (1 rep, no noise estimate)"


def summary_text(p):
    L = [f"machine: {p['environment']['cpu_model']} x{p['environment']['logical_cpus']}"
         f"{' (VM)' if p['environment']['hypervisor'] else ''}",
         "meter tiers: " + ", ".join(f"{t['name']}={'yes' if t['available'] else 'no'}" for t in p["meter_tiers"]),
         "memory levels (derived from the measured latency curve):"]
    if p["memory_levels"] is None:
        L.append("  not computed: " + p["memory_levels_method"])
    for l in p["memory_levels"] or []:
        rng = f"{fmt_bytes(l['from_bytes'])}-{fmt_bytes(l['to_bytes'])}"
        if l["kind"] == "transition":
            lo, hi = l["latency_ns_range"]
            L.append(f"  transition: {lo:.1f}-{hi:.1f} ns over {rng}")
            continue
        cap = ""
        if "capacity_bytes_between" in l:
            lo, hi = l["capacity_bytes_between"]
            cap = (f", capacity between {fmt_bytes(lo)} and {fmt_bytes(hi)}" if hi is not None
                   else f", capacity above {fmt_bytes(lo)} (next level not found: upper end unknown)")
        L.append(f"  level {l['level']}: {l['latency_ns']:.2f} ± {l['latency_ns_se']:.2f} ns over {rng} "
                 f"(rises x{l['rise_across_span']:.2f} across it){cap}")
    if p.get("memory_levels_excluded_bytes"):
        L.append("  excluded as interference (slower than a larger size): "
                 + ", ".join(fmt_bytes(b) for b in p["memory_levels_excluded_bytes"]))
    if p.get("sweep_reach", {}).get("warning"):
        L.append("  WARNING: " + p["sweep_reach"]["warning"])
    L.append("reported caches: " + ", ".join(f"L{c['level']}{c['type'][0]} {fmt_bytes(c['size_bytes'])}" for c in p["reported_caches"]))
    for k, v in p["bandwidth"].items():
        L.append(f"bandwidth {k}: {v['median']:.2f} GB/s{noise(v)}")
    for k, v in p["compute"].items():
        L.append(f"compute {k}: {v['median']:.1f} GFLOP/s{noise(v)}")
    d = p["derived"]
    L.append(f"vector ISA of this build: {p['build'].get('vector_isa', 'unknown')}")
    st = [x for x in p["settings"].get("cpu_steal_fraction_per_rep", []) if x is not None]
    if st:
        L.append(f"cpu steal during run (host interference, measured): max {max(st):.1%} of CPU time in a rep")
    if p["consistency_problems"]:
        L.append(f"CONSISTENCY PROBLEMS: {len(p['consistency_problems'])}")
    L.append(f"ridge point (f64, all cores): {d['ridge_flops_per_byte_f64']['value']:.1f} flop/byte (derived)")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
