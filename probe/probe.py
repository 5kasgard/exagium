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
def summarise(xs):
    """Median, mean, sample SD, standard error of the mean, CV. n>=1."""
    n = len(xs)
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else None  # undefined for one rep: null, never NaN
    se = sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "median": statistics.median(xs), "mean": mean, "sd": sd, "se": se,
            "cv": sd / mean if n > 1 and mean else None, "min": min(xs), "max": max(xs)}


# ---------------------------------------------------------------- level detection
def segment_plateaus(sizes, lat, se=None):
    """Piecewise-constant fit of log(latency) vs size, segment count chosen by BIC.

    Each point is weighted by its own measured noise (1/var of log latency, from the
    repeats), so a split must beat this run's noise, not just differ. Nothing about
    cache sizes is assumed; the only choice is the standard BIC penalty.
    Returns segments as (start_index, end_index_inclusive)."""
    y = [math.log(v) for v in lat]
    n = len(y)
    rel = [s / v if s is not None and s > 0 else None for s, v in zip(se or [None] * n, lat)]
    known = sorted(r for r in rel if r)
    fill = known[len(known) // 2] if known else 1.0  # points with no noise estimate get the typical one
    w = [1.0 / (r or fill) ** 2 for r in rel]
    pw, pwy, pwy2 = [0.0], [0.0], [0.0]
    for wi, v in zip(w, y):
        pw.append(pw[-1] + wi); pwy.append(pwy[-1] + wi * v); pwy2.append(pwy2[-1] + wi * v * v)

    def sse(i, j):  # weighted SSE (chi-square) of points i..j around their weighted mean
        W = pw[j + 1] - pw[i]; s = pwy[j + 1] - pwy[i]
        return max((pwy2[j + 1] - pwy2[i]) - s * s / W, 0.0)

    kmax = max(1, min(n // 2, 16))
    INF = float("inf")
    # dp[k][j] = best SSE for first j+1 points with k segments
    dp = [[INF] * n for _ in range(kmax + 1)]
    back = [[-1] * n for _ in range(kmax + 1)]
    for j in range(n):
        dp[1][j] = sse(0, j)
    for k in range(2, kmax + 1):
        for j in range(k - 1, n):
            for i in range(k - 1, j + 1):
                c = dp[k - 1][i - 1] + sse(i, j)
                if c < dp[k][j]:
                    dp[k][j], back[k][j] = c, i
    best_k, best_bic = 1, INF
    for k in range(1, kmax + 1):
        # known-variance BIC: chi-square + (k means + k-1 breakpoints) * ln n
        bic = dp[k][n - 1] + (2 * k - 1) * math.log(n)
        if bic < best_bic:
            best_k, best_bic = k, bic
    segs, j, k = [], n - 1, best_k
    while k >= 1:
        i = back[k][j] if k > 1 else 0
        segs.append((i, j)); j, k = i - 1, k - 1
    return segs[::-1]


def describe_levels(sizes, med, se=None):
    """Plateaus (>=2 points at one latency) and ramps (consecutive one-point segments,
    i.e. latency still climbing). A ramp is not a cache level and is not reported as one."""
    segs = segment_plateaus(sizes, med, se)
    groups = []  # merge runs of single-point segments into one ramp
    for i, j in segs:
        single = (i == j)
        if single and groups and groups[-1][2] == "ramp":
            groups[-1][1] = j
        else:
            groups.append([i, j, "ramp" if single else "plateau"])
    # a lone single point at either end of the sweep is an edge plateau, not a ramp
    for g in (groups[0], groups[-1]):
        if g[2] == "ramp" and g[0] == g[1]:
            g[2] = "plateau"
    out, k = [], 0
    for idx, (i, j, kind) in enumerate(groups):
        pts = med[i:j + 1]
        lvl = {"kind": kind, "points": j - i + 1, "from_bytes": sizes[i], "to_bytes": sizes[j],
               "latency_ns_median": statistics.median(pts), "latency_ns_range": [min(pts), max(pts)],
               "label": "measured"}
        if kind == "plateau":
            k += 1
            lvl["plateau"] = k
            if idx + 1 < len(groups):
                # capacity lies between the last size on this plateau and the first after it
                lvl["capacity_bytes_between"] = [sizes[j], sizes[j + 1]]
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
    exe = shutil.which("perf")
    if not exe:
        return {"available": False, "evidence": f"perf not installed; perf_event_paranoid={para}"}
    rc, out, err = run([exe, "stat", "-x,", "-e", "instructions,cycles", "true"], timeout=20)
    txt = out + err
    if rc != 0 or "not supported" in txt or "<not counted>" in txt or "Permission" in txt:
        return {"available": False, "evidence": f"perf present but counters unusable (paranoid={para}): {txt.strip()[:200]}"}
    return {"available": True, "evidence": f"perf stat counted instructions,cycles (paranoid={para})"}


def tier_nvml():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "evidence": "nvidia-smi not found"}
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
        {"tier": 1, "name": "nvml_gpu_power", "kind": "measured_energy", **tier_nvml()},
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
    return exe, {"compiler": ver, "flags": flags, "kernels_sha256": sha256_file(src),
                 "probe_sha256": sha256_file(__file__)}


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


# ---------------------------------------------------------------- analysis
def analyse(raw, env, caches, tier_list, build_info, settings, elapsed):
    """Everything in the profile except raw/env/tiers is computed here from raw reps,
    so `--reanalyse` can regenerate (and check) any stored profile."""
    def stat(name, key):
        return summarise([x[key] for x in raw[name]])

    sizes = sorted(int(n.split(":")[1]) for n in raw if n.startswith("chase:"))
    lat = [stat(f"chase:{s}", "ns_per_load") for s in sizes]
    meds = [x["median"] for x in lat]
    levels = describe_levels(sizes, meds, [x["se"] for x in lat])

    ncpu = env["logical_cpus"] or 1
    thr = sorted({1, ncpu})
    bw = {f"{k}_{th}t": {**stat(f"stream:{k}:{th}", "GBps"), "unit": "GB/s", "label": "measured",
                         "bytes": raw[f"stream:{k}:{th}"][0]["bytes"]}
          for k in ("dram", "l1") for th in thr}
    fl = {f"{op}_{th}t": {**stat(f"flops:{op}:{th}", "GFLOPs"), "unit": "GFLOP/s", "label": "measured"}
          for op in ("scalar_f64", "vec_f64", "vec_f32") for th in thr}
    timer = {"resolution_ns": stat("timer", "resolution_ns"), "read_overhead_ns": stat("timer", "read_overhead_ns"),
             "label": "measured"}
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
        "memory_levels": levels,
        "bandwidth": bw,
        "compute": fl,
        "timer": timer,
        "derived": {
            "ridge_flops_per_byte_f64": {"value": peak_f64 / dram_bw, "label": "derived",
                                         "from": [f"compute.vec_f64_{ncpu}t", f"bandwidth.dram_{ncpu}t"],
                                         "meaning": "arithmetic intensity above which all-core f64 work is compute-bound (roofline)"},
            "dram_latency_over_l1": {"value": levels[-1]["latency_ns_median"] / levels[0]["latency_ns_median"],
                                     "label": "derived", "meaning": "cost of a far load vs a near one"},
        },
        "caveats": [
            "Tier 2 only unless a tier-1 meter is listed as available: these are times, not energy.",
            "Latency includes TLB effects at large sizes; huge pages were requested (madvise) and THP mode is recorded.",
            "In a VM, reported caches and core counts are the hypervisor's claim; measured plateaus may differ.",
            "Noise is within this run on this machine; it says nothing about other runs or other hosts.",
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
                    "exit 1 if the stored numbers differ")
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
        for r in range(reps):
            order = meas[:]
            rnd.shuffle(order)  # interleave: drift spreads over every measurement, not one
            for name, args in order:
                rc, out, err = run([exe, *args], timeout=600)
                if rc != 0:
                    sys.exit(f"{name} failed: {err}")
                rec = json.loads(out)
                raw[name].append(rec)
            print(f"rep {r + 1}/{reps} done ({time.time() - t_start:.0f}s)", file=sys.stderr)

    profile = analyse(raw, env, caches, tier_list, build_info,
                      {"reps": reps, "min_ms": min_ms, "seed": seed, "quick": a.quick,
                       "order": "shuffled per rep, all measurements interleaved", **plan_info},
                      round(time.time() - t_start, 1))
    with open(a.out, "w") as f:
        json.dump(profile, f, indent=1, allow_nan=False)
    print(summary_text(profile))
    print(f"\nwrote {a.out}", file=sys.stderr)


def reanalyse(path, out=None):
    with open(path) as f:
        old = json.load(f)
    new = analyse(old["raw"], old["environment"], old["reported_caches"], old["meter_tiers"],
                  old["build"], old["settings"], old["elapsed_s"])
    new["created_utc"] = old["created_utc"]
    same = json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)
    print(summary_text(new))
    if out:
        with open(out, "w") as f:
            json.dump(new, f, indent=1)
        print(f"\nwrote {out}", file=sys.stderr)
    print(f"\nrecompute matches stored profile: {same}", file=sys.stderr)
    return 0 if same else 1


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
         "memory plateaus (measured):"]
    for l in p["memory_levels"]:
        rng = f"{fmt_bytes(l['from_bytes'])}-{fmt_bytes(l['to_bytes'])}"
        if l["kind"] == "ramp":
            lo, hi = l["latency_ns_range"]
            L.append(f"  ramp: {lo:.1f}->{hi:.1f} ns over {rng}")
            continue
        cap = f", capacity between {fmt_bytes(l['capacity_bytes_between'][0])} and {fmt_bytes(l['capacity_bytes_between'][1])}" \
            if "capacity_bytes_between" in l else ""
        L.append(f"  plateau {l['plateau']}: {l['latency_ns_median']:.1f} ns ({rng}){cap}")
    L.append("reported caches: " + ", ".join(f"L{c['level']}{c['type'][0]} {fmt_bytes(c['size_bytes'])}" for c in p["reported_caches"]))
    for k, v in p["bandwidth"].items():
        L.append(f"bandwidth {k}: {v['median']:.2f} GB/s{noise(v)}")
    for k, v in p["compute"].items():
        L.append(f"compute {k}: {v['median']:.1f} GFLOP/s{noise(v)}")
    d = p["derived"]
    L.append(f"ridge point (f64, all cores): {d['ridge_flops_per_byte_f64']['value']:.1f} flop/byte (derived)")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
