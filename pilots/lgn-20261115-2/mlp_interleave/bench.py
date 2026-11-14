"""Steady-state per-image time: (t(N) - t(0)) / N, binaries pinned to one CPU, interleaved random order, median of rounds."""
import subprocess, time, random, statistics, sys, json, os
bins = sys.argv[1].split(','); weights = sys.argv[2].split(','); N = int(sys.argv[3]); R = int(sys.argv[4])
X = sys.argv[5]; cpu = str(max(os.sched_getaffinity(0)))
res = {b: {'full': [], 'zero': []} for b in bins}
ref = {}
rng = random.Random(1)
for r in range(R):
    jobs = [(b, w, k) for b, w in zip(bins, weights) for k in ('full', 'zero')]
    rng.shuffle(jobs)
    for b, w, k in jobs:
        n = N if k == 'full' else 0
        t0 = time.perf_counter_ns()
        o = subprocess.run(['taskset', '-c', cpu, b, w, X, str(n)], capture_output=True).stdout
        t = time.perf_counter_ns() - t0
        res[b][k].append(t)
        if k == 'full':
            h = hash(o); assert ref.setdefault(b, h) == h
out = {}
for b in bins:
    diffs = [(f - z) / N for f, z in zip(res[b]['full'], res[b]['zero'])]
    out[os.path.basename(b)] = (statistics.median(diffs), statistics.quantiles(diffs, n=4))
for k, (m, q) in out.items():
    print(f"{k:14s} median {m:7.2f} ns/img   IQR {q[0]:7.2f}..{q[2]:7.2f}")
