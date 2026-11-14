"""Per-image time at several run sizes: (t(N) - t(0)) / N per round, interleaved, pinned; median and IQR."""
import subprocess, time, random, statistics, sys, os, json
spec = json.load(open(sys.argv[1])); R = int(sys.argv[2])
sizes = [(956, '/home/claude/pilot/dev.bin'), (9560, 'mid.bin'), (95600, 'big.bin')]
cpu = str(max(os.sched_getaffinity(0)))
rng = random.Random(3)
t = {(n, s): [] for n in spec for s in [0] + [k for k, _ in sizes]}
outs = {}
for r in range(R):
    jobs = [(name, k, x) for name in spec for k, x in [(0, 'big.bin')] + sizes]
    rng.shuffle(jobs)
    for name, k, x in jobs:
        b, w = spec[name]
        t0 = time.perf_counter_ns()
        o = subprocess.run(['taskset', '-c', cpu, b, w, x, str(k)], capture_output=True).stdout
        t[(name, k)].append(time.perf_counter_ns() - t0)
        if k == 956: assert outs.setdefault(name, o) == o
res = {}
for name in spec:
    row = {}
    for k, _ in sizes:
        d = [(a - z) / k for a, z in zip(t[(name, k)], t[(name, 0)])]
        q = statistics.quantiles(d, n=4)
        row[k] = (round(statistics.median(d), 1), round(q[0], 1), round(q[2], 1))
    row['startup_us'] = round(statistics.median(t[(name, 0)]) / 1000, 1)
    res[name] = row
    print(f"{name:12s} startup {row['startup_us']:7.1f} us | " + " | ".join(f"N={k}: {row[k][0]:6.1f} ({row[k][1]:.0f}-{row[k][2]:.0f})" for k, _ in sizes))
json.dump(res, open(sys.argv[3], 'w'), indent=1)
