#!/usr/bin/env python3
"""Steady-state cost per image, without hardware counters.

For every program in a JSON spec {"name": [binary, weights]}, runs `binary weights data N` at N = 0 and N = 95,600
(data/big.bin), pinned to one CPU when `taskset` exists, all programs and both sizes in a fresh random order each
round. Per round: ns/img = (t(N) - t(0)) / N. Reports median and IQR over rounds, plus (if valgrind exists)
simulated instructions per image at the same N and instructions per ns. Checks every program prints the same output
every round. Writes the raw per-round times to results/<tag>.json so others can recompute.

Usage: python3 bench.py programs.json --rounds 21 --tag myhost
"""
import argparse, json, os, platform, random, re, shutil, statistics, subprocess, time

ap = argparse.ArgumentParser()
ap.add_argument('spec'); ap.add_argument('--rounds', type=int, default=21); ap.add_argument('--tag', default='run')
ap.add_argument('--n', type=int, default=95600); ap.add_argument('--data', default='data/big.bin')
ap.add_argument('--no-valgrind', action='store_true')
a = ap.parse_args()
spec = json.load(open(a.spec))
pin = ['taskset', '-c', str(max(os.sched_getaffinity(0)))] if shutil.which('taskset') else []
rng = random.Random(1)
raw = {k: {'full': [], 'zero': []} for k in spec}
outs = {}
for r in range(a.rounds):
    jobs = [(k, n) for k in spec for n in (0, a.n)]
    rng.shuffle(jobs)
    for k, n in jobs:
        b, w = spec[k]
        t0 = time.perf_counter_ns()
        p = subprocess.run(pin + [b, w, a.data, str(n)], capture_output=True)
        t = time.perf_counter_ns() - t0
        if p.returncode != 0:
            raise SystemExit(f'{k} exited {p.returncode}')
        raw[k]['full' if n else 'zero'].append(t)
        if n and outs.setdefault(k, p.stdout) != p.stdout:
            raise SystemExit(f'{k}: output changed between rounds')
res = {}
for k in spec:
    d = [(f - z) / a.n for f, z in zip(raw[k]['full'], raw[k]['zero'])]
    q = statistics.quantiles(d, n=4)
    res[k] = {'ns_per_img_median': round(statistics.median(d), 2), 'iqr': [round(q[0], 2), round(q[2], 2)]}
    if not a.no_valgrind and shutil.which('valgrind'):
        def ir(n):
            e = subprocess.run(['valgrind', '--tool=cachegrind', '--cache-sim=no', '--cachegrind-out-file=/dev/null',
                                spec[k][0], spec[k][1], a.data, str(n)], capture_output=True, text=True).stderr
            return int(re.search(r'I\s+refs:\s+([\d,]+)', e).group(1).replace(',', ''))
        i = (ir(a.n) - ir(0)) / a.n
        res[k]['ir_per_img'] = round(i, 1)
        res[k]['ir_per_ns'] = round(i / res[k]['ns_per_img_median'], 2)
same = {}
for k, o in outs.items():
    same.setdefault(o, []).append(k)
cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor())
meta = {'tag': a.tag, 'cpu': cpu, 'machine': platform.machine(), 'kernel': platform.release(), 'rounds': a.rounds,
        'n': a.n, 'pinned': bool(pin), 'identical_output_groups': list(same.values())}
os.makedirs('results', exist_ok=True)
json.dump({'meta': meta, 'results': res, 'raw_ns': raw}, open(f'results/{a.tag}.json', 'w'), indent=1)
print(f"{cpu} ({platform.machine()}), {a.rounds} rounds, N={a.n}, pinned={bool(pin)}")
for k, v in res.items():
    extra = f"  Ir/img {v['ir_per_img']:8.1f}  Ir/ns {v['ir_per_ns']:5.2f}" if 'ir_per_img' in v else ''
    print(f"{k:16s} {v['ns_per_img_median']:8.2f} ns/img  IQR {v['iqr'][0]:.1f}-{v['iqr'][1]:.1f}{extra}")
print('programs with identical output:', meta['identical_output_groups'])
print(f'raw times: results/{a.tag}.json')
