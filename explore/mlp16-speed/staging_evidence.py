#!/usr/bin/env python3
"""Raw evidence for reports/staging-2026-11-17.md: clock readings, page faults per run, output identity at many N,
and paired differences (with the smallest difference each comparison could detect). Writes results/staging-evidence.json."""
import json, os, resource, statistics, subprocess, hashlib
cpu = str(max(os.sched_getaffinity(0))); pin = ['taskset', '-c', cpu]
ev = {'clock_ghz': [subprocess.run(pin + ['bin/clock', '100'], capture_output=True, text=True).stdout.strip() for _ in range(3)]}
W = '../../loop/candidates/mlp16/weights.bin'; L = 'weights/lgn_d.bin'
stage = json.load(open('programs_stage.json')); full = json.load(open('programs_staging_full.json'))
def faults(b, w):
    mf = []
    for _ in range(5):
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_minflt
        subprocess.run(pin + [b, w, 'data/big.bin', '95600'], capture_output=True)
        mf.append(resource.getrusage(resource.RUSAGE_CHILDREN).ru_minflt - r0)
    return statistics.median(mf)
ev['page_faults_per_run_N95600'] = {k: faults(*v) for k, v in {**stage, **full}.items()}
Ns = [0, 1, 63, 64, 65, 956, 1023, 1024, 1025, 2047, 2049, 50000, 95600]
def md5(b, w, n): return hashlib.md5(subprocess.run([b, w, 'data/big.bin', str(n)], capture_output=True).stdout).hexdigest()
ident = {}
for group, (ref, others) in {'stage': ('st_bigbuf', [k for k in stage if k != 'st_bigbuf']),
                             'mlp16': ('mlp16', ['mlp16_chunked', 'mlp16_mmap']), 'lgn_d': ('lgn_d', ['lgn_d_blockread'])}.items():
    spec = {**stage, **full}
    ident[group] = {o: all(md5(*spec[o], n) == md5(*spec[ref], n) for n in Ns) for o in others}
ev['output_identical_vs_original_at_N'] = {'N': Ns, 'result': ident}
def paired(f, a, b):
    d = json.load(open(f)); raw = d['raw_ns']; N = d['meta']['n']
    per = lambda k: [(x - z) / N for x, z in zip(raw[k]['full'], raw[k]['zero'])]
    x = [p - q for p, q in zip(per(a), per(b))]; q = statistics.quantiles(x, n=4)
    return {'median': round(statistics.median(x), 2), 'iqr': [round(q[0], 2), round(q[2], 2)], 'positive_rounds': sum(v > 0 for v in x), 'rounds': len(x)}
ev['paired'] = {
 'staging: st_bigbuf - st_chunk1024': paired('results/staging-xeon.json', 'st_bigbuf', 'st_chunk1024'),
 'staging: st_mmap - st_chunk1024': paired('results/staging-xeon.json', 'st_mmap', 'st_chunk1024'),
 'staging: st_mmap_pop - st_chunk1024': paired('results/staging-xeon.json', 'st_mmap_pop', 'st_chunk1024'),
 'staging: st_read1024 - st_chunk1024': paired('results/staging-xeon.json', 'st_read1024', 'st_chunk1024'),
 'staging: st_read16 - st_chunk1024': paired('results/staging-xeon.json', 'st_read16', 'st_chunk1024'),
 'full: mlp16_mmap - mlp16_chunked': paired('results/staging-full-xeon.json', 'mlp16_mmap', 'mlp16_chunked'),
}
json.dump(ev, open('results/staging-evidence.json', 'w'), indent=1)
print(json.dumps({k: ev[k] for k in ('clock_ghz', 'output_identical_vs_original_at_N', 'paired')}, indent=1))
