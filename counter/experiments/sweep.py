import json, subprocess, os, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
p=json.load(open(sys.argv[1])); cfgname=sys.argv[2]; seed=int(sys.argv[3]) if len(sys.argv)>3 else 7
cfgs={"host":[], "llL2":["--D1=32768,8,64","--I1=32768,8,64","--LL=1048576,16,64"], "reference":["--I1=32768,8,64","--D1=49152,12,64","--LL=8388608,16,64"], "probeL1":["--D1=49152,12,64","--I1=32768,8,64","--LL=1048576,16,64"]}
sizes=[e['bytes'] for e in p['latency_curve']]
def one(b):
    n=b//64; laps=max(2, (1<<19)//n)
    a=run1(b,laps); c=run1(b,2*laps)
    d={k:c[k]-a[k] for k in a}
    return b, d['Dr'], d['D1mr']/d['Dr'], d['DLmr']/d['Dr']
def run1(b,laps):
    with tempfile.TemporaryDirectory() as td:
        o=os.path.join(td,"o")
        subprocess.run(["valgrind","--command-line-only=yes","--tool=cachegrind","--cache-sim=yes","--instr-at-start=no",f"--cachegrind-out-file={o}",*cfgs[cfgname],"./chase_count",str(b),str(laps),str(seed)],capture_output=True,check=True)
        for l in open(o):
            if l.startswith("events:"):ev=l.split()[1:]
            if l.startswith("summary:"):sm=dict(zip(ev,map(int,l.split()[1:])))
    return sm
with ThreadPoolExecutor(2) as ex: res=list(ex.map(one,sizes))
lat={e['bytes']:e['ns_per_load']['median'] for e in p['latency_curve']}
out=[dict(bytes=b,loads=dr,d1_miss=d1,ll_miss=ll,ns=lat[b]) for b,dr,d1,ll in res]
json.dump(out,open(f"sweep_{cfgname}_{seed}.json","w"))
for r in out: print(f"{r['bytes']:>11} {r['ns']:8.2f}ns  D1miss {r['d1_miss']:.3f}  LLmiss {r['ll_miss']:.3f}")
