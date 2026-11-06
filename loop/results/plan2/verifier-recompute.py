import hashlib, json, math, statistics, subprocess, sys
R="/home/claude/exagium/loop/"
rows=[[int(t) for t in l.split(",")] for l in open(R+"task/optdigits.tes")]
y=[r[64] for r in rows]; open("x.bin","wb").write(bytes(v for r in rows for v in r[:64]))
D={"linear":"baselines/linear","thermo":"baselines/thermo","mlp16":"candidates/mlp16","mlp32":"candidates/mlp32","mlp64":"candidates/mlp64"}
P={}
for k,d in D.items():
    subprocess.run(["gcc","-O2","-std=c11","-ffp-contract=off","-o",k,R+d+"/infer.c","-lm"],check=True)
    o=subprocess.run(["./"+k,R+d+"/weights.bin","x.bin",str(len(rows))],capture_output=True,check=True).stdout
    P[k]=[c-48 for c in o[:-1]]
    print(k, sum(p==t for p,t in zip(P[k],y)), hashlib.sha256(o[:-1]).hexdigest()[:16])
for a in ("mlp16","mlp32","mlp64"):
    for b in ("linear","thermo"):
        d=[(pa==t)-(pb==t) for pa,pb,t in zip(P[a],P[b],y)]
        w=sum(x==1 for x in d); l=sum(x==-1 for x in d)
        print(a,b,"wins",w,"losses",l,"d",statistics.fmean(d),"se",statistics.stdev(d)/math.sqrt(len(d)))
