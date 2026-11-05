# run a command under cachegrind N times; print summary counts per rep
import subprocess, sys, json, os, tempfile
def run(cmd, extra=(), env=None, pre=()):
    with tempfile.TemporaryDirectory() as td:
        out=os.path.join(td,"cg.out")
        p=subprocess.run([*pre,"valgrind","--tool=cachegrind","--cache-sim=yes",f"--cachegrind-out-file={out}",*extra,*cmd],capture_output=True,text=True,env=env)
        ev=sm=None
        for line in open(out):
            if line.startswith("events:"): ev=line.split()[1:]
            if line.startswith("summary:"): sm=[int(x) for x in line.split()[1:]]
        return dict(zip(ev,sm)), p.stderr
if __name__=="__main__":
    n=int(sys.argv[1]); mode=sys.argv[2]; cmd=sys.argv[3:]
    extra=[]; pre=[]; env=dict(os.environ)
    if "pin" in mode: extra=["--I1=32768,8,64","--D1=32768,8,64","--LL=8388608,16,64"]
    if "noaslr" in mode: pre=["setarch","-R"]
    if "hashseed" in mode: env["PYTHONHASHSEED"]="0"
    if "1thr" in mode: env["OPENBLAS_NUM_THREADS"]="1"; env["OMP_NUM_THREADS"]="1"
    rows=[run(cmd,extra,env,pre)[0] for _ in range(n)]
    keys=rows[0].keys()
    for k in keys:
        v=[r[k] for r in rows]; print(f"{k:5s} min={min(v):>14,} max={max(v):>14,} spread={(max(v)-min(v))/max(1,min(v)):.2e}")
