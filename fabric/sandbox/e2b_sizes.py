"""Sized E2B templates for the box connector, and what each size really gives (owner, 2026-11-26: "match the power
to what it's doing"). E2B fixes memory per template; since SDK 2.64 a template built with max_cpu_count is resizable:
each sandbox picks 1..max CPUs at create (Sandbox.create(cpu_count=...)), not while running (checked in the SDK
source, e2b 2.64.0 sandbox_sync/main.py). The docs disagree on whether custom sizes need Pro; a build on this
account settles it, and a refusal is recorded as the result.

  python3 e2b_sizes.py OUT.json        build the templates, then create one sandbox per size, measure, kill it

Each sandbox runs: nproc, memory, a single-thread benchmark and an all-cores benchmark (the same work split across
nproc processes), so the record shows whether more CPUs really make work faster. Kind: measured."""
import json, os, sys, time
from datetime import datetime, timezone
from e2b import Sandbox, Template

RUN = f"{os.environ.get('GITHUB_WORKFLOW', 'local')}#{os.environ.get('GITHUB_RUN_ID', '-')}"
WHO = "fabric/sandbox/e2b_sizes.py (written by Claude Opus 5.5), run " + RUN
# name -> (memory MB, start CPUs, max CPUs); listed tiers on e2b.dev/pricing (read 2026-11-26).
# Run 45793709339: "400: Resizable templates are not enabled for your team", so each size is its own fixed template
# (max = start). A refusal of these too would mean custom sizes are Pro-only on this account.
TEMPLATES = {"forge-box-s": (2048, 1, 1), "forge-box": (4096, 2, 2), "forge-box-l": (4096, 4, 4),
             "forge-box-xl": (8192, 8, 8)}
PROBES = [(n, c) for n, (_, c, _) in TEMPLATES.items()]
# per-second prices, e2b.dev/pricing read 2026-11-26: $0.000014 per vCPU, $0.0000045 per GiB
PRICE_CPU, PRICE_GIB = 0.000014, 0.0000045
WORK = 24_000_000
BENCH = ("import time,os,multiprocessing as m\n"
         "def f(n): return sum(i*i for i in range(n))\n"
         "if __name__=='__main__':\n"
         f" W={WORK}; t=time.time(); f(W//8); one=time.time()-t\n"
         " p=os.cpu_count(); t=time.time()\n"
         " with m.Pool(p) as pool: pool.map(f,[W//p]*p)\n"
         " print(round(one,3), round(time.time()-t,3))\n")


def now():
    return datetime.now(timezone.utc).isoformat()[:19] + "Z"


def sh(sb, cmd, timeout=180):
    try:
        r = sb.commands.run(cmd, timeout=timeout)
        return (r.stdout or "").strip()[-300:]
    except Exception as e:
        return "error: " + str(e)[-200:]


def build():
    out = {}
    image = Template().from_base_image().apt_install(["bubblewrap", "procps"])
    for name, (mem, cpu, cpu_max) in TEMPLATES.items():
        t = time.time()
        try:
            extra = {"max_cpu_count": cpu_max} if cpu_max != cpu else {}
            Template.build(image, name, cpu_count=cpu, memory_mb=mem, **extra)
            out[name] = {"ok": True, "build_s": round(time.time() - t, 1), "memory_mb": mem, "cpu": [cpu, cpu_max]}
        except Exception as e:   # e.g. a plan refusal: that is the answer to the Hobby question
            out[name] = {"ok": False, "error": str(e)[-400:]}
        print(name, out[name], flush=True)
    return out


def probe(name, cpus):
    rec = {"template": name, "cpu_asked": cpus, "at": now()}
    t = time.time()
    try:
        sb = Sandbox.create(template=name, timeout=300, metadata={"forge": "sizes"})   # fixed size: no cpu_count
    except Exception as e:
        return rec | {"ok": False, "error": str(e)[-400:]}
    try:
        rec["create_s"] = round(time.time() - t, 2)
        rec["nproc"] = sh(sb, "nproc")
        rec["mem_mb"] = sh(sb, "free -m | awk 'NR==2{print $2}'")
        sh(sb, f"cat > /tmp/b.py <<'EOF'\n{BENCH}EOF")
        runs = [sh(sb, "python3 /tmp/b.py").split() for _ in range(3)]
        rec["bench_one_all_s"] = runs
        ok = [r for r in runs if len(r) == 2]
        if ok:
            all_s = sorted(float(r[1]) for r in ok)[len(ok) // 2]
            rec["all_cores_median_s"] = all_s
            gib = TEMPLATES[name][0] / 1024
            rec["usd_per_hour"] = round((cpus * PRICE_CPU + gib * PRICE_GIB) * 3600, 4)
            rec["usd_for_the_work"] = round(all_s * (cpus * PRICE_CPU + gib * PRICE_GIB), 7)
        rec["ok"] = rec["nproc"] == str(cpus)
    finally:
        sb.kill()
    return rec


if __name__ == "__main__":
    res = {"recorded_by": WHO, "kind": "measured", "at": now(), "templates": build(), "probes": []}
    for name, cpus in PROBES:
        if res["templates"].get(name, {}).get("ok"):
            p = probe(name, cpus); res["probes"].append(p); print(json.dumps(p), flush=True)
    os.makedirs(os.path.dirname(sys.argv[1]) or ".", exist_ok=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
