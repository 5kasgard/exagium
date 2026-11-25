"""Measure an E2B sandbox against the Forge's sandbox criteria (audit/fabric/sandbox-decision-2026-11-25.md, sec. 5;
issue #191). Standard library + the e2b SDK. State lives in a JSON file (sandbox id, pid, cycles), not in any chat.

  python3 e2b_eval.py create STATE     # make one sandbox, run the jail and CPU checks, start a ticking process, pause
  python3 e2b_eval.py check STATE      # from a fresh client: resume by id, check files, tools, process, then pause
  python3 e2b_eval.py kill STATE       # end it

recorded_by is written into every record: the model and run that made it. Kind: measured."""
import json, os, sys, time
from datetime import datetime, timezone
from e2b import Sandbox

RUN = f"{os.environ.get('GITHUB_WORKFLOW', 'local')}#{os.environ.get('GITHUB_RUN_ID', '-')}"
WHO = "fabric/sandbox/e2b_eval.py (written by Claude Opus 5.5), run " + RUN
LANDLOCK = ("import ctypes,os;l=ctypes.CDLL(None,use_errno=True);"
            "v=l.syscall(444,None,0,1);print(v if v>0 else 'none errno %d'%ctypes.get_errno())")
BENCH = "python3 -c \"import time;t=time.time();s=sum(i*i for i in range(3_000_000));print(round(time.time()-t,3))\""


def now():
    return datetime.now(timezone.utc).isoformat()[:19] + "Z"


def sh(sb, cmd, timeout=180):
    try:
        r = sb.commands.run(cmd, timeout=timeout)
        return {"rc": r.exit_code, "out": (r.stdout or "").strip()[-400:], "err": (r.stderr or "").strip()[-200:]}
    except Exception as e:   # a non-zero exit raises in the SDK; keep the evidence
        return {"rc": getattr(e, "exit_code", -1), "out": str(getattr(e, "stdout", "") or "")[-400:], "err": str(e)[-300:]}


def create(path):
    t = time.time()
    sb = Sandbox.create(timeout=10 * 60, lifecycle={"on_timeout": "pause", "auto_resume": True},
                        metadata={"forge": "sandbox-eval", "issue": "191"})
    rec = {"recorded_by": WHO, "kind": "measured", "provider": "e2b", "sandbox_id": sb.sandbox_id,
           "created_at": now(), "create_s": round(time.time() - t, 2), "checks": {}}
    c = rec["checks"]
    c["uname"] = sh(sb, "uname -srm")["out"]
    c["nproc"] = sh(sb, "nproc")["out"]
    c["mem"] = sh(sb, "free -m | awk 'NR==2{print $2\" MB\"}'")["out"]
    c["landlock_abi"] = sh(sb, f'python3 -c "{LANDLOCK}"')["out"]
    c["apt_bwrap"] = sh(sb, "sudo apt-get update -qq >/dev/null 2>&1; sudo apt-get install -y -qq bubblewrap >/dev/null 2>&1; bwrap --version", 300)
    c["bwrap_proc"] = sh(sb, "bwrap --ro-bind / / --proc /proc --dev /dev --unshare-all true && echo ok")
    c["cpu_bench_s"] = [sh(sb, BENCH)["out"] for _ in range(5)]
    sh(sb, "echo forge-$(date +%s) > /home/user/marker.txt && pip install -q --user cowsay >/dev/null 2>&1; echo done")
    h = sb.commands.run("sh -c 'while true; do date +%s >> /home/user/ticks.log; sleep 10; done'", background=True, timeout=0)
    rec["pid"] = h.pid
    time.sleep(25)
    rec["ticks_before_pause"] = sh(sb, "wc -l < /home/user/ticks.log")["out"]
    t = time.time(); sb.pause(); rec["pause_s"] = round(time.time() - t, 2)
    rec["paused_at"] = now(); rec["cycles"] = []
    json.dump(rec, open(path, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("create_s", "pause_s")} | {"checks": c}, indent=1))


def check(path):
    rec = json.load(open(path))
    t = time.time()
    try:
        sb = Sandbox.connect(rec["sandbox_id"])
    except Exception as e:
        cyc = {"at": now(), "ok": False, "error": str(e)[:300]}
        rec["cycles"].append(cyc); json.dump(rec, open(path, "w"), indent=1); print(cyc); return
    resume_s = round(time.time() - t, 2)
    idle_h = round((datetime.now(timezone.utc) - datetime.fromisoformat(rec["paused_at"].replace("Z", "+00:00"))).total_seconds() / 3600, 2)
    time.sleep(15)   # long enough for the ticking process to write again if it survived
    cyc = {"at": now(), "recorded_by": WHO, "idle_h": idle_h, "resume_s": resume_s,
           "file": sh(sb, "cat /home/user/marker.txt")["out"],
           "tool": sh(sb, "python3 -c 'import cowsay; print(\"ok\")'")["out"],
           "process_alive": sh(sb, f"kill -0 {rec['pid']} && echo alive || echo gone")["out"],
           "last_ticks": sh(sb, "tail -n 3 /home/user/ticks.log; date +%s")["out"]}
    cyc["ok"] = cyc["file"].startswith("forge-") and cyc["tool"] == "ok" and cyc["process_alive"] == "alive"
    t = time.time(); sb.pause(); cyc["pause_s"] = round(time.time() - t, 2)
    rec["paused_at"] = now(); rec["cycles"].append(cyc)
    json.dump(rec, open(path, "w"), indent=1)
    print(json.dumps(cyc, indent=1))


def kill(path):
    rec = json.load(open(path))
    Sandbox.kill(rec["sandbox_id"])
    rec["killed_at"] = now(); json.dump(rec, open(path, "w"), indent=1)
    ok = sum(c.get("ok") for c in rec["cycles"])
    print(f"killed; {ok}/{len(rec['cycles'])} resume cycles fully ok")


if __name__ == "__main__":
    {"create": create, "check": check, "kill": kill}[sys.argv[1]](sys.argv[2])
