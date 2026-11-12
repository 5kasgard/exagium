"""bench-4: one harness attempt, run from the host. Standard library only.

usage: attempt.py HARNESS TASK_MD WORKSPACE OUT_DIR IMAGE MODEL CAP WALL_S METER_HOST [PORT]
Env: BENCH4_KEY_FILE, a file holding the key; it is handed to the meter, which reads and deletes it. This
process never reads the key, and the key is in no process's environment, so nothing this process copies
from the agent's workspace can carry it out.

1. Starts meter.py on METER_HOST:PORT (the docker bridge gateway), logging to OUT_DIR/meter.jsonl.
2. Unless BENCH4_SEAL_CHECK=0 (local tests only), a throwaway container from the same image must reach the meter and be refused by
   the outside world, or the attempt does not start (ending "seal_failed").
3. Runs the harness in IMAGE with only WORKSPACE (at /workspace) and OUT_DIR/harness mounted.
4. Every 2 s, copies WORKSPACE/record (symlinks copied as links, never followed) and keeps the copy as
   OUT_DIR/snaps/NNN if its regular files changed, with the number of model calls forwarded so far.
5. Stops the container 30 s after the meter first refuses a call at the cap, or at WALL_S.
Always writes OUT_DIR/run.json: ending ("exited", "call_cap", "timeout", "start_failed", "seal_failed"),
harness exit code, wall time, snapshot list.
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time, urllib.request

H, TASK, WS, OUT, IMAGE, MODEL, CAP, WALL, MHOST = sys.argv[1:10]
PORT = int(sys.argv[10]) if len(sys.argv) > 10 else 8088
CAP, WALL = int(CAP), int(WALL)
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, "snaps"), exist_ok=True)
os.makedirs(os.path.join(OUT, "harness"), exist_ok=True)
run = {"harness": H, "model": MODEL, "cap": CAP, "wall_cap_s": WALL, "image": IMAGE, "snaps": [], "snapshot_errors": []}
rec = os.path.join(WS, "record")
name = f"b4-{os.getpid()}"
last = {"h": None}
t0 = time.time()

PROBE = r"""
import socket, urllib.request, json, sys
r = {}
try:
    r["meter"] = json.loads(urllib.request.urlopen(sys.argv[1] + "/meter/count", timeout=10).read()) is not None
except Exception:
    r["meter"] = False
for name, host, port in (("ip", "1.1.1.1", 443), ("ip6", "2606:4700:4700::1111", 443), ("dns", "github.com", 443)):
    try:
        socket.create_connection((host, port), timeout=8).close(); r[name] = "reached"
    except Exception:
        r[name] = "blocked"
print(json.dumps(r))
"""


def meter_count():
    try:
        return json.loads(urllib.request.urlopen(f"http://{MHOST}:{PORT}/meter/count", timeout=5).read())
    except Exception:
        return None


def tree_hash(d):
    """Hash of the regular files under d. Links are neither followed nor read."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(d, followlinks=False):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__" and not os.path.islink(os.path.join(root, x)))
        for f in sorted(files):
            p = os.path.join(root, f)
            if os.path.islink(p):
                h.update(b"L" + os.path.relpath(p, d).encode() + b"\0")
                continue
            try:
                if os.path.isfile(p) and os.path.getsize(p) < 2_000_000:
                    h.update(os.path.relpath(p, d).encode() + b"\0" + open(p, "rb").read() + b"\0")
            except OSError:
                pass
    return h.hexdigest()


def snapshot():
    """Copy first (links as links), then hash the copy, then read the call count: a snapshot is one state,
    labelled with the count at or after the time it was taken."""
    tmp = None
    try:
        tmp = tempfile.mkdtemp(dir=OUT)
        dst = os.path.join(tmp, "record")
        shutil.copytree(rec, dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
        h = tree_hash(dst)
        calls = (meter_count() or {}).get("calls")
        if h != last["h"]:
            i = len(run["snaps"])
            os.replace(dst, os.path.join(OUT, "snaps", f"{i:03d}"))
            run["snaps"].append({"i": i, "calls": calls, "hash": h, "t": round(time.time() - t0, 1)})
            last["h"] = h
    except Exception as e:  # a snapshot must never break an attempt
        run["snapshot_errors"].append(f"{type(e).__name__}: {e}"[:300])
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


menv = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "OPENROUTER_API_KEY_FILE": os.environ.get("BENCH4_KEY_FILE", ""),
        "BENCH4_UPSTREAM": os.environ.get("BENCH4_UPSTREAM", "https://openrouter.ai/api/v1")}
meter = subprocess.Popen([sys.executable, "-I", os.path.join(here, "meter.py"), MHOST, str(PORT),
                          os.path.join(OUT, "meter.jsonl"), MODEL, str(CAP)], env=menv,
                         stdout=open(os.path.join(OUT, "meter.out"), "w"), stderr=subprocess.STDOUT)
proc = None
try:
    for _ in range(50):
        if meter_count() is not None:
            break
        time.sleep(0.2)
    if os.environ.get("BENCH4_SEAL_CHECK", "1") != "0":
        p = subprocess.run(["docker", "run", "--rm", IMAGE, "python3", "-c", PROBE, f"http://{MHOST}:{PORT}"],
                           capture_output=True, text=True, timeout=120)
        try:
            seal = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            seal = {"error": (p.stdout + p.stderr)[-300:]}
        seal["ok"] = seal.get("meter") is True and all(seal.get(k) == "blocked" for k in ("ip", "ip6", "dns"))
        run["seal"] = seal
        if not seal["ok"]:
            run["ending"] = "seal_failed"
            raise SystemExit(0)
    t0 = time.time()
    snapshot()
    shutil.copyfile(TASK, os.path.join(OUT, "harness", "task.md"))
    cmd = ["docker", "run", "--rm", "--name", name, "--cpus", "2", "--memory", "6g", "--pids-limit", "2048",
           "-v", f"{os.path.abspath(WS)}:/workspace", "-v", f"{os.path.abspath(OUT)}/harness:/out",
           "-e", f"BENCH4_BASE=http://{MHOST}:{PORT}", "-e", f"BENCH4_MODEL={MODEL}",
           IMAGE, "sh", "/opt/bench4/run_in_container.sh", H, "/out/task.md", "/out"]
    proc = subprocess.Popen(cmd, stdout=open(os.path.join(OUT, "docker.out"), "w"), stderr=subprocess.STDOUT)
    ending, cap_seen = None, None
    while proc.poll() is None:
        time.sleep(2)
        snapshot()
        c = meter_count() or {}
        if c.get("refused") and cap_seen is None:
            cap_seen = time.time()
        if cap_seen and time.time() - cap_seen > 30:
            ending = "call_cap"
        elif time.time() - t0 > WALL:
            ending = "timeout"
        if ending:
            subprocess.run(["docker", "kill", name], capture_output=True)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
    snapshot()
    try:
        run["harness_rc"] = int(open(os.path.join(OUT, "harness", "rc")).read().strip())
    except (OSError, ValueError):
        run["harness_rc"] = None
    run["docker_rc"] = proc.returncode
    run["meter_final"] = meter_count() or {}
    if ending is None and run["meter_final"].get("refused"):
        ending = "call_cap"          # it reached the cap, then stopped by itself within the 30 s grace
    run["ending"] = ending or ("exited" if run["harness_rc"] is not None else "start_failed")
except SystemExit:
    pass
except Exception as e:
    run["ending"] = run.get("ending") or "start_failed"
    run["attempt_error"] = f"{type(e).__name__}: {e}"[:500]
finally:
    if proc is not None and proc.poll() is None:
        subprocess.run(["docker", "kill", name], capture_output=True)
    run["meter"] = meter_count() or {}
    meter.terminate()
    try:
        meter.wait(10)
    except subprocess.TimeoutExpired:
        meter.kill()
    run["wall_s"] = round(time.time() - t0, 1)
    json.dump(run, open(os.path.join(OUT, "run.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in run.items() if k != "snaps"}))
