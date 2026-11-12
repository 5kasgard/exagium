"""bench-4: one harness attempt, run from the host. Standard library only.

usage: attempt.py HARNESS TASK_MD WORKSPACE OUT_DIR IMAGE MODEL CAP WALL_S METER_HOST [PORT]
Env: OPENROUTER_API_KEY (passed to the meter only).

1. Starts meter.py on METER_HOST:PORT (the docker bridge gateway), logging to OUT_DIR/meter.jsonl.
2. Runs the harness in IMAGE with only WORKSPACE (at /workspace), the task and OUT_DIR/harness mounted.
3. Every 2 s, copies WORKSPACE/record into OUT_DIR/snaps/NNN when its contents change, with the number of
   model calls the meter has forwarded so far (the same clock for every harness).
4. Stops the container 30 s after the meter first refuses a call at the cap, or at WALL_S.
Writes OUT_DIR/run.json: ending ("exited", "call_cap", "timeout", "start_failed"), harness exit code, wall time,
snapshot list. The network seal is set up by the workflow, outside this script.
"""
import hashlib, json, os, shutil, subprocess, sys, threading, time, urllib.request

H, TASK, WS, OUT, IMAGE, MODEL, CAP, WALL, MHOST = sys.argv[1:10]
PORT = int(sys.argv[10]) if len(sys.argv) > 10 else 8088
CAP, WALL = int(CAP), int(WALL)
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, "snaps"), exist_ok=True)
os.makedirs(os.path.join(OUT, "harness"), exist_ok=True)
run = {"harness": H, "model": MODEL, "cap": CAP, "wall_cap_s": WALL, "image": IMAGE, "snaps": [], "snapshot_errors": []}
rec = os.path.join(WS, "record")
name = f"b4-{os.getpid()}"


def meter_count():
    try:
        return json.loads(urllib.request.urlopen(f"http://{MHOST}:{PORT}/meter/count", timeout=5).read())
    except Exception:
        return None


def tree_hash(d):
    h = hashlib.sha256()
    for root, dirs, files in os.walk(d):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            p = os.path.join(root, f)
            try:
                if os.path.isfile(p) and os.path.getsize(p) < 2_000_000:
                    h.update(os.path.relpath(p, d).encode() + b"\0" + open(p, "rb").read() + b"\0")
            except OSError:
                pass
    return h.hexdigest()


last = {"h": None}


def snapshot(calls):
    try:
        h = tree_hash(rec)
        if h != last["h"]:
            i = len(run["snaps"])
            shutil.copytree(rec, os.path.join(OUT, "snaps", f"{i:03d}"), ignore=shutil.ignore_patterns("__pycache__"))
            run["snaps"].append({"i": i, "calls": calls, "hash": h, "t": round(time.time() - t0, 1)})
            last["h"] = h
    except Exception as e:  # a snapshot must never break an attempt
        run["snapshot_errors"].append(f"{type(e).__name__}: {e}"[:300])


meter = subprocess.Popen([sys.executable, os.path.join(here, "meter.py"), MHOST, str(PORT), os.path.join(OUT, "meter.jsonl"),
                          MODEL, str(CAP)], stdout=open(os.path.join(OUT, "meter.out"), "w"), stderr=subprocess.STDOUT)
for _ in range(50):
    if meter_count() is not None:
        break
    time.sleep(0.2)
# Seal check: before the harness runs, a throwaway container from the same image must reach the meter and
# be refused by the outside world. If not, the attempt does not start.
probe = r"""
import socket, urllib.request, json, sys
r = {}
try:
    r["meter"] = json.loads(urllib.request.urlopen(sys.argv[1] + "/meter/count", timeout=10).read()) is not None
except Exception as e:
    r["meter"] = False
for name, host, port in (("ip", "1.1.1.1", 443), ("dns", "github.com", 443)):
    try:
        socket.create_connection((host, port), timeout=8).close(); r[name] = "reached"
    except Exception as e:
        r[name] = "blocked"
print(json.dumps(r))
"""
if os.environ.get("BENCH4_SEAL_CHECK") == "1":
    p = subprocess.run(["docker", "run", "--rm", IMAGE, "python3", "-c", probe, f"http://{MHOST}:{PORT}"],
                       capture_output=True, text=True, timeout=120)
    try:
        seal = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        seal = {"error": (p.stdout + p.stderr)[-300:]}
    seal["ok"] = seal.get("meter") is True and seal.get("ip") == "blocked" and seal.get("dns") == "blocked"
    run["seal"] = seal
    if not seal["ok"]:
        meter.terminate()
        run["ending"] = "seal_failed"
        json.dump(run, open(os.path.join(OUT, "run.json"), "w"), indent=1)
        print(json.dumps(run)); sys.exit(0)
t0 = time.time()
snapshot(0)
shutil.copyfile(TASK, os.path.join(OUT, "harness", "task.md"))
cmd = ["docker", "run", "--rm", "--name", name, "--cpus", "2", "--memory", "6g", "--pids-limit", "2048",
       "-v", f"{os.path.abspath(WS)}:/workspace", "-v", f"{os.path.abspath(OUT)}/harness:/out",
       "-e", f"BENCH4_BASE=http://{MHOST}:{PORT}", "-e", f"BENCH4_MODEL={MODEL}",
       IMAGE, "sh", "/opt/bench4/run_in_container.sh", H, "/out/task.md", "/out"]
proc = subprocess.Popen(cmd, stdout=open(os.path.join(OUT, "docker.out"), "w"), stderr=subprocess.STDOUT)
ending, cap_seen = None, None
while proc.poll() is None:
    time.sleep(2)
    c = meter_count() or {}
    snapshot(c.get("calls"))
    if c.get("refused") and cap_seen is None:
        cap_seen = time.time()
    if cap_seen and time.time() - cap_seen > 30:
        ending = "call_cap"
    elif time.time() - t0 > WALL:
        ending = "timeout"
    if ending:
        subprocess.run(["docker", "kill", name], capture_output=True)
        proc.wait(timeout=60)
        break
c = meter_count() or {}
snapshot(c.get("calls"))
meter.terminate()
rc_file = os.path.join(OUT, "harness", "rc")
run["harness_rc"] = int(open(rc_file).read().strip()) if os.path.exists(rc_file) else None
run["docker_rc"] = proc.returncode
run["ending"] = ending or ("exited" if run["harness_rc"] is not None else "start_failed")
run["meter"] = c
run["wall_s"] = round(time.time() - t0, 1)
json.dump(run, open(os.path.join(OUT, "run.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in run.items() if k != "snaps"}))
