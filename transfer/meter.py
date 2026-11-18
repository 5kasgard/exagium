"""Score and meter every trained artifact of one task (PLAN.md, step 1).

python3 transfer/meter.py TASK_DIR WEIGHTS_DIR OUT_JSON [--rounds 7] [--timing-items 20000] [--seed 1]

WEIGHTS_DIR holds <family>-s<seed>.bin (+ .pred, + .json: the training log line) and the compiled programs
infer_table and infer_lgn. For each artifact:
  - per-item outcomes: the C program's class for every eval item, which must equal families.py's numpy run of
    the same exported form byte for byte (else the artifact is listed under "problems");
  - instructions per item: valgrind (cachegrind, no cache simulation) "I refs" of one process classifying all
    eval items, minus the same at N = 0, divided by n_eval: the marginal count, start-up removed;
  - native ns per item: wall time of one process on a timing file (eval items repeated in order up to
    --timing-items), minus the same process at N = 0, divided by the item count. Rounds are the unit of
    replication; every round runs every artifact in a fresh random order, pinned to one CPU.
No jail: every program measured here was written in this plan, not proposed (PLAN.md, "Limits").
Exit codes: 0 done, 1 a check failed (see "problems"), 3 failure.
"""
import argparse, base64, glob, json, os, platform, random, re, subprocess, sys, time

import numpy as np


def ir(prog, w, x, n):
    r = subprocess.run(["valgrind", "--tool=cachegrind", "--cache-sim=no", "--cachegrind-out-file=/dev/null",
                        prog, w, x, str(n)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    m = re.search(r"I\s+refs:\s+([\d,]+)", r.stderr)
    if r.returncode or not m:
        raise RuntimeError(f"valgrind failed on {w}: {r.stderr[-400:]}")
    return int(m.group(1).replace(",", ""))


def wall_ns(prog, w, x, n, cpu):
    t0 = time.perf_counter_ns()
    r = subprocess.run([prog, w, x, str(n)], stdout=subprocess.DEVNULL,
                       preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
    t1 = time.perf_counter_ns()
    if r.returncode:
        raise RuntimeError(f"{prog} {w} exited {r.returncode}")
    return t1 - t0


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("wdir"); ap.add_argument("out")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--timing-items", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    card = json.load(open(os.path.join(a.task, "card.json")))
    d, n = card["d"], card["n_eval"]
    ex = os.path.join(a.task, "eval.x")
    y = np.fromfile(os.path.join(a.task, "eval.y"), dtype=np.uint8)
    reps = -(-a.timing_items // n)
    tx = os.path.join(a.wdir, "timing.x")
    np.tile(np.fromfile(ex, dtype=np.uint8).reshape(n, d), (reps, 1))[:a.timing_items].tofile(tx)
    problems, arts = [], {}
    for w in sorted(glob.glob(os.path.join(a.wdir, "*-s*.bin"))):
        name = os.path.basename(w)[:-4]
        fam = name.rsplit("-s", 1)[0]
        prog = os.path.join(a.wdir, "infer_lgn" if fam == "lgn" else "infer_table")
        r = subprocess.run([prog, w, ex, str(n)], stdout=subprocess.PIPE)
        pred = np.frombuffer(r.stdout, dtype=np.uint8)
        ref = np.fromfile(w + ".pred", dtype=np.uint8)
        if r.returncode or len(pred) != n or not np.array_equal(pred, ref):
            problems.append(f"{name}: C output differs from the numpy run of the exported form")
        ir_n, ir_0 = ir(prog, w, ex, n), ir(prog, w, ex, 0)
        arts[name] = dict(family=fam, seed=int(name.rsplit("-s", 1)[1]), prog=os.path.basename(prog),
                          train=json.load(open(w[:-4] + ".json")), accuracy=float((pred == y).mean()),
                          ir_per_item=(ir_n - ir_0) / n, ir_startup=ir_0, ns_rounds=[],
                          pred_b64=base64.b64encode(pred.tobytes()).decode())
    rng = random.Random(a.seed)
    cpu = sorted(os.sched_getaffinity(0))[-1]
    names = sorted(arts)
    for _ in range(a.rounds):
        rng.shuffle(names)
        for name in names:
            w = os.path.join(a.wdir, name + ".bin")
            prog = os.path.join(a.wdir, arts[name]["prog"])
            t0 = wall_ns(prog, w, tx, 0, cpu)
            tn = wall_ns(prog, w, tx, a.timing_items, cpu)
            arts[name]["ns_rounds"].append((tn - t0) / a.timing_items)
    res = dict(task=card["task"], card=card, n_eval=n, eval_y_b64=base64.b64encode(y.tobytes()).decode(),
               rounds=a.rounds, timing_items=a.timing_items, host=dict(cpu=cpu_model(), machine=platform.machine(),
               kernel=platform.release(), runner=os.environ.get("RUNNER_NAME", ""),
               run_id=os.environ.get("GITHUB_RUN_ID", "")), artifacts=arts, problems=problems)
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    for name in sorted(arts):
        s = arts[name]
        print(f"{name:12s} acc {s['accuracy']:.4f}  Ir/item {s['ir_per_item']:9.1f}  "
              f"ns/item {np.median(s['ns_rounds']):8.1f}")
    for p in problems:
        print("PROBLEM", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print("FAILURE", e); sys.exit(3)
