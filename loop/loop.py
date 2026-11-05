#!/usr/bin/env python3
"""exagium first loop: train, freeze, reproduce and score small classifiers on one task.

A candidate is a directory holding exactly train.py, infer.c and weights.bin (plus an optional
README.md). The contracts:
  train:  python3 -s -P train.py train.csv weights.bin   in a fresh sandbox holding only train.py and a
          copy of the training file, fixed environment, no network (if --no-net). Must be deterministic.
  infer:  ./infer weights.bin x.bin N   prints N digits ('0'..'9') then a newline; x.bin is N rows of
          64 uint8 pixels. N = 0 is startup only (load the weights, classify nothing).

    python3 loop.py freeze DIR...            # digests for a plan's "inputs"
    python3 loop.py reproduce DIR [--no-net] # retrain in a sandbox; weights must be byte-identical
    python3 loop.py evaluate DIR... [--test dev | --test test --plan SEQ] [--rounds R] [--passes P] [--seed S]

evaluate scores every candidate on the same machine, interleaved:
  - accuracy on the test file (UCI's writer-disjoint test set), with paired differences and SE;
  - simulated instructions (cachegrind, via counter/count.py, reference cache): count with N images
    minus count with N = 0, per image and per correct answer. Deterministic, labelled "simulated";
  - native time per image: (time for N*P images) - (time for 0), R interleaved rounds, mean and SE.
--test dev scores on the last 25% of the training file instead: for smoke tests and for proposers,
who must never see the test set before their plan is in the ledger. --test test refuses to run unless
the ledger verifies and entry SEQ is a plan whose "inputs" hold every digest `freeze` gives for these
candidates: the test set is scored only for candidates frozen in a plan already on record.

Exit codes: 0 done; 1 a check failed (weights not reproduced, output malformed or inconsistent,
counter not exact or clean); 3 failure. Stdlib only. Linux, gcc, valgrind.
"""
import argparse, hashlib, json, math, os, platform, random, shutil, statistics, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TASK = {"train": os.path.join(HERE, "task", "optdigits.tra"), "test": os.path.join(HERE, "task", "optdigits.tes")}
COUNT = os.path.join(REPO, "counter", "count.py")
FILES = ("train.py", "infer.c", "weights.bin")
OPTIONAL = ("README.md",)
CFLAGS = ["-O2", "-std=c11", "-ffp-contract=off"]  # an explicit choice: no -march, so the counted path is the one that runs
ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C", "LANG": "C", "TZ": "UTC", "PYTHONHASHSEED": "0",
       "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
TRAIN_TIMEOUT = 1800


class Failed(Exception):
    """A check failed: exit 1."""


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def rel(p):
    return os.path.relpath(os.path.abspath(p), REPO)


def candidate(d):
    d = os.path.abspath(d)
    have = sorted(os.listdir(d))
    extra = [f for f in have if f not in FILES + OPTIONAL and f != "infer"]
    missing = [f for f in FILES if f not in have]
    if missing or extra:
        raise Failed(f"{rel(d)}: needs exactly {', '.join(FILES)} (optional README.md); missing {missing}, extra {extra}")
    for f in FILES:
        if os.path.islink(os.path.join(d, f)) or not os.path.isfile(os.path.join(d, f)):
            raise Failed(f"{rel(d)}/{f}: must be a regular file")
    return d


def load(path):
    """Rows of 64 pixels in 0..16 and a label in 0..9, from a UCI optdigits CSV."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            v = [int(t) for t in line.strip().split(",")]
            if len(v) != 65 or not all(0 <= t <= 16 for t in v[:64]) or not 0 <= v[64] <= 9:
                raise Failed(f"{path}:{i + 1}: malformed row")
            rows.append(v)
    return rows


def split(name):
    if name == "test":
        return load(TASK["test"])
    tr = load(TASK["train"])
    return tr[len(tr) * 3 // 4:]  # dev: the same holdout the baselines use to pick their L2 strength


def compile_infer(d):
    exe = os.path.join(d, "infer")
    r = subprocess.run(["gcc", *CFLAGS, "-o", exe, os.path.join(d, "infer.c"), "-lm"], capture_output=True, text=True)
    if r.returncode:
        raise Failed(f"{rel(d)}/infer.c does not compile:\n{r.stderr}")
    return exe


def gcc_version():
    return subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]


def digests(dirs):
    tools = (TASK["train"], TASK["test"], os.path.abspath(__file__), COUNT)
    out = [{"name": rel(p), "sha256": sha256_file(p)} for p in tools]
    for d in dirs:
        out += [{"name": rel(os.path.join(d, f)), "sha256": sha256_file(os.path.join(d, f))} for f in FILES]
    return out


def require_plan(seq, dirs):
    r = subprocess.run([sys.executable, "-I", os.path.join(REPO, "record", "record.py"), "verify"], capture_output=True, text=True)
    if r.returncode:
        raise Failed(f"the ledger does not verify (exit {r.returncode}):\n{r.stdout}{r.stderr}")
    path = os.path.join(REPO, "ledger", "entries", f"{seq:06d}.json")
    with open(path, "rb") as f:
        e = json.loads(f.read())
    if e.get("kind") != "plan":
        raise Failed(f"ledger entry {seq} is not a plan")
    have = {(x["name"], x["sha256"]) for x in e["body"].get("inputs", [])}
    missing = [x["name"] for x in digests(dirs) if (x["name"], x["sha256"]) not in have]
    if missing:
        raise Failed(f"plan {seq} does not freeze these files as they are now: {', '.join(missing)}")
    return {"seq": seq, "sha256": sha256_file(path)}


# ---------- reproduce ----------
def reproduce(d, no_net):
    want = sha256_file(os.path.join(d, "weights.bin"))
    with tempfile.TemporaryDirectory(prefix="exagium-train-") as sb:
        shutil.copyfile(os.path.join(d, "train.py"), os.path.join(sb, "train.py"))
        shutil.copyfile(TASK["train"], os.path.join(sb, "train.csv"))
        env = dict(ENV, HOME=sb, TMPDIR=sb)
        cmd = [sys.executable, "-s", "-P", "train.py", "train.csv", "weights.bin"]
        if no_net:
            cmd = ["unshare", "-rn", "--"] + cmd
        t0 = time.monotonic()
        r = subprocess.run(cmd, cwd=sb, env=env, capture_output=True, text=True, timeout=TRAIN_TIMEOUT)
        secs = time.monotonic() - t0
        if r.returncode:
            raise Failed(f"{rel(d)}: train.py exited {r.returncode}:\n{r.stderr[-2000:]}")
        got = os.path.join(sb, "weights.bin")
        if not os.path.isfile(got):
            raise Failed(f"{rel(d)}: train.py wrote no weights.bin")
        have = sha256_file(got)
    return {"candidate": rel(d), "expected_sha256": want, "retrained_sha256": have, "identical": have == want,
            "seconds": round(secs, 1), "no_net": no_net, "python": platform.python_version(),
            "numpy": _numpy_version(), "machine": platform.machine(), "cpu": cpu_model()}


def _numpy_version():
    r = subprocess.run([sys.executable, "-c", "import numpy; print(numpy.__version__)"], capture_output=True, text=True)
    return r.stdout.strip() or None


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.split(":")[0].strip() in ("model name", "CPU part"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


# ---------- evaluate ----------
def run_infer(exe, wdir, xfile, n):
    r = subprocess.run([exe, "weights.bin", xfile, str(n)], cwd=wdir, env=ENV, capture_output=True, timeout=600)
    if r.returncode:
        raise Failed(f"{exe} exited {r.returncode} for N={n}")
    return r.stdout


def predictions(out, n, who):
    if len(out) != n + 1 or out[-1:] != b"\n" or any(c < 48 or c > 57 for c in out[:n]):
        raise Failed(f"{who}: output must be exactly {n} digits and a newline")
    return [c - 48 for c in out[:n]]


def count(d, exe, xfile, n, outdir):
    """Counter record for N images, run in its fixed sandbox; returns (record, its path)."""
    os.makedirs(outdir, exist_ok=True)
    before = set(os.listdir(outdir))
    with tempfile.TemporaryDirectory(prefix="exagium-cnt-") as td:
        for src, name in ((exe, "infer"), (os.path.join(d, "weights.bin"), "weights.bin"), (xfile, "x.bin")):
            shutil.copy2(src, os.path.join(td, name))
        r = subprocess.run([sys.executable, "-I", COUNT, "--reps", "2", "--out-dir", outdir,
                            "--input", os.path.join(td, "infer"), "--input", os.path.join(td, "weights.bin"),
                            "--input", os.path.join(td, "x.bin"), "--", "./infer", "weights.bin", "x.bin", str(n)],
                           capture_output=True, text=True)
    new = sorted(set(os.listdir(outdir)) - before)
    if r.returncode == 3 or len(new) != 1:
        raise Failed(f"{rel(d)}: counter failed (exit {r.returncode}):\n{r.stdout}{r.stderr}")
    path = os.path.join(outdir, new[0])
    with open(path) as f:
        rec = json.load(f)
    return rec, path, r.returncode


def mean_se(xs):
    m = statistics.fmean(xs)
    return m, (statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None)


def evaluate(dirs, test, rounds, passes, seed, outdir):
    rows = split(test)
    n = len(rows)
    labels = [r[64] for r in rows]
    work = tempfile.mkdtemp(prefix="exagium-eval-")
    xfile = os.path.join(work, "x.bin")
    with open(xfile, "wb") as f:
        f.write(bytes(v for r in rows for v in r[:64]))
    rng = random.Random(seed)
    perms = [rng.sample(range(n), n) for _ in range(passes)]  # timing input: P shuffled copies, so no pass repeats the last
    tfile = os.path.join(work, "xt.bin")
    with open(tfile, "wb") as f:
        f.write(bytes(v for p in perms for i in p for v in rows[i][:64]))
    empty = os.path.join(work, "x0.bin")
    open(empty, "wb").close()

    cands, problems = [], []
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        exe = compile_infer(d)
        out = run_infer(exe, d, xfile, n)
        pred = predictions(out, n, name)
        c = {"name": name, "dir": rel(d), "exe": exe, "pred": pred,
             "weights_bytes": os.path.getsize(os.path.join(d, "weights.bin")),
             "correct": sum(p == y for p, y in zip(pred, labels))}
        rec_n, path_n, rc_n = count(d, exe, xfile, n, outdir)
        rec_0, path_0, rc_0 = count(d, exe, empty, 0, outdir)
        if rec_n["reps"][0]["stdout_sha256"] != sha256_bytes(out):
            problems.append(f"{name}: output under the counter differs from the native output")
        for rc, rec in ((rc_n, rec_n), (rc_0, rec_0)):
            if rc != 0:
                problems.append(f"{name}: counter exit {rc} (exact={rec['verdict']['exact']}, "
                                f"client_request_code={rec['verdict']['client_request_code']})")
        cn, c0 = rec_n["reps"][0]["counts"], rec_0["reps"][0]["counts"]
        c["counts_minus_startup"] = {k: cn[k] - c0[k] for k in cn}
        c["counter_records"] = [{"name": rel(p), "sha256": sha256_file(p)} for p in (path_n, path_0)]
        c["startup_Ir"] = c0["Ir"]
        cands.append(c)

    # native timing, interleaved: every round runs each candidate on N*P images and on 0, in a fresh random order
    times = {c["name"]: [] for c in cands}
    for _ in range(rounds):
        jobs = [(c, k) for c in cands for k in ("full", "zero")]
        rng.shuffle(jobs)
        t = {}
        for c, k in jobs:
            t0 = time.perf_counter_ns()
            out = run_infer(c["exe"], os.path.dirname(c["exe"]), tfile if k == "full" else empty, n * passes if k == "full" else 0)
            t[(c["name"], k)] = time.perf_counter_ns() - t0
            if k == "full":
                want = [c["pred"][i] for p in perms for i in p]
                if predictions(out, n * passes, c["name"]) != want:
                    problems.append(f"{c['name']}: timing run's predictions differ from its counted predictions")
        for c in cands:
            times[c["name"]].append((t[(c["name"], "full")] - t[(c["name"], "zero")]) / (n * passes))

    board = []
    for c in cands:
        ir = c["counts_minus_startup"]["Ir"]
        tm, tse = mean_se(times[c["name"]])
        acc = c["correct"] / n
        board.append({
            "name": c["name"], "dir": c["dir"], "correct": c["correct"], "n": n, "accuracy": acc,
            "accuracy_se": math.sqrt(acc * (1 - acc) / n),
            "simulated_Ir_per_image": ir / n, "simulated_Ir_per_correct": ir / c["correct"] if c["correct"] else None,
            "simulated_counts_minus_startup": c["counts_minus_startup"], "simulated_startup_Ir": c["startup_Ir"],
            "native_ns_per_image": tm, "native_ns_per_image_se": tse, "native_ns_per_image_rounds": times[c["name"]],
            "weights_bytes": c["weights_bytes"], "counter_records": c["counter_records"],
            "predictions_sha256": sha256_bytes(bytes(48 + p for p in c["pred"])),
        })
    pairs = []
    for a in cands:
        for b in cands:
            if a is b:
                continue
            da = [(pa == y) - (pb == y) for pa, pb, y in zip(a["pred"], b["pred"], labels)]
            dt = [x - y for x, y in zip(times[a["name"]], times[b["name"]])]
            am, ase = mean_se(da)
            tm, tse = mean_se(dt)
            pairs.append({"a": a["name"], "b": b["name"], "accuracy_diff": am, "accuracy_diff_se": ase,
                          "Ir_ratio": a["counts_minus_startup"]["Ir"] / b["counts_minus_startup"]["Ir"],
                          "ns_per_image_diff": tm, "ns_per_image_diff_se": tse})
    shutil.rmtree(work)
    return {"schema": "exagium-loop/1", "split": test, "n": n, "rounds": rounds, "passes": passes, "seed": seed,
            "labels": {"accuracy": "measured on the split", "simulated_*": "cachegrind, reference cache, count(N) - count(0)",
                       "native_*": "wall time on this host, (N*P images) - (0 images), interleaved rounds"},
            "host": {"cpu": cpu_model(), "machine": platform.machine(), "kernel": platform.release(),
                     "nproc": os.cpu_count()},
            "cflags": CFLAGS, "gcc": gcc_version(), "inputs": digests(dirs),
            "candidates": board, "pairs": pairs, "problems": problems}


def show(s):
    print(f"split {s['split']} (n={s['n']}), {s['rounds']} rounds x {s['passes']} passes, host {s['host']['cpu']}")
    for c in s["candidates"]:
        se = c["native_ns_per_image_se"]
        print(f"  {c['name']:14s} acc {c['accuracy']:.4f} ±{c['accuracy_se']:.4f}  "
              f"Ir/img {c['simulated_Ir_per_image']:>9,.0f}  Ir/correct {c['simulated_Ir_per_correct'] or 0:>9,.0f}  "
              f"ns/img {c['native_ns_per_image']:8.1f}" + (f" ±{se:.1f}" if se is not None else ""))
    for p in s["problems"]:
        print("  PROBLEM:", p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze"); f.add_argument("dirs", nargs="+")
    r = sub.add_parser("reproduce"); r.add_argument("dirs", nargs="+"); r.add_argument("--no-net", action="store_true")
    r.add_argument("--out")
    e = sub.add_parser("evaluate"); e.add_argument("dirs", nargs="+")
    e.add_argument("--test", choices=("test", "dev"), default="dev")
    e.add_argument("--rounds", type=int, default=15); e.add_argument("--passes", type=int, default=30)
    e.add_argument("--seed", type=int, default=1); e.add_argument("--out", help="write the scoreboard JSON here")
    e.add_argument("--plan", type=int, help="ledger seq of the plan that froze these candidates (required for --test test)")
    e.add_argument("--records", default=os.path.join(REPO, "runs", "loop"), help="where counter records go")
    a = ap.parse_args(argv)
    try:
        dirs = [candidate(d) for d in a.dirs]
        if a.cmd == "freeze":
            print(json.dumps(digests(dirs), indent=1))
            return 0
        if a.cmd == "reproduce":
            res = [reproduce(d, a.no_net) for d in dirs]
            for x in res:
                print(f"{x['candidate']}: {'identical' if x['identical'] else 'DIFFERENT'} weights "
                      f"({x['seconds']} s, {x['cpu']}, numpy {x['numpy']}, no_net={x['no_net']})")
            if a.out:
                with open(a.out, "w") as fh:
                    json.dump(res, fh, indent=1)
            return 0 if all(x["identical"] for x in res) else 1
        if a.rounds < 2 or a.passes < 1:
            ap.error("--rounds must be >= 2 and --passes >= 1")
        plan = None
        if a.test == "test":
            if a.plan is None:
                ap.error("--test test needs --plan SEQ: the test set is scored only for candidates frozen in a plan")
            plan = require_plan(a.plan, dirs)
        s = evaluate(dirs, a.test, a.rounds, a.passes, a.seed, a.records)
        s["plan"] = plan
        show(s)
        if a.out:
            with open(a.out, "x") as fh:  # never overwrite a scoreboard
                json.dump(s, fh, indent=1)
            print(f"wrote {a.out} sha256 {sha256_file(a.out)}")
        return 1 if s["problems"] else 0
    except Failed as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # anything unexpected is a failure, never a verdict
        print(f"loop failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
