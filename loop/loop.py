#!/usr/bin/env python3
"""exagium first loop: train, freeze, reproduce and score small classifiers on one task.

A candidate is a directory holding exactly train.py, infer.c and weights.bin (plus an optional
README.md that is never shown to the candidate's code). The contracts:
  train:  python3 -s -P train.py train.csv weights.bin  -- must be deterministic.
  infer:  ./infer weights.bin x.bin N  -- prints N digits ('0'..'9') then a newline; x.bin is N rows
          of 64 uint8 pixels. N = 0 is startup only and must print just the newline.
All candidate code runs in a jail (bubblewrap): fresh tmpfs /tmp, read-only /usr, no network, its own
PID namespace, uid 65534 with every capability dropped, a fresh session keyring, and a working
directory holding only the files that step needs. The repo, the test
labels and the frozen weights are not in it. Known copies of the dataset (files named optdigits* or
digits.csv*) under /usr and the Python prefix are masked.

    python3 loop.py freeze DIR...            # digests for a plan's "inputs"
    python3 loop.py reproduce DIR...         # retrain in the jail; weights must be byte-identical
    python3 loop.py evaluate DIR... [--test dev | --test test --plan SEQ] [--rounds R] [--seed S]

evaluate scores every candidate on the same machine:
  - accuracy on the split, with paired differences and their SE;
  - simulated instructions (cachegrind via counter/count.py, reference cache) for one process that
    classifies all N images, divided by N: startup included, so nothing escapes the count. The count
    for N = 0 is reported alongside as startup;
  - native time for the same one-process run, per image, measured inside the jail by timer.c pinned
    to one CPU; R rounds, candidates in a fresh random order each round, rows freshly shuffled each run.
--test dev scores the last 25% of the training file. It is IN-SAMPLE for any candidate trained on
the whole training file (the baselines are), so it is only a smoke test. --test test refuses unless
(1) the ledger verifies and its anchors check (record.py verify and check-anchors exit 0), (2) entry
SEQ is a plan whose "inputs" hold every digest `freeze` gives now, and (3) every candidate's training
reproduces its weights byte for byte, in the jail, here.

Exit codes: 0 done; 1 a check failed (weights not reproduced, output malformed or inconsistent,
counter not exact or clean, plan missing); 3 failure. Stdlib only. Linux, gcc, valgrind, bwrap.
"""
import argparse, ctypes, hashlib, json, math, os, platform, random, shutil, statistics, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TASK = {"train": os.path.join(HERE, "task", "optdigits.tra"), "test": os.path.join(HERE, "task", "optdigits.tes")}
COUNT = os.path.join(REPO, "counter", "count.py")
RECORD = os.path.join(REPO, "record", "record.py")
TIMER = os.path.join(HERE, "timer.c")
FILES = ("train.py", "infer.c", "weights.bin")
OPTIONAL = ("README.md",)
CFLAGS = ["-O2", "-std=c11", "-ffp-contract=off"]  # an explicit choice: no -march, so the counted path is the one that runs
ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C", "LANG": "C", "TZ": "UTC", "PYTHONHASHSEED": "0",
       "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "HOME": "/tmp", "TMPDIR": "/tmp"}
TRAIN_TIMEOUT = 1800
MASK_NAMES = ("optdigits", "digits.csv")
KEYCTL = {"x86_64": 250, "aarch64": 219}  # keyctl syscall numbers
CPU_SLACK = (1.25, 2_000_000)  # a run whose CPU time exceeds 1.25 x wall + 2 ms used more than one CPU
RATIO_BAND = 30  # native ns per simulated Ir more than 30x the lowest candidate's: count and time disagree


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
    extra = [f for f in have if f not in FILES + OPTIONAL]
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


def digests(dirs):
    tools = (TASK["train"], TASK["test"], os.path.abspath(__file__), TIMER, COUNT, RECORD)
    out = [{"name": rel(p), "sha256": sha256_file(p)} for p in tools]
    for d in dirs:
        out += [{"name": rel(os.path.join(d, f)), "sha256": sha256_file(os.path.join(d, f))} for f in FILES]
    return out


# ---------- the jail ----------
_masks = None


def python_roots():
    roots = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))))}
    return sorted(r for r in roots if not r.startswith("/usr/"))


def masks():
    """Files that look like a copy of the dataset, in what the jail can see (by name only: a limit)."""
    global _masks
    if _masks is None:
        _masks = []
        for root in ["/usr"] + python_roots():
            for dp, dn, fn in os.walk(root):
                _masks += [os.path.join(dp, f) for f in fn if f.lower().startswith(MASK_NAMES)]
    return _masks


def _fresh_keyring():
    """In the child, before exec: join a new anonymous session keyring, so no keys pass between jails."""
    nr = KEYCTL.get(platform.machine())
    if nr is None or ctypes.CDLL(None, use_errno=True).syscall(nr, 1, None) < 0:  # KEYCTL_JOIN_SESSION_KEYRING
        os._exit(126)


def jail(cmd, binds, timeout, cwd="/w"):
    """Run cmd in bubblewrap. binds: [(src, dst, writable)]. Returns CompletedProcess (bytes)."""
    a = ["bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
         "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin", "--proc", "/proc", "--dev", "/dev",
         "--tmpfs", "/tmp", "--unshare-all", "--uid", "65534", "--gid", "65534", "--cap-drop", "ALL",
         "--die-with-parent", "--new-session", "--clearenv"]
    if os.path.exists("/etc/ld.so.cache"):
        a += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    for r in python_roots():
        a += ["--ro-bind", r, r]
    for m in masks():
        a += ["--ro-bind", "/dev/null", m]
    for src, dst, rw in binds:
        a += ["--bind" if rw else "--ro-bind", src, dst]
    for k, v in ENV.items():
        a += ["--setenv", k, v]
    a += ["--chdir", cwd, "--", *cmd]
    return subprocess.run(a, capture_output=True, timeout=timeout, preexec_fn=_fresh_keyring)


def check_jail():
    if not shutil.which("bwrap"):
        raise Failed("bubblewrap (bwrap) is needed: candidate code only runs in the jail")
    r = jail(["/bin/sh", "-c", "id -u; grep -E '^Cap(Eff|Prm|Inh|Bnd|Amb)' /proc/self/status"], [], 60, cwd="/")
    if r.returncode:
        raise Failed(f"bwrap cannot make a jail here: {r.stderr.decode(errors='replace')[-500:]}")
    lines = r.stdout.decode().split()
    if lines[0] != "65534" or any(v != "0000000000000000" for v in lines[2::2]):
        raise Failed(f"the jail still has a privileged uid or capabilities: {r.stdout.decode()}")


# ---------- reproduce ----------
def reproduce(d):
    want = sha256_file(os.path.join(d, "weights.bin"))
    with tempfile.TemporaryDirectory(prefix="exagium-train-") as sb:
        shutil.copyfile(os.path.join(d, "train.py"), os.path.join(sb, "train.py"))
        shutil.copyfile(TASK["train"], os.path.join(sb, "train.csv"))
        r = jail([sys.executable, "-s", "-P", "train.py", "train.csv", "weights.bin"], [(sb, "/w", True)], TRAIN_TIMEOUT)
        if r.returncode:
            raise Failed(f"{rel(d)}: train.py exited {r.returncode}:\n{r.stderr.decode(errors='replace')[-2000:]}")
        got = os.path.join(sb, "weights.bin")
        if os.path.islink(got) or not os.path.isfile(got):
            raise Failed(f"{rel(d)}: train.py wrote no weights.bin")
        have = sha256_file(got)
    return {"candidate": rel(d), "expected_sha256": want, "retrained_sha256": have, "identical": have == want,
            "jailed": True, **versions()}


def versions():
    def first(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
            return (r.stdout or r.stderr).strip().splitlines()[0]
        except (OSError, IndexError):
            return None
    return {"python": platform.python_version(), "numpy": first([sys.executable, "-c", "import numpy; print(numpy.__version__)"]),
            "gcc": first(["gcc", "--version"]), "valgrind": first(["valgrind", "--version"]),
            "machine": platform.machine(), "cpu": cpu_model(), "kernel": platform.release()}


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
def build(src, out):
    r = subprocess.run(["gcc", *CFLAGS, "-o", out, src, "-lm"], capture_output=True, text=True, env=ENV)
    if r.returncode:
        raise Failed(f"{src} does not compile:\n{r.stderr}")


def predictions(out, n, who):
    if len(out) != n + 1 or out[-1:] != b"\n" or any(c < 48 or c > 57 for c in out[:n]):
        raise Failed(f"{who}: output must be exactly {n} digits and a newline")
    return [c - 48 for c in out[:n]]


def timed(c, rows, timer, cpu):
    """One jailed run of the candidate on these rows: (predictions, ns)."""
    with tempfile.TemporaryDirectory(prefix="exagium-run-") as w:
        for src, name in ((c["exe"], "infer"), (os.path.join(c["dir"], "weights.bin"), "weights.bin"), (timer, "timer")):
            shutil.copy2(src, os.path.join(w, name))
        with open(os.path.join(w, "x.bin"), "wb") as f:
            f.write(bytes(v for r in rows for v in r[:64]))
        r = jail(["./timer", str(cpu), "./infer", "weights.bin", "x.bin", str(len(rows))], [(w, "/w", False)], 600)
    out, sep, tail = r.stdout.rpartition(b"\nexagium-timer ")  # the timer's own line is always last
    f = tail.split()
    if r.returncode or not sep or len(f) != 4 or not tail.endswith(b"\n") or not all(x.isdigit() for x in f):
        raise Failed(f"{c['name']}: timer failed (exit {r.returncode}): {r.stderr.decode(errors='replace')[-300:]}")
    ns, status, cpu_ns, pids = (int(x) for x in f)
    if status:
        raise Failed(f"{c['name']}: infer exited {status}")
    if pids != 1:
        raise Failed(f"{c['name']}: made {pids - 1} more process(es) or thread(s); candidates must be one single-threaded process")
    if cpu_ns > CPU_SLACK[0] * ns + CPU_SLACK[1]:
        raise Failed(f"{c['name']}: used more than one CPU ({cpu_ns} ns CPU in {ns} ns wall); candidates must be single-threaded")
    return predictions(out, len(rows), c["name"]), ns


def count(c, rows, outdir):
    """Counter record for one jailed process classifying these rows: (record, path, exit code).
    count.py runs inside the jail, non-dumpable, and hands its record back on stdout only."""
    os.makedirs(outdir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exagium-cnt-") as tool:
        for src, name in ((c["exe"], "infer"), (os.path.join(c["dir"], "weights.bin"), "weights.bin"), (COUNT, "count.py")):
            shutil.copy2(src, os.path.join(tool, name))
        with open(os.path.join(tool, "x.bin"), "wb") as f:
            f.write(bytes(v for r in rows for v in r[:64]))
        r = jail([sys.executable, "-I", "/tool/count.py", "--stdout-record", "--single-process", "--reps", "2", "--input", "/tool/infer",
                  "--input", "/tool/weights.bin", "--input", "/tool/x.bin", "--", "./infer", "weights.bin", "x.bin",
                  str(len(rows))], [(tool, "/tool", False)], 1800, cwd="/tmp")
    if r.returncode not in (0, 1) or r.stdout.count(b"\n") != 1 or not r.stdout.endswith(b"\n"):
        raise Failed(f"{c['name']}: counter failed (exit {r.returncode}):\n{r.stderr.decode(errors='replace')[-1000:]}")
    try:
        rec = json.loads(r.stdout)
        body = {k: v for k, v in rec.items() if k != "digest_sha256"}
        ok = sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()) == rec["digest_sha256"]
        counts = rec["reps"][0]["counts"]
        ok = ok and all(type(counts[k]) is int and counts[k] >= 0 for k in counts) and counts["Ir"] > 0
    except (ValueError, KeyError, TypeError, IndexError):
        ok = False
    if not ok:
        raise Failed(f"{c['name']}: counter record is malformed or its digest doesn't match")
    path = os.path.join(outdir, f"{c['name']}-n{len(rows)}-{rec['digest_sha256'][:16]}.json")
    with open(path, "xb") as f:
        f.write(r.stdout)
    return rec, path, r.returncode


def mean_se(xs):
    m = statistics.fmean(xs)
    return m, (statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None)


def evaluate(dirs, test, rounds, seed, outdir):
    rows = split(test)
    n = len(rows)
    labels = [r[64] for r in rows]
    rng = random.Random(seed)
    cpu = min(os.sched_getaffinity(0))
    names = [os.path.basename(d.rstrip("/")) for d in dirs]
    if len(set(names)) != len(names):
        raise Failed(f"candidate names must be unique: {names}")
    work = tempfile.mkdtemp(prefix="exagium-eval-")
    try:
        timer = os.path.join(work, "timer")
        build(TIMER, timer)
        cands, problems = [], []
        for d, name in zip(dirs, names):
            os.makedirs(os.path.join(work, name))
            src = os.path.join(work, name, "infer.c")  # compiled from a copy: __FILE__ doesn't point into the repo
            shutil.copyfile(os.path.join(d, "infer.c"), src)
            c = {"name": name, "dir": d, "exe": os.path.join(work, name, "infer"),
                 "weights_bytes": os.path.getsize(os.path.join(d, "weights.bin"))}
            build(src, c["exe"])
            rec, path, rc = count(c, rows, outdir)
            rec0, path0, rc0 = count(c, [], outdir)
            for code, r in ((rc, rec), (rc0, rec0)):
                if code != 0:
                    problems.append(f"{name}: counter exit {code} (exact={r['verdict']['exact']}, "
                                    f"client_request_code={r['verdict']['client_request_code']})")
            if rec0["reps"][0]["stdout_sha256"] != sha256_bytes(b"\n"):
                problems.append(f"{name}: with N = 0 it must print only a newline")
            c["pred"], _ = timed(c, rows, timer, cpu)
            if rec["reps"][0]["stdout_sha256"] != sha256_bytes(bytes(48 + p for p in c["pred"]) + b"\n"):
                problems.append(f"{name}: output under the counter differs from the native output")
            c["counts"], c["startup"] = rec["reps"][0]["counts"], rec0["reps"][0]["counts"]
            c["counter_records"] = [{"name": rel(p), "sha256": sha256_file(p)} for p in (path, path0)]
            c["correct"] = sum(p == y for p, y in zip(c["pred"], labels))
            cands.append(c)

        times = {c["name"]: [] for c in cands}
        marg = {c["name"]: [] for c in cands}
        for _ in range(rounds):
            jobs = [(c, k) for c in cands for k in ("full", "zero")]
            rng.shuffle(jobs)
            t = {}
            for c, k in jobs:
                if k == "zero":
                    t[(c["name"], k)] = timed(c, [], timer, cpu)[1]
                    continue
                perm = rng.sample(range(n), n)
                pred, t[(c["name"], k)] = timed(c, [rows[i] for i in perm], timer, cpu)
                if pred != [c["pred"][i] for i in perm]:
                    problems.append(f"{c['name']}: a timing run's predictions differ from its counted predictions")
            for c in cands:
                times[c["name"]].append(t[(c["name"], "full")] / n)
                marg[c["name"]].append((t[(c["name"], "full")] - t[(c["name"], "zero")]) / n)
    finally:
        shutil.rmtree(work)

    ratio = {c["name"]: statistics.fmean(times[c["name"]]) / (c["counts"]["Ir"] / n) for c in cands}
    low = min(ratio.values())
    for name, x in ratio.items():
        if x > RATIO_BAND * low:
            problems.append(f"{name}: native time per simulated instruction is {x / low:.0f}x the lowest candidate's: "
                            f"its count may be falsified (or it is memory-bound far beyond the others)")
    board = []
    for c in cands:
        ir = c["counts"]["Ir"]
        tm, tse = mean_se(times[c["name"]])
        acc = c["correct"] / n
        board.append({
            "name": c["name"], "dir": rel(c["dir"]), "correct": c["correct"], "n": n, "accuracy": acc,
            "accuracy_se": math.sqrt(acc * (1 - acc) / n),
            "simulated_Ir_per_image": ir / n, "simulated_Ir_per_correct": ir / c["correct"] if c["correct"] else None,
            "simulated_counts": c["counts"], "simulated_startup_counts": c["startup"],
            "native_ns_per_image": tm, "native_ns_per_image_se": tse, "native_ns_per_image_rounds": times[c["name"]],
            "marginal_simulated_Ir_per_image": (ir - c["startup"]["Ir"]) / n,
            "marginal_native_ns_per_image": mean_se(marg[c["name"]]),
            "native_ns_per_simulated_Ir": ratio[c["name"]],
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
                          "Ir_ratio": a["counts"]["Ir"] / b["counts"]["Ir"],
                          "ns_per_image_diff": tm, "ns_per_image_diff_se": tse})
    return {"schema": "exagium-loop/2", "split": "test" if test == "test" else "dev-in-sample", "n": n,
            "rounds": rounds, "seed": seed, "cpu_pinned": cpu,
            "labels": {"accuracy": "measured on the split",
                       "simulated_*": "cachegrind, reference cache: one process classifying all N images (startup included), / N",
                       "native_*": "wall time inside the jail, one process classifying all N images (startup included), / N; "
                                   "mean and SE over interleaved rounds",
                       "marginal_*": "minus the N = 0 run, per image: INFORMATIONAL and gameable, since a candidate "
                                     "can move work into its N = 0 run; never use it to decide",
                       "SE": "assumes independent images; the test set has 13 writers, so true SEs are larger"},
            "host": versions(), "cflags": CFLAGS, "inputs": digests(dirs),
            "candidates": board, "pairs": pairs, "problems": problems}


def require_plan(seq, dirs):
    path = os.path.join(REPO, "ledger", "entries", f"{seq:06d}.json")
    if not os.path.isfile(path):
        raise Failed(f"no ledger entry {seq}")
    with open(path, "rb") as f:
        e = json.loads(f.read())
    if e.get("kind") != "plan":
        raise Failed(f"ledger entry {seq} is not a plan")
    have = {(x["name"], x["sha256"]) for x in e["body"].get("inputs", [])}
    missing = [x["name"] for x in digests(dirs) if (x["name"], x["sha256"]) not in have]
    if missing:
        raise Failed(f"plan {seq} does not freeze these files as they are now: {', '.join(missing)}")
    for sub in ("verify", "check-anchors"):
        r = subprocess.run([sys.executable, "-I", RECORD, sub], capture_output=True, text=True, cwd=REPO)
        if r.returncode:
            raise Failed(f"record.py {sub} exit {r.returncode}: the plan can't be trusted here\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
    return {"seq": seq, "sha256": sha256_file(path)}


def show(s):
    print(f"split {s['split']} (n={s['n']}), {s['rounds']} rounds, host {s['host']['cpu']}")
    for c in s["candidates"]:
        se = c["native_ns_per_image_se"]
        ipc = c["simulated_Ir_per_correct"]
        print(f"  {c['name']:14s} acc {c['accuracy']:.4f} ±{c['accuracy_se']:.4f}  "
              f"Ir/img {c['simulated_Ir_per_image']:>9,.0f}  Ir/correct {'-' if ipc is None else f'{ipc:,.0f}':>9}  "
              f"ns/img {c['native_ns_per_image']:8.1f}" + (f" ±{se:.1f}" if se is not None else ""))
        m, mse = c["marginal_native_ns_per_image"]
        print(f"  {'':14s} marginal (informational): Ir/img {c['marginal_simulated_Ir_per_image']:,.0f}  ns/img {m:.1f}"
              + (f" ±{mse:.1f}" if mse is not None else ""))
    for p in s["problems"]:
        print("  PROBLEM:", p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze"); f.add_argument("dirs", nargs="+")
    r = sub.add_parser("reproduce"); r.add_argument("dirs", nargs="+"); r.add_argument("--out")
    e = sub.add_parser("evaluate"); e.add_argument("dirs", nargs="+")
    e.add_argument("--test", choices=("test", "dev"), default="dev")
    e.add_argument("--rounds", type=int, default=30); e.add_argument("--seed", type=int, default=1)
    e.add_argument("--out", help="write the scoreboard JSON here (never overwritten)")
    e.add_argument("--plan", type=int, help="ledger seq of the plan that froze these candidates (required for --test test)")
    e.add_argument("--records", default=os.path.join(REPO, "runs", "loop"), help="where counter records go")
    a = ap.parse_args(argv)
    try:
        dirs = [candidate(d) for d in a.dirs]
        if a.cmd == "freeze":
            print(json.dumps(digests(dirs), indent=1))
            return 0
        if a.cmd == "evaluate" and a.rounds < 2:
            ap.error("--rounds must be >= 2")
        if a.cmd == "evaluate" and a.test == "test" and a.plan is None:
            ap.error("--test test needs --plan SEQ: the test set is scored only for candidates frozen in a plan")
        check_jail()
        if a.cmd == "reproduce":
            res = [reproduce(d) for d in dirs]
            for x in res:
                print(f"{x['candidate']}: {'identical' if x['identical'] else 'DIFFERENT'} weights ({x['cpu']}, numpy {x['numpy']})")
            if a.out:
                with open(a.out, "w") as fh:
                    json.dump(res, fh, indent=1)
            return 0 if all(x["identical"] for x in res) else 1
        plan = repro = None
        if a.test == "test":
            plan = require_plan(a.plan, dirs)
            repro = [reproduce(d) for d in dirs]
            bad = [x["candidate"] for x in repro if not x["identical"]]
            if bad:
                raise Failed(f"training does not reproduce the frozen weights here: {', '.join(bad)}")
        s = evaluate(dirs, a.test, a.rounds, a.seed, a.records)
        s["plan"], s["reproduced"] = plan, repro
        show(s)
        if a.out:
            with open(a.out, "x") as fh:
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
