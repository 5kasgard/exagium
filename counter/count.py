#!/usr/bin/env python3
"""exagium counter: deterministic instruction and simulated cache-miss counts for one command.

Runs a command under `valgrind --tool=cachegrind` several times and records what it counted.
These are the deterministic half of meter tier 2 (timing is the other half). Every count is
labelled "simulated": cachegrind replays the program on a model CPU, it does not read hardware.

    python3 count.py -- ./candidate args            # 3 reps, reference cache, record in ../runs/
    python3 count.py --reps 5 --cache host -- python3 net.py
    python3 count.py --check ../runs/<record>.json   # re-count on this machine and compare

Exit codes: 0 counted, every rep identical (or --check: same counts); 1 reps differ (or --check:
counts differ, or the record was edited); 2 --check could not compare like for like (another
valgrind, cache model or architecture); 3 the command or valgrind failed.
Stdlib only. Needs valgrind >= 3.22 (cachegrind with --cache-sim).
"""
import argparse, datetime, gzip, hashlib, json, os, platform, shutil, statistics, subprocess, sys, tempfile

SCHEMA = "exagium-count/1"
# Reference cache model. An explicit choice (see README "Reference cache"): comparable across
# hosts, not a model of any one host. Sizes in bytes, then ways, then line size.
REFERENCE = {"I1": [32768, 8, 64], "D1": [49152, 12, 64], "LL": [8388608, 16, 64]}
# The command sees this environment and nothing else, so counts don't drift with the caller's.
BASE_ENV = {"LC_ALL": "C", "LANG": "C", "TZ": "UTC", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
EVENTS = ["Ir", "I1mr", "ILmr", "Dr", "D1mr", "DLmr", "Dw", "D1mw", "DLmw"]


def canonical(obj):
    """One byte string per value; fields are never joined by a separator."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_out(path):
    """Read one cachegrind.out file: the simulated cache model and the summary counts."""
    desc, events, summary = {}, None, None
    with open(path) as f:
        for line in f:
            if line.startswith("desc: ") and " cache:" in line:
                name, rest = line[6:].split(" cache:", 1)
                b, ln, ways = [p.strip() for p in rest.split(",")]
                desc[name.strip()] = [int(b.split()[0]), int(ways.split("-")[0]), int(ln.split()[0])]
            elif line.startswith("events:"):
                events = line.split()[1:]
            elif line.startswith("summary:"):
                summary = [int(x) for x in line.split()[1:]]
    if events is None or summary is None or len(events) != len(summary):
        raise ValueError(f"{path}: no events/summary line")
    return desc, dict(zip(events, summary))


def shared_libs(prog):
    """sha256 of every shared library the dynamic loader would map for prog (empty if not dynamic ELF)."""
    if not os.path.isfile(prog) or not shutil.which("ldd"):
        return {}
    p = subprocess.run(["ldd", prog], capture_output=True, text=True)
    libs = {}
    for line in p.stdout.splitlines():
        parts = line.split("=>")[-1].split("(")[0].strip()
        if parts.startswith("/") and os.path.isfile(parts):
            libs[parts] = sha256_file(parts)
    return libs


def valgrind_version(exe):
    return subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()


def host():
    model = ""
    try:
        with open("/proc/cpuinfo") as f:
            info = {}
            for l in f:
                k, _, v = l.partition(":")
                info.setdefault(k.strip(), v.strip())
        model = info.get("model name") or " ".join(
            f"{k} {info[k]}" for k in ("CPU implementer", "CPU part", "CPU variant", "CPU revision") if k in info)
    except OSError:
        pass
    return {"machine": platform.machine(), "system": platform.system(), "kernel": platform.release(),
            "cpu_model": model, "cpus": os.cpu_count(), "label": "reported"}


def cache_args(cache):
    if cache == "host":
        return []  # valgrind reads the host's CPUID; the simulated model is recorded from its output
    return [f"--{k}={','.join(map(str, cache[k]))}" for k in ("I1", "D1", "LL")]


def one_rep(cmd, cache, env, exe, timeout, keep_raw=None):
    """Run cmd once under cachegrind, children included. Returns counts summed over every process."""
    with tempfile.TemporaryDirectory() as td:
        vg = [exe, "--command-line-only=yes", "--tool=cachegrind", "--cache-sim=yes",
              "--trace-children=yes", f"--cachegrind-out-file={td}/cg.%p", *cache_args(cache), "--", *cmd]
        t0 = datetime.datetime.now(datetime.timezone.utc)
        p = subprocess.run(vg, env=env, capture_output=True, timeout=timeout)
        wall = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
        outs = sorted(f for f in os.listdir(td) if f.startswith("cg."))
        if p.returncode != 0 or not outs:
            tail = p.stderr.decode(errors="replace").strip()[-400:]
            raise RuntimeError(f"exit {p.returncode}, {len(outs)} count files: {tail}")
        if keep_raw:
            os.makedirs(keep_raw, exist_ok=True)
            for f in outs:
                with open(os.path.join(td, f), "rb") as src, gzip.open(os.path.join(keep_raw, f"{f}.{os.getpid()}.{len(os.listdir(keep_raw))}.gz"), "xb") as dst:
                    shutil.copyfileobj(src, dst)
        total, models = dict.fromkeys(EVENTS, 0), []
        for f in outs:
            desc, counts = parse_out(os.path.join(td, f))
            models.append(desc)
            for k in EVENTS:
                total[k] += counts.get(k, 0)
        if any(m != models[0] for m in models):
            raise RuntimeError("processes simulated different cache models")
        return {"counts": total, "processes": len(outs), "wall_s": round(wall, 3),
                "stdout_sha256": hashlib.sha256(p.stdout).hexdigest(), "model": models[0]}


def summarise(reps):
    out = {}
    for k in EVENTS:
        v = [r["counts"][k] for r in reps]
        lo, hi = min(v), max(v)
        out[k] = {"median": statistics.median(v), "min": lo, "max": hi,
                  "rel_spread": (hi - lo) / lo if lo else (0.0 if hi == 0 else None)}
    return out


def count(cmd, reps=3, cache=REFERENCE, extra_env=(), hash_files=(), timeout=3600, keep_raw=None):
    exe = shutil.which("valgrind")
    if not exe:
        raise RuntimeError("valgrind not found")
    if not cmd:
        raise ValueError("no command")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/"), **BASE_ENV}
    for kv in extra_env:
        k, _, v = kv.partition("=")
        env[k] = v
    prog = shutil.which(cmd[0]) or cmd[0]
    rows = [one_rep(cmd, cache, env, exe, timeout, keep_raw) for _ in range(reps)]
    model = rows[0]["model"]
    if any(r["model"] != model for r in rows):
        raise RuntimeError("cache model changed between reps")
    requested = None if cache == "host" else {k: list(cache[k]) for k in ("I1", "D1", "LL")}
    exact = all(r["counts"] == rows[0]["counts"] for r in rows)
    stdout_same = len({r["stdout_sha256"] for r in rows}) == 1
    rec = {
        "schema": SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": list(cmd),
        "cwd": os.getcwd(),
        "env": env,
        "inputs": {"program": {"path": prog, "sha256": sha256_file(prog) if os.path.isfile(prog) else None,
                               "shared_libs": shared_libs(prog)},
                   **{f: {"sha256": sha256_file(f)} for f in hash_files}},
        "tool": {"valgrind": valgrind_version(exe), "options": ["--tool=cachegrind", "--cache-sim=yes",
                 "--trace-children=yes", "--command-line-only=yes"]},
        "host": host(),
        "cache": {"requested": requested if requested else "host", "simulated": model,
                  "matches_request": requested is None or model == requested, "label": "chosen" if requested else "reported"},
        "reps": [{k: r[k] for k in ("counts", "processes", "wall_s", "stdout_sha256")} for r in rows],
        "summary": summarise(rows),
        "verdict": {"exact": exact, "stdout_identical": stdout_same, "reps": reps},
        "labels": {"counts": "simulated", "wall_s": "measured (under valgrind; not the program's native time)"},
        "caveats": [
            "Counts are for the CPU valgrind presents, not the host's: on x86 it offers at most AVX2, so code "
            "that dispatches on CPU features runs its AVX2-or-lower path here even where the host has AVX-512.",
            "The cache model is two levels (L1 and LL), LRU, no prefetching; miss counts are relative to that model, "
            "not predictions of this machine's latency.",
            "Exact repeatability holds on one host; across hosts it is a hypothesis being tested (README).",
        ],
    }
    rec["digest_sha256"] = hashlib.sha256(canonical(rec)).hexdigest()
    return rec


def write_record(rec, out_dir):
    """One file per run, never overwritten."""
    os.makedirs(out_dir, exist_ok=True)
    name = f"count-{rec['created_utc'].replace(':', '')}-{rec['digest_sha256'][:12]}.json"
    path = os.path.join(out_dir, name)
    with open(path, "x") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    return path


def check(path, reps=None):
    """Re-count a recorded command here and compare. Returns (exit code, message)."""
    with open(path) as f:
        rec = json.load(f)
    body = {k: v for k, v in rec.items() if k != "digest_sha256"}
    if hashlib.sha256(canonical(body)).hexdigest() != rec.get("digest_sha256"):
        return 1, "record digest does not match its contents (edited after it was written)"
    for i, r in enumerate(rec["reps"]):
        if set(r["counts"]) != set(EVENTS):
            return 1, f"rep {i}: wrong event set"
    if rec["verdict"]["exact"] != all(r["counts"] == rec["reps"][0]["counts"] for r in rec["reps"]):
        return 1, "verdict.exact disagrees with the recorded reps"
    if summarise(rec["reps"]) != rec["summary"]:
        return 1, "summary disagrees with the recorded reps"
    cwd = os.getcwd()
    try:
        os.chdir(rec["cwd"])
        prog = rec["inputs"]["program"]
        if prog["sha256"] and (not os.path.isfile(prog["path"]) or sha256_file(prog["path"]) != prog["sha256"]):
            return 2, f"program {prog['path']} is missing or differs from the recorded one"
        for f, v in rec["inputs"].items():
            if f != "program" and (not os.path.isfile(f) or sha256_file(f) != v["sha256"]):
                return 2, f"input {f} is missing or differs"
        exe = shutil.which("valgrind")
        if not exe or valgrind_version(exe) != rec["tool"]["valgrind"]:
            return 2, "valgrind missing or a different version"
        if platform.machine() != rec["host"]["machine"]:
            return 2, "different architecture"
        cache = rec["cache"]["requested"]
        extra = [f"{k}={v}" for k, v in rec["env"].items() if k not in ("PATH", "HOME")]
        new = count(rec["command"], reps or len(rec["reps"]), cache, extra)
    except (OSError, KeyError):
        return 2, f"cannot reproduce the recorded setting here (cwd {rec.get('cwd')})"
    finally:
        os.chdir(cwd)
    if new["cache"]["simulated"] != rec["cache"]["simulated"]:
        return 2, "a different cache model was simulated (host cache differs)"
    old_c, new_c = rec["reps"][0]["counts"], new["reps"][0]["counts"]
    if rec["verdict"]["exact"] and new["verdict"]["exact"] and old_c == new_c:
        return 0, "identical counts"
    diff = {k: [old_c[k], new_c[k]] for k in EVENTS if old_c[k] != new_c[k]}
    return 1, f"counts differ: {json.dumps(diff)} (recorded exact={rec['verdict']['exact']}, now exact={new['verdict']['exact']})"


def parse_cache(s):
    if s in ("reference", "host"):
        return REFERENCE if s == "reference" else "host"
    cfg = {}
    for part in s.split(";"):
        k, _, v = part.partition("=")
        cfg[k.strip()] = [int(x) for x in v.split(",")]
    if set(cfg) != {"I1", "D1", "LL"} or any(len(v) != 3 for v in cfg.values()):
        raise argparse.ArgumentTypeError("give all three: 'I1=size,ways,line;D1=...;LL=...'")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cache", type=parse_cache, default=REFERENCE,
                    help="'reference' (default), 'host' (valgrind reads CPUID), or 'I1=s,w,l;D1=s,w,l;LL=s,w,l'")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the command")
    ap.add_argument("--hash", action="append", default=[], help="input file to fingerprint in the record")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
    ap.add_argument("--keep-raw", metavar="DIR", help="also keep each process's cachegrind.out (gzipped), for cg_diff")
    ap.add_argument("--check", metavar="RECORD")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    try:
        if a.check:
            code, msg = check(a.check)
            print(msg)
            return code
        if a.reps < 1:
            ap.error("--reps must be at least 1")
        rec = count(cmd, a.reps, a.cache, a.env, a.hash, keep_raw=a.keep_raw)
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as e:
        print(f"count failed: {e}", file=sys.stderr)
        return 3
    path = write_record(rec, a.out_dir)
    s, v = rec["summary"], rec["verdict"]
    print(f"{' '.join(cmd)}: {v['reps']} reps, {'every rep identical' if v['exact'] else 'reps DIFFER'}"
          f"{'' if v['stdout_identical'] else ', stdout differs'}")
    for k in EVENTS:
        sp = s[k]["rel_spread"]
        print(f"  {k:5s} {s[k]['median']:>16,.0f}" + ("" if not sp else f"  spread {sp:.2e}"))
    if not rec["cache"]["matches_request"]:
        print("  WARNING: valgrind simulated a different cache model than requested")
    print(f"wrote {path}\ndigest {rec['digest_sha256']}")
    return 0 if v["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
