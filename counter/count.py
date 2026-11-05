#!/usr/bin/env python3
"""exagium counter: deterministic instruction and simulated cache-miss counts for one command.

Runs a command under `valgrind --tool=cachegrind` several times and records what it counted.
These are the deterministic half of meter tier 2 (timing is the other half). Every count is
labelled "simulated": cachegrind replays the program on a model CPU, it does not read hardware.

Each rep runs in a fresh sandbox at a fixed path: an empty working directory holding only the
declared --input files, an empty HOME and an empty TMPDIR, with a fixed environment.

    python3 count.py -- ./candidate args                        # program from PATH or absolute
    python3 count.py --input net.py --input w.bin -- python3 net.py
    python3 count.py --input cand -- ./cand 1000                # ./name must be an --input
    python3 count.py --check ../runs/<record>.json [--input-dir DIR]

Exit codes: 0 counted, every rep identical and no code that can switch counting off (--check:
same counts and output); 1 reps differ or such code was found (--check: counts or output differ,
or the record is edited or malformed); 2 --check could not compare like for like (another
program, input, library, loader cache, valgrind, architecture or cache model); 3 failure.
Stdlib only. Needs Linux and valgrind >= 3.22.
"""
import argparse, datetime, fcntl, threading, gzip, hashlib, json, os, platform, shutil, statistics, subprocess, sys, tempfile

SCHEMA = "exagium-count/3"
# Reference cache model. An explicit choice (see README "Reference cache"): comparable across
# hosts, not a model of any one host. Sizes in bytes, then ways, then line size.
REFERENCE = {"I1": [32768, 8, 64], "D1": [49152, 12, 64], "LL": [8388608, 16, 64]}
# Fixed strings, so the command's environment is byte-identical on every host and every run.
SANDBOX = "/tmp/exagium-count"
PATH = "/usr/local/bin:/usr/bin:/bin"
BASE_ENV = {"PATH": PATH, "HOME": SANDBOX + "/home", "TMPDIR": SANDBOX + "/tmp", "LC_ALL": "C", "LANG": "C",
            "TZ": "UTC", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
# These would change what valgrind runs (VALGRIND_LIB can swap the tool) or what the loader maps unseen.
DENY_ENV_PREFIXES = ("VALGRIND_", "LD_", "GCONV_PATH", "GLIBC_TUNABLES")
EVENTS = ["Ir", "I1mr", "ILmr", "Dr", "D1mr", "DLmr", "Dw", "D1mw", "DLmw"]
# The preamble of every valgrind client request (valgrind.h). Code holding it can switch counting
# off (CACHEGRIND_STOP_INSTRUMENTATION) or detect valgrind (RUNNING_ON_VALGRIND).
CLIENT_REQUEST_MAGIC = {
    "x86_64": bytes.fromhex("48c1c70348c1c70d48c1c73d48c1c733"),  # rol rdi 3,13,61,51
    "aarch64": bytes.fromhex("8c0dcc938c35cc938ccdcc938cf5cc93"),  # ror x12 3,13,51,61
}


class Failure(Exception):
    pass


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
    """One cachegrind.out file -> (cache model, counts). Fails unless every event and cache is there."""
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
    if events is None or summary is None or len(events) != len(summary) or set(events) != set(EVENTS):
        raise Failure(f"{os.path.basename(path)}: missing or unexpected events/summary")
    if set(desc) != {"I1", "D1", "LL"}:
        raise Failure(f"{os.path.basename(path)}: cache model not described")
    return desc, dict(zip(events, summary))


def valgrind_version(exe):
    return subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()


def tool_fingerprint(exe):
    """sha256 of the valgrind launcher, the cachegrind tool binary it loads, and this script."""
    p = subprocess.run([exe, "--command-line-only=yes", "-v", "--tool=cachegrind", "--cachegrind-out-file=/dev/null", "/bin/true"],
                       capture_output=True, text=True, env={"PATH": PATH})
    tool = next((l.split("Reading syms from ", 1)[1].strip() for l in p.stderr.splitlines()
                 if "Reading syms from " in l and "/cachegrind-" in l), None)
    if not tool or not os.path.isfile(tool):
        raise Failure("cannot locate the cachegrind tool binary")
    return {"launcher": {"path": exe, "sha256": sha256_file(exe)}, "cachegrind": {"path": tool, "sha256": sha256_file(tool)},
            "count.py": sha256_file(os.path.abspath(__file__))}


def host():
    info = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info.setdefault(k.strip(), v.strip())
    except OSError:
        pass
    model = info.get("model name") or " ".join(
        f"{k} {info[k]}" for k in ("CPU implementer", "CPU part", "CPU variant", "CPU revision") if k in info)
    ldcache = "/etc/ld.so.cache"
    return {"machine": platform.machine(), "system": platform.system(), "kernel": platform.release(),
            "libc": " ".join(platform.libc_ver()), "cpu_model": model, "cpus": os.cpu_count(),
            "ld_so_cache_sha256": sha256_file(ldcache) if os.path.isfile(ldcache) else None, "label": "reported"}


def shared_libs(path):
    """sha256 of every shared library the dynamic loader would map for path (empty if not dynamic ELF)."""
    if not shutil.which("ldd"):
        return {}
    p = subprocess.run(["ldd", path], capture_output=True, text=True)
    libs = {}
    for line in p.stdout.splitlines():
        lib = line.split("=>")[-1].split("(")[0].strip()
        if lib.startswith("/") and os.path.isfile(lib):
            libs[lib] = sha256_file(lib)
    return libs


def interpreter(path):
    """The '#!' interpreter of a script, resolved on the fixed PATH, or None."""
    with open(path, "rb") as f:
        head = f.read(256)
    if not head.startswith(b"#!"):
        return None
    words = head[2:].split(b"\n")[0].decode(errors="replace").split()
    if not words or not os.path.isabs(words[0]):
        raise Failure(f"{path}: '#!' must name an absolute interpreter")
    if os.path.basename(words[0]) == "env":
        raise Failure(f"{path}: '#!/usr/bin/env' starts the interpreter with exec, which can't be counted; "
                      f"run the interpreter directly, e.g. 'python3 {os.path.basename(path)}'")
    return words[0]


def fingerprint(cmd, inputs):
    """Hash what the command will execute: program, script interpreter, their shared libraries."""
    name = cmd[0]
    if "/" not in name:
        prog = shutil.which(name, path=PATH)
        if not prog:
            raise Failure(f"{name} not found on the fixed PATH {PATH}; give an absolute path")
    elif os.path.isabs(name):
        prog = name
    else:
        base = os.path.basename(name)
        if os.path.normpath(name) != base or base not in inputs:
            raise Failure(f"{name}: a relative program must be ./<name> of an --input file")
        prog = inputs[base]["path"]
    if not os.path.isfile(prog):
        raise Failure(f"{prog}: not a file")
    fp = {"program": {"path": name if "/" in name and not os.path.isabs(name) else prog,
                      "sha256": sha256_file(prog), "shared_libs": shared_libs(prog)}}
    interp = interpreter(prog)
    if interp:
        if not os.path.isfile(interp):
            raise Failure(f"interpreter {interp} not found")
        fp["interpreter"] = {"path": interp, "sha256": sha256_file(interp), "shared_libs": shared_libs(interp)}
    return fp, prog


def client_request_code(paths):
    magic = CLIENT_REQUEST_MAGIC.get(platform.machine())
    if magic is None:
        return None  # unknown architecture: can't tell
    found = []
    for p in sorted(set(paths)):
        try:
            with open(p, "rb") as f:
                if magic in f.read():
                    found.append(p)
        except OSError:
            found.append(p + " (unreadable)")
    return found


def cache_args(cache):
    if cache == "host":
        return []  # valgrind reads the host's CPUID; the simulated model is recorded from its output
    return [f"--{k}={','.join(map(str, cache[k]))}" for k in ("I1", "D1", "LL")]


def fresh_sandbox(inputs):
    shutil.rmtree(SANDBOX, ignore_errors=True)
    if os.path.exists(SANDBOX):
        raise Failure(f"cannot clear {SANDBOX}")
    for d in ("work", "home", "tmp"):
        os.makedirs(os.path.join(SANDBOX, d))
    for base, v in inputs.items():
        shutil.copy2(v["path"], os.path.join(SANDBOX, "work", base))


def one_rep(cmd, cache, env, exe, inputs, timeout, keep_raw=None):
    """Run cmd once under cachegrind in a fresh sandbox, children included. Counts summed over processes."""
    fresh_sandbox(inputs)
    with tempfile.TemporaryDirectory() as td:
        # One shared log stream on a pipe (valgrind keeps it on a reserved fd the program can't close). Each
        # process image valgrind starts writes one "Command:" header, so a pid with two headers exec'd. A pipe,
        # unlike a file, can't be edited afterwards: a header once written can't be removed (review 2).
        rfd, wfd = os.pipe()
        chunks = []
        reader = threading.Thread(target=lambda: chunks.extend(iter(lambda: os.read(rfd, 65536), b"")), daemon=True)
        reader.start()
        vg = [exe, "--command-line-only=yes", "--tool=cachegrind", "--cache-sim=yes", "--trace-children=yes",
              f"--cachegrind-out-file={td}/cg.%p", f"--log-fd={wfd}", *cache_args(cache), "--", *cmd]
        t0 = datetime.datetime.now(datetime.timezone.utc)
        try:
            p = subprocess.run(vg, env=env, cwd=os.path.join(SANDBOX, "work"), capture_output=True,
                               timeout=timeout, pass_fds=(wfd,))
        finally:
            os.close(wfd)
        wall = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
        reader.join(10)
        if reader.is_alive():  # someone still holds the log open: a process outlived the command
            raise Failure("a process outlived the command (its work would be counted late or not at all)")
        os.close(rfd)
        log = b"".join(chunks).decode(errors="replace")
        if p.returncode != 0:
            raise Failure(f"exit {p.returncode}: {log.strip()[-400:]}")
        # Cross-check the pipe (append-only) against the count files (which a program could edit or delete):
        # each process image writes one "Command:" header when valgrind starts it (a fork child doesn't), and
        # every process, fork children included, writes "I refs: N" when it ends.
        starts, irefs = {}, {}
        for line in log.splitlines():
            if not line.startswith("==") or line.find("==", 2) < 0:
                continue
            pid, text = line[2:line.index("==", 2)], line[line.index("==", 2) + 2:].strip()
            if text.startswith("Command: "):
                starts[pid] = starts.get(pid, 0) + 1
            elif text.startswith("I refs:"):
                if pid in irefs:
                    raise Failure(f"process {pid} reported its total twice")
                irefs[pid] = int(text.split(":", 1)[1].replace(",", ""))
        # Only the command itself may start a process image. A second header means an exec (by the command, or
        # by a fork child: valgrind drops the counts of any image before its exec, so work there would vanish).
        if sum(starts.values()) != 1:
            raise Failure(f"{sum(starts.values()) - 1} exec call(s) (processes {', '.join(sorted(starts))}): valgrind "
                          f"drops the counts of an image before it execs, so this could undercount. Only fork "
                          f"children are allowed; run programs directly, without exec, shells or subprocess spawns.")
        if set(starts) - set(irefs):
            raise Failure(f"process {', '.join(sorted(set(starts) - set(irefs)))} ended without reporting counts")
        outs = sorted(f for f in os.listdir(td) if f.startswith("cg."))
        if not outs or {f[3:] for f in outs} != set(irefs):
            raise Failure(f"{len(irefs)} processes reported totals but {len(outs)} left count files")
        if keep_raw:
            os.makedirs(keep_raw, exist_ok=True)
            for f in outs:
                dst = os.path.join(keep_raw, f"{f}.{os.getpid()}.{len(os.listdir(keep_raw))}.gz")
                with open(os.path.join(td, f), "rb") as s, gzip.open(dst, "xb") as d:
                    shutil.copyfileobj(s, d)
        total, models = dict.fromkeys(EVENTS, 0), []
        for f in outs:
            desc, counts = parse_out(os.path.join(td, f))
            if counts["Ir"] != irefs[f[3:]]:
                raise Failure(f"count file of process {f[3:]} disagrees with the total it reported")
            models.append(desc)
            for k in EVENTS:
                total[k] += counts[k]
        if any(m != models[0] for m in models):
            raise Failure("processes simulated different cache models")
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


def resolve_inputs(paths):
    inputs = {}
    for p in paths:
        base = os.path.basename(p)
        if base in inputs:
            raise Failure(f"two inputs named {base}")
        if not os.path.isfile(p):
            raise Failure(f"input {p} is not a file")
        inputs[base] = {"path": os.path.abspath(p), "sha256": sha256_file(p)}
    return inputs


def count(cmd, reps=3, cache=REFERENCE, extra_env=(), input_paths=(), timeout=3600, keep_raw=None):
    exe = shutil.which("valgrind")
    if not exe:
        raise Failure("valgrind not found")
    if not cmd:
        raise Failure("no command")
    env = dict(BASE_ENV)
    for kv in extra_env:
        k, sep, v = kv.partition("=")
        if not sep or not k.replace("_", "").isalnum() or k in ("PATH", "HOME", "TMPDIR") or k.startswith(DENY_ENV_PREFIXES):
            raise Failure(f"--env {kv}: must be KEY=VALUE, and not PATH, HOME, TMPDIR or {', '.join(DENY_ENV_PREFIXES)}*")
        env[k] = v
    inputs = resolve_inputs(input_paths)
    fp, prog = fingerprint(cmd, inputs)
    scanned = [prog, *[v["path"] for v in inputs.values()], *fp["program"]["shared_libs"]]
    if "interpreter" in fp:
        scanned += [fp["interpreter"]["path"], *fp["interpreter"]["shared_libs"]]
    with open(SANDBOX + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one counter at a time owns the fixed sandbox path
        try:
            rows = [one_rep(cmd, cache, env, exe, inputs, timeout, keep_raw) for _ in range(reps)]
        finally:
            shutil.rmtree(SANDBOX, ignore_errors=True)
    model = rows[0]["model"]
    requested = None if cache == "host" else {k: list(cache[k]) for k in ("I1", "D1", "LL")}
    if any(r["model"] != model for r in rows) or (requested and model != requested):
        raise Failure(f"valgrind simulated {model}, not the requested {requested}")
    crc = client_request_code(scanned)
    rec = {
        "schema": SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": list(cmd),
        "env": env,
        "inputs": {b: {"path": v["path"], "sha256": v["sha256"]} for b, v in inputs.items()},
        "executes": fp,
        "tool": {"valgrind": valgrind_version(exe), "files": tool_fingerprint(exe), "options": ["--tool=cachegrind", "--cache-sim=yes",
                 "--trace-children=yes", "--command-line-only=yes"], "sandbox": SANDBOX},
        "host": host(),
        "cache": {"requested": requested or "host", "simulated": model, "label": "chosen" if requested else "reported"},
        "reps": [{k: r[k] for k in ("counts", "processes", "wall_s", "stdout_sha256")} for r in rows],
        "summary": summarise(rows),
        "verdict": {"exact": all(r["counts"] == rows[0]["counts"] for r in rows),
                    "stdout_identical": len({r["stdout_sha256"] for r in rows}) == 1,
                    "client_request_code": crc, "reps": reps},
        "labels": {"counts": "simulated", "wall_s": "measured (under valgrind; not the program's native time)"},
        "caveats": [
            "Counts are for the CPU valgrind presents, not the host's: on x86 it offers at most AVX2, so code "
            "that dispatches on CPU features runs its AVX2-or-lower path here even where the host has AVX-512.",
            "The cache model is two levels (L1 and LL), LRU, no prefetching; miss counts are relative to that model, "
            "not predictions of this machine's latency.",
            "The sandbox is a fresh directory, HOME and TMPDIR, not a container: a program can still reach absolute "
            "paths elsewhere, other processes and the network. Run untrusted candidates inside a fresh container.",
            "client_request_code scans the program, interpreter, their linked libraries and inputs; code loaded "
            "later (dlopen, JIT) is not scanned.",
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


def _validate(rec):
    body = {k: v for k, v in rec.items() if k != "digest_sha256"}
    if hashlib.sha256(canonical(body)).hexdigest() != rec.get("digest_sha256"):
        return "record digest does not match its contents (edited after it was written)"
    if rec.get("schema") != SCHEMA:
        return f"schema {rec.get('schema')!r}, expected {SCHEMA}"
    cmd, env, inputs, cache = rec["command"], rec["env"], rec["inputs"], rec["cache"]["requested"]
    if not (isinstance(cmd, list) and cmd and all(isinstance(c, str) for c in cmd)):
        return "command must be a non-empty list of strings"
    if not (isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())):
        return "env must map strings to strings"
    if not (isinstance(inputs, dict) and all(isinstance(v, dict) and isinstance(v.get("sha256"), str) for v in inputs.values())):
        return "inputs must map names to {path, sha256}"
    if cache != "host" and not (isinstance(cache, dict) and set(cache) == {"I1", "D1", "LL"} and all(
            isinstance(v, list) and len(v) == 3 and all(isinstance(x, int) and x > 0 for x in v) for v in cache.values())):
        return "cache.requested must be 'host' or I1/D1/LL as [size, ways, line]"
    reps = rec["reps"]
    if not isinstance(reps, list) or not reps or len(reps) != rec["verdict"]["reps"]:
        return "rep list missing or wrong length"
    for i, r in enumerate(reps):
        if set(r["counts"]) != set(EVENTS) or not all(isinstance(x, int) and x >= 0 for x in r["counts"].values()):
            return f"rep {i}: wrong events or non-integer counts"
    if rec["verdict"]["exact"] != all(r["counts"] == reps[0]["counts"] for r in reps):
        return "verdict.exact disagrees with the recorded reps"
    if rec["verdict"]["stdout_identical"] != (len({r["stdout_sha256"] for r in reps}) == 1):
        return "verdict.stdout_identical disagrees with the recorded reps"
    if summarise(reps) != rec["summary"]:
        return "summary disagrees with the recorded reps"
    return None


def check(path, input_dir=None, reps=None):
    """Re-count a recorded command here and compare. Returns (exit code, message)."""
    try:
        with open(path) as f:
            rec = json.load(f)
        problem = _validate(rec)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        return 1, f"malformed record: {type(e).__name__}: {e}"
    if problem:
        return 1, problem
    try:
        return _recheck(rec, input_dir, reps)
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        return 1, f"malformed record: {type(e).__name__}: {e}"


def _recheck(rec, input_dir, reps):
    paths = []
    for base, v in rec["inputs"].items():
        p = os.path.join(input_dir, base) if input_dir else v["path"]
        if not os.path.isfile(p) or sha256_file(p) != v["sha256"]:
            return 2, f"input {base} is missing or differs at {p}"
        paths.append(p)
    exe = shutil.which("valgrind")
    if not exe or valgrind_version(exe) != rec["tool"]["valgrind"]:
        return 2, "valgrind missing or a different version"
    tf = tool_fingerprint(exe)
    if {k: tf[k] for k in ("launcher", "cachegrind")} != {k: rec["tool"]["files"][k] for k in ("launcher", "cachegrind")}:
        return 2, "a different valgrind launcher or cachegrind binary"
    if not rec["verdict"]["exact"] or rec["verdict"]["client_request_code"] != []:
        return 2, "the recorded run was not a clean exact count (reps differ or counting could be switched off)"
    h = host()
    if h["machine"] != rec["host"]["machine"]:
        return 2, "different architecture"
    if h["ld_so_cache_sha256"] != rec["host"]["ld_so_cache_sha256"]:
        return 2, "a different loader cache (another system image): counts may legitimately differ"
    try:
        fp, _ = fingerprint(rec["command"], resolve_inputs(paths))
    except Failure as e:
        return 2, str(e)
    if fp != rec["executes"]:
        return 2, "the program, interpreter or a shared library differs from the recorded one"
    extra = [f"{k}={v}" for k, v in rec["env"].items() if k not in ("PATH", "HOME", "TMPDIR")]
    new = count(rec["command"], reps or len(rec["reps"]), rec["cache"]["requested"], extra, paths)
    if new["cache"]["simulated"] != rec["cache"]["simulated"]:
        return 2, "a different cache model was simulated (host cache differs)"
    diffs = []
    old_r, new_r = rec["reps"][0], new["reps"][0]
    for k in EVENTS:
        if old_r["counts"][k] != new_r["counts"][k]:
            diffs.append(f"{k} {old_r['counts'][k]}->{new_r['counts'][k]}")
    for k in ("processes", "stdout_sha256"):
        if old_r[k] != new_r[k]:
            diffs.append(f"{k} differs")
    if new["verdict"]["client_request_code"] != []:
        diffs.append("code that can switch counting off is now present")
    if not diffs and new["verdict"]["exact"]:
        return 0, "identical counts and output"
    return 1, (f"differs: {', '.join(diffs) or 'reps not exact'} "
               f"(recorded exact={rec['verdict']['exact']}, now exact={new['verdict']['exact']})")


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
    ap.add_argument("--input", action="append", default=[], help="file copied into the sandbox and fingerprinted")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the command")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
    ap.add_argument("--keep-raw", metavar="DIR", help="also keep each process's cachegrind.out (gzipped), for cg_diff")
    ap.add_argument("--check", metavar="RECORD")
    ap.add_argument("--input-dir", help="with --check: find inputs here by name instead of their recorded paths")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if a.reps < 1:
        ap.error("--reps must be at least 1")
    try:
        if a.check:
            code, msg = check(a.check, a.input_dir)
            print(msg)
            return code
        rec = count(cmd, a.reps, a.cache, a.env, a.input, keep_raw=a.keep_raw)
        path = write_record(rec, a.out_dir)
    except Exception as e:  # anything unexpected is a failure, never a verdict
        print(f"count failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    s, v = rec["summary"], rec["verdict"]
    print(f"{' '.join(cmd)}: {v['reps']} reps, {'every rep identical' if v['exact'] else 'reps DIFFER'}"
          f"{'' if v['stdout_identical'] else ', stdout differs'}")
    for k in EVENTS:
        sp = s[k]["rel_spread"]
        print(f"  {k:5s} {s[k]['median']:>16,.0f}" + ("" if not sp else f"  spread {sp:.2e}"))
    crc = v["client_request_code"]
    if crc:
        print(f"  WARNING: code that can switch counting off or detect valgrind: {', '.join(crc)}")
    elif crc is None:
        print("  WARNING: could not scan for valgrind client requests on this architecture")
    print(f"wrote {path}\ndigest {rec['digest_sha256']}")
    return 0 if v["exact"] and v["stdout_identical"] and crc == [] else 1


if __name__ == "__main__":
    sys.exit(main())
