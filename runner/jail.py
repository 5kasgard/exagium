"""The runner's jail (queue Q10): loop.py's jail plus what Q10 found missing, without editing loop.py.

loop.py's bubblewrap mode had no seccomp filter and allowed nested user namespaces, and neither mode limited memory
or disk. Here every jailed command runs under runner/limits.c, which sets rlimits (address space, file size, open
files, CPU seconds), sets no_new_privs and, unless the command is the trusted timer (which forks and filters its
child itself), adds loop's seccomp filter. In bubblewrap mode the jail also gets --disable-userns, and --size on its
tmpfs when this bubblewrap supports them; check() records which. Confine mode (Landlock + seccomp, loop.py's fallback)
already had the filter; it gains the rlimits.

The counter's run: count.py forks valgrind, so the filter can't sit above it. In bubblewrap mode count.py runs under
the rlimits and starts valgrind inside loop's confine (Landlock + the filter) within the jail (counter_confine); in
confine mode count.py runs outside and starts valgrind under confine, without the rlimits (a gap, stated).
Gaps, stated: training writes to a host directory bound at /w, so its total disk use there is bounded only by the
file-size and open-file limits; and without --size (refused by check()), total disk use in
bubblewrap's tmpfs is bounded only by the file-size and open-file limits and the host's memory.
"""
import os, platform, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "loop"))
import loop as L  # noqa: E402

LIMITS = dict(as_bytes=8 << 30, fsize=1 << 30, nofile=256)   # 8 GiB address space (valgrind needs headroom), 1 GiB per file
TMPFS_BYTES = 2 << 30
STATE = {"exe": None, "dir": None, "bwrap_flags": None}

# Q55: numpy's OpenBLAS picks its kernels per CPU, so trained weights differed across hosts. Forcing one kernel type
# made them identical across three hosts, two CPU families and two Python versions (runner/results/2026-11-18-q10-q55/). Haswell needs AVX2,
# which every x86 CI host seen so far has; ARM hosts keep OpenBLAS's own choice. This sets the jail's environment for
# this process only (loop.ENV is a dict; loop.py is not edited).
if platform.machine() == "x86_64":
    L.ENV["OPENBLAS_CORETYPE"] = "Haswell"


def _limits_args(cpu_s, seccomp):
    a = ["/run/exagium/limits", "--as", str(LIMITS["as_bytes"]), "--fsize", str(LIMITS["fsize"]),
         "--nofile", str(LIMITS["nofile"]), "--cpu", str(int(cpu_s))]
    return a + (["--seccomp"] if seccomp else []) + ["--"]


def _bwrap_flags():
    if STATE["bwrap_flags"] is None:
        h = subprocess.run(["bwrap", "--help"], capture_output=True, text=True)
        text = h.stdout + h.stderr
        STATE["bwrap_flags"] = {"disable_userns": "--disable-userns" in text, "size": "--size" in text}
    return STATE["bwrap_flags"]


def build():
    """Compile limits.c and loop's confine.c into one directory, bound read-only into every jail at /run/exagium."""
    if STATE["exe"]:
        return
    d = tempfile.mkdtemp(prefix="exagium-limits-")
    for src, out, extra in ((os.path.join(HERE, "limits.c"), "limits", ["-I", os.path.dirname(L.SECCOMP)]),
                            (L.CONFINE, "confine", [])):
        r = subprocess.run(["gcc", "-O2", "-std=c11", *extra, "-o", os.path.join(d, out), src],
                           capture_output=True, text=True, env=L.ENV)
        if r.returncode:
            raise L.Failed(f"{src} does not compile here:\n{r.stderr}")
    STATE["exe"], STATE["dir"] = os.path.join(d, "limits"), d


def counter_confine(tool_dst):
    """count.py arguments that start valgrind (and so the candidate) inside loop's confine (Landlock + the filter),
    from inside the bubblewrap jail: count.py itself forks, so the filter goes one level down (review 44257135013)."""
    a = ["--confine", "/run/exagium/confine", "--confine-ro", tool_dst, "--confine-ro", "/run/exagium"]
    for flag, p in L.read_allow():
        a += ["--confine-ro" if flag == "--ro" else "--confine-ls", p]
    return a


def jail(cmd, binds, timeout, cwd="/w", seccomp=True):
    """Like loop.jail, with limits.c in front of cmd. seccomp=False only for trusted programs that fork: the timer,
    and gcc (compile_candidate), whose input is untrusted but which cannot run under the fork-refusing filter."""
    build()
    binds = list(binds) + [(STATE["dir"], "/run/exagium", False)]
    pre = _limits_args(timeout, seccomp)
    if L.JAIL["mode"] == "confine":
        # confine adds loop's filter itself (unless seccomp=False, for the timer), so limits only sets the rlimits
        return L.confined(_limits_args(timeout, False) + list(cmd), binds, timeout, cwd, seccomp)
    f = _bwrap_flags()
    a = ["bwrap"]
    if f["size"]:
        a += ["--size", str(TMPFS_BYTES), "--tmpfs", "/tmp"]
    a += ["--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
          "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin", "--proc", "/proc", "--dev", "/dev"]
    if not f["size"]:
        a += ["--tmpfs", "/tmp"]
    a += ["--unshare-all", "--uid", "65534", "--gid", "65534", "--cap-drop", "ALL", "--die-with-parent",
          "--new-session", "--clearenv"]
    if f["disable_userns"]:
        a += ["--unshare-user", "--disable-userns"]   # --unshare-all only tries the user namespace (CI run 44257075574)
    if os.path.exists("/etc/ld.so.cache"):
        a += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    for r in L.python_roots():
        a += ["--ro-bind", r, r]
    for m in L.masks():
        a += ["--ro-bind", "/dev/null", m]
    for src, dst, rw in binds:
        a += ["--bind" if rw else "--ro-bind", src, dst]
    for k, v in L.ENV.items():
        a += ["--setenv", k, v]
    a += ["--chdir", cwd, "--", *pre, *cmd]
    return subprocess.run(a, capture_output=True, timeout=timeout, preexec_fn=L._fresh_keyring)


def compile_candidate(src, out, timeout=300):
    """Queue Q56: compile a candidate's infer.c inside the jail, never on the host. Only the source is copied in
    (/w, the one writable place); gcc sees /usr, the Python roots and the jail's own files, so an #include of a host
    path (a key file, the repo, the runner's home) fails to compile instead of copying the file into the binary and so
    into the output. gcc forks (cc1, as, ld), so it runs with the rlimits and no_new_privs but without the seccomp
    filter, as the timer does; the namespaces (bwrap) or Landlock (confine) still bound what it can read and write.
    Gap, stated: a compiler bug triggered by hostile source would run with that reach, not the filtered one."""
    build()
    with tempfile.TemporaryDirectory(prefix="exagium-cc-") as w:
        shutil.copy2(src, os.path.join(w, "infer.c"))
        r = jail(["gcc", *L.CFLAGS, "-pipe", "-o", "/w/infer", "/w/infer.c", "-lm"], [(w, "/w", True)], timeout,
                 seccomp=False)
        if r.returncode or not os.path.exists(os.path.join(w, "infer")):
            raise L.Failed(f"{src} does not compile in the jail:\n{r.stderr.decode(errors='replace')[-2000:]}")
        shutil.copy2(os.path.join(w, "infer"), out)


INCLUDE_OK = "#include <stdio.h>\nint main(void){puts(\"ok\");return 0;}\n"
INCLUDE_HOST = "#include \"%s\"\nint main(void){return 0;}\n"


def check_compile():
    """Fail closed unless the jailed compiler builds a plain program and refuses to #include a host file."""
    with tempfile.TemporaryDirectory(prefix="exagium-ccprobe-") as d:
        secret = os.path.join(d, "host-secret.h")
        open(secret, "w").write("int host_secret = 42;\n")
        ok, bad = os.path.join(d, "ok.c"), os.path.join(d, "bad.c")
        open(ok, "w").write(INCLUDE_OK)
        open(bad, "w").write(INCLUDE_HOST % secret)
        compile_candidate(ok, os.path.join(d, "ok"))
        try:
            compile_candidate(bad, os.path.join(d, "bad"))
        except L.Failed:
            return True
        raise L.Failed("the jailed compiler read a host file through #include (Q56 probe)")


PROBE = r"""
import os, resource, socket, sys
bad = []
want = {resource.RLIMIT_AS: %(as_bytes)d, resource.RLIMIT_FSIZE: %(fsize)d, resource.RLIMIT_NOFILE: %(nofile)d,
        resource.RLIMIT_CPU: 120}
for res, v in want.items():
    if resource.getrlimit(res) != (v, v):
        bad.append(f"rlimit {res} is {resource.getrlimit(res)}")
    try:   # a candidate must not be able to raise a limit
        resource.setrlimit(res, (v + 1, v + 1)); bad.append(f"raised rlimit {res}")
    except (OSError, ValueError):
        pass
import ctypes
if ctypes.CDLL(None, use_errno=True).prctl(39, 0, 0, 0, 0) != 1:   # PR_GET_NO_NEW_PRIVS
    bad.append("no_new_privs not set")
for name, f in (("fork", os.fork), ("socket", socket.socket), ("userns", lambda: os.unshare(os.CLONE_NEWUSER))):
    try:
        r = f()
        if name == "fork" and r == 0:
            os._exit(0)
        bad.append(name)
    except OSError:
        pass
try:
    bytearray(9 << 30); bad.append("9 GiB allocated")
except MemoryError:
    pass
try:
    with open("big", "wb") as fh:
        fh.write(b"x" * ((1 << 30) + 4096))
    bad.append("file over 1 GiB")
except OSError:
    pass
print("bad:" + ",".join(bad) if bad else "limited")
""" % LIMITS


COUNTER_PROBE = r"""
import os, socket
bad = []
for name, f in (("fork", os.fork), ("socket", socket.socket)):
    try:
        r = f()
        if name == "fork" and r == 0:
            os._exit(0)
        bad.append(name)
    except OSError:
        pass
print("bad:" + ",".join(bad) if bad else "filtered")
"""


def check():
    """Pick loop's jail mode, then prove here that the limits hold (fail closed): returns what the jail has."""
    L.check_jail()
    build()
    with tempfile.TemporaryDirectory(prefix="exagium-jailprobe-") as w:
        r = jail([sys.executable, "-I", "-c", PROBE], [(w, "/w", True)], 120)   # 120 s: the RLIMIT_CPU read back
        out = r.stdout.decode(errors="replace").strip()
        if r.returncode or out != "limited":
            raise L.Failed(f"the runner's jail does not hold here (exit {r.returncode}, {out!r}): "
                           f"{r.stderr.decode(errors='replace')[-500:]}")
    if L.JAIL["mode"] == "bwrap":
        f = _bwrap_flags()
        if not (f["disable_userns"] and f["size"]):
            raise L.Failed(f"this bubblewrap lacks --disable-userns or --size ({f}); the runner's jail needs both")
        # --disable-userns on its own: a command without the seccomp filter must still be refused a user namespace
        r = jail([sys.executable, "-I", "-c", "import os\ntry:\n os.unshare(os.CLONE_NEWUSER); print('allowed')\n"
                  "except OSError:\n print('refused')"], [], 60, cwd="/", seccomp=False)
        if r.stdout.decode().strip() != "refused":
            raise L.Failed(f"a new user namespace is not refused without the filter: {r.stdout!r} {r.stderr[-300:]!r}")
        # the counter path: confine inside bubblewrap must start and must refuse fork and sockets
        ro = [x for flag, p in L.read_allow() for x in (("--ro", p) if flag == "--ro" else ("--ls", p))]
        with tempfile.TemporaryDirectory(prefix="exagium-jailprobe2-") as w:
            r = jail(["/run/exagium/confine", *ro, "--ro", "/run/exagium", "--rw", "/w", "--rw", "/dev/null", "--",
                      sys.executable, "-I", "-c", COUNTER_PROBE], [(w, "/w", True)], 120, seccomp=False)
            out = r.stdout.decode(errors="replace").strip()
            if r.returncode or out != "filtered":
                raise L.Failed(f"confine inside the jail (the counter path) does not hold here (exit {r.returncode}, "
                               f"{out!r}): {r.stderr.decode(errors='replace')[-500:]}")
    check_compile()   # Q56: candidate C is compiled in the jail, and a host #include must fail
    f = _bwrap_flags() if L.JAIL["mode"] == "bwrap" else {}
    return dict(mode=L.JAIL["mode"], limits=LIMITS, seccomp_on_candidates=True, counter_filtered=True, compile_jailed=True,
                disable_userns=f.get("disable_userns"), tmpfs_size=f.get("size") and TMPFS_BYTES)
