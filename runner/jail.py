"""The runner's jail (queue Q10): loop.py's jail plus what Q10 found missing, without editing loop.py.

loop.py's bubblewrap mode had no seccomp filter and allowed nested user namespaces, and neither mode limited memory
or disk. Here every jailed command runs under runner/limits.c, which sets rlimits (address space, file size, open
files, CPU seconds), sets no_new_privs and, unless the command is the trusted timer (which forks and filters its
child itself), adds loop's seccomp filter. In bubblewrap mode the jail also gets --disable-userns, and --size on its
tmpfs when this bubblewrap supports them; check() records which. Confine mode (Landlock + seccomp, loop.py's fallback)
already had the filter; it gains the rlimits.

Gaps, stated: the counter's run gets the rlimits but not the filter in bubblewrap mode (count.py forks valgrind, so
the filter can't sit above it), and the filter but not the rlimits in confine mode (count.py runs outside and starts
valgrind under confine). The program counted is the one already run under the filter in the timed runs; without --size, total disk use in
bubblewrap's tmpfs is bounded only by the file-size and open-file limits and the host's memory.
"""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "loop"))
import loop as L  # noqa: E402

LIMITS = dict(as_bytes=8 << 30, fsize=1 << 30, nofile=256)   # 8 GiB address space (valgrind needs headroom), 1 GiB per file
TMPFS_BYTES = 2 << 30
STATE = {"exe": None, "dir": None, "bwrap_flags": None}


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
    if STATE["exe"]:
        return
    d = tempfile.mkdtemp(prefix="exagium-limits-")
    exe = os.path.join(d, "limits")
    r = subprocess.run(["gcc", "-O2", "-std=c11", "-I", os.path.dirname(L.SECCOMP), "-o", exe,
                        os.path.join(HERE, "limits.c")], capture_output=True, text=True, env=L.ENV)
    if r.returncode:
        raise L.Failed(f"runner/limits.c does not compile here:\n{r.stderr}")
    STATE["exe"], STATE["dir"] = exe, d


def jail(cmd, binds, timeout, cwd="/w", seccomp=True):
    """Like loop.jail, with limits.c in front of cmd. seccomp=False only for the trusted timer."""
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
        a += ["--disable-userns"]
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


PROBE = r"""
import os, resource, socket, sys
bad = []
want = {resource.RLIMIT_AS: %(as_bytes)d, resource.RLIMIT_FSIZE: %(fsize)d, resource.RLIMIT_NOFILE: %(nofile)d}
for res, v in want.items():
    if resource.getrlimit(res) != (v, v):
        bad.append(f"rlimit {res} is {resource.getrlimit(res)}")
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


def check():
    """Pick loop's jail mode, then prove here that the limits hold (fail closed): returns what the jail has."""
    L.check_jail()
    build()
    with tempfile.TemporaryDirectory(prefix="exagium-jailprobe-") as w:
        r = jail([sys.executable, "-I", "-c", PROBE], [(w, "/w", True)], 120)
        out = r.stdout.decode(errors="replace").strip()
        if r.returncode or out != "limited":
            raise L.Failed(f"the runner's jail does not hold here (exit {r.returncode}, {out!r}): "
                           f"{r.stderr.decode(errors='replace')[-500:]}")
    f = _bwrap_flags() if L.JAIL["mode"] == "bwrap" else {}
    return dict(mode=L.JAIL["mode"], limits=LIMITS, seccomp_on_candidates=True,
                disable_userns=f.get("disable_userns"), tmpfs_size=f.get("size") and TMPFS_BYTES)
