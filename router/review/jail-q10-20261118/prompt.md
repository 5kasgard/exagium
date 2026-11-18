Adversarial security review of a jail hardening change. No praise. For each problem: file and function, the concrete attack (what the untrusted candidate code does), whether it succeeds against THIS code (trace it), and a fix. Grade critical / major / minor. Only report what you can trace through the code shown; say "not traceable" otherwise. End with: ship / fix then ship / do not ship.

Context: candidates are untrusted Python (train.py) and C (infer.c) programs. They run inside bubblewrap (bwrap mode, used on CI) or, where bwrap can't make namespaces, inside "confine" (Landlock + a seccomp filter, loop/confine.c). The previous jail (loop.py, frozen, not shown) built the same bwrap arguments but had NO seccomp filter and NO resource limits in bwrap mode. This change adds runner/limits.c in front of every jailed command and builds the bwrap arguments in runner/jail.py. loop/seccomp_filter.h (shown) is the filter: no fork/clone/clone3, sockets, keyring, ptrace, namespaces, mounts, io_uring, bpf etc.

Questions to answer specifically:
1. Can the candidate run before the filter and limits apply, or raise the limits afterwards?
2. Timed runs: the trusted timer (loop/timer.c, not shown; it forks the candidate and, given --seccomp, installs the same filter in the child before exec) runs under limits.c WITHOUT --seccomp. Is anything lost?
3. The counter run (count.py forks valgrind) gets rlimits but no filter in bwrap mode. What can a candidate do there that it couldn't in the timed runs?
4. Is the jail probe in jail.py (check()) able to pass while a protection is missing?
5. Are the limits (8 GiB address space, 1 GiB per file, 256 open files, CPU seconds = timeout) adequate on a 16 GB CI runner? What resource is still unbounded?

=== runner/jail.py
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
=== runner/limits.c
// Resource limits for the runner's jail (queue Q10), run inside the jail just before the candidate's command:
//   limits [--seccomp] [--as BYTES] [--fsize BYTES] [--nofile N] [--cpu SECONDS] -- command args...
// Sets each limit (soft = hard, so the command cannot raise it), sets PR_SET_NO_NEW_PRIVS, adds loop's seccomp filter
// (loop/seccomp_filter.h: no new processes or threads, sockets, namespaces, keyring, ptrace ...) when --seccomp is
// given, then execs the command. Without --seccomp it is for the trusted timer, which forks and adds the filter in
// its child. Exit 125 if a limit or the filter cannot be set (fail closed); else the command's own status.
// Compile: gcc -O2 -std=c11 -I loop limits.c -o limits
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <unistd.h>
#include "seccomp_filter.h"

static int set(int res, const char *v) {
  char *end;
  unsigned long long x = strtoull(v, &end, 10);
  if (!*v || *end) return -1;
  struct rlimit r = {(rlim_t)x, (rlim_t)x};
  return setrlimit(res, &r);
}

int main(int argc, char **argv) {
  int i = 1, seccomp = 0;
  for (; i < argc && strcmp(argv[i], "--"); i++) {
    int r = 0;
    if (!strcmp(argv[i], "--seccomp")) { seccomp = 1; continue; }
    if (i + 1 >= argc) break;
    if (!strcmp(argv[i], "--as")) r = set(RLIMIT_AS, argv[++i]);
    else if (!strcmp(argv[i], "--fsize")) r = set(RLIMIT_FSIZE, argv[++i]);
    else if (!strcmp(argv[i], "--nofile")) r = set(RLIMIT_NOFILE, argv[++i]);
    else if (!strcmp(argv[i], "--cpu")) r = set(RLIMIT_CPU, argv[++i]);
    else { fprintf(stderr, "limits: unknown option %s\n", argv[i]); return 125; }
    if (r) { perror("limits: setrlimit"); return 125; }
  }
  if (i >= argc - 1) { fprintf(stderr, "usage: limits [--seccomp] [--as B] [--fsize B] [--nofile N] [--cpu S] -- cmd args...\n"); return 125; }
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) { perror("limits: no_new_privs"); return 125; }
  if (seccomp && exagium_seccomp((uint32_t)getpid())) { perror("limits: seccomp"); return 125; }
  execvp(argv[i + 1], argv + i + 1);
  perror("limits: exec");
  return 125;
}
=== loop/seccomp_filter.h
// The seccomp half of the jail's fallback (see confine.c), shared by confine.c and timer.c:
// no new processes or threads (fork, vfork, clone, clone3), no sockets, no keyring, no ptrace, no process_vm_*,
// no pidfd, no namespaces or mounts, no io_uring, bpf, perf_event_open or userfaultfd, and kill/tkill/tgkill only
// to the process itself. Also no channel that outlives the run or crosses Landlock, which governs file contents but
// not these: SysV shared memory, semaphores and message queues and POSIX message queues (the host's IPC namespace,
// unlike bwrap's private one), and file metadata (chmod, chown, timestamps, extended attributes), which a candidate
// with the harness's uid could otherwise set on the harness's own files as a cache (system review 2026-11-11; the
// SysV hole found independently by an outside reviewer). Nor may it change another process's rlimits, CPU affinity,
// priority or memory placement: the harness shares our uid and pid namespace, so a nice value set on loop.py would
// slow every later timed run and an rlimit could carry state between runs (found by a separate fact-check agent). Call after PR_SET_NO_NEW_PRIVS; returns 0 on success.
#ifndef EXAGIUM_SECCOMP_FILTER_H
#define EXAGIUM_SECCOMP_FILTER_H
#include <errno.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#ifndef __NR_clone3
#define __NR_clone3 435
#endif
#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif
#ifndef __NR_pidfd_send_signal
#define __NR_pidfd_send_signal 424
#endif
#ifndef __NR_fchmodat2
#define __NR_fchmodat2 452
#endif
#ifndef __NR_setxattrat
#define __NR_setxattrat 463
#endif
#ifndef __NR_removexattrat
#define __NR_removexattrat 466
#endif

#define LD_NR BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr))
#define DENY(nr, err) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 1), BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (err))
// signals: allowed only when args[0] (low 32 bits; the arches here are little-endian) is our own pid
#define SELF_ONLY(nr, pid) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 4), \
  BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[0])), \
  BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (pid), 1, 0), \
  BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM), \
  BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)
// resource and scheduling settings: allowed on our own pid or 0 (= ourselves), never on another process
#define SELF_OR_ZERO(nr, pid) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 5), \
  BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[0])), \
  BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (pid), 2, 0), \
  BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, 0), \
  BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM), \
  BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)

#if defined(__x86_64__)
#define ARCH AUDIT_ARCH_X86_64
#elif defined(__aarch64__)
#define ARCH AUDIT_ARCH_AARCH64
#else
#error "confine: only x86-64 and arm64 are supported"
#endif

static int exagium_seccomp(uint32_t pid) {
  struct sock_filter f[] = {
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, ARCH, 1, 0),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    LD_NR,
#if defined(__x86_64__)
    BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, 0x40000000, 0, 1),  // x32 syscalls
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    DENY(__NR_fork, EPERM), DENY(__NR_vfork, EPERM),
    DENY(__NR_chmod, EPERM), DENY(__NR_chown, EPERM), DENY(__NR_lchown, EPERM),
    DENY(__NR_utime, EPERM), DENY(__NR_utimes, EPERM), DENY(__NR_futimesat, EPERM),
#endif
    DENY(__NR_clone, EPERM), DENY(__NR_clone3, ENOSYS),
    DENY(__NR_socket, EPERM), DENY(__NR_socketpair, EPERM),
    DENY(__NR_keyctl, EPERM), DENY(__NR_add_key, EPERM), DENY(__NR_request_key, EPERM),
    DENY(__NR_ptrace, EPERM), DENY(__NR_process_vm_readv, EPERM), DENY(__NR_process_vm_writev, EPERM),
    DENY(__NR_pidfd_open, EPERM), DENY(__NR_pidfd_getfd, EPERM),
    DENY(__NR_unshare, EPERM), DENY(__NR_setns, EPERM), DENY(__NR_mount, EPERM),
    DENY(__NR_io_uring_setup, EPERM), DENY(__NR_bpf, EPERM), DENY(__NR_perf_event_open, EPERM),
    DENY(__NR_userfaultfd, EPERM), DENY(__NR_pidfd_send_signal, EPERM),
    DENY(__NR_shmget, EPERM), DENY(__NR_shmat, EPERM), DENY(__NR_shmctl, EPERM),
    DENY(__NR_semget, EPERM), DENY(__NR_semop, EPERM), DENY(__NR_semctl, EPERM), DENY(__NR_semtimedop, EPERM),
    DENY(__NR_msgget, EPERM), DENY(__NR_msgsnd, EPERM), DENY(__NR_msgrcv, EPERM), DENY(__NR_msgctl, EPERM),
    DENY(__NR_mq_open, EPERM), DENY(__NR_mq_unlink, EPERM), DENY(__NR_mq_timedsend, EPERM),
    DENY(__NR_mq_timedreceive, EPERM), DENY(__NR_mq_notify, EPERM), DENY(__NR_mq_getsetattr, EPERM),
    DENY(__NR_fchmod, EPERM), DENY(__NR_fchmodat, EPERM), DENY(__NR_fchmodat2, EPERM),
    DENY(__NR_fchown, EPERM), DENY(__NR_fchownat, EPERM), DENY(__NR_utimensat, EPERM),
    DENY(__NR_setxattr, EPERM), DENY(__NR_lsetxattr, EPERM), DENY(__NR_fsetxattr, EPERM), DENY(__NR_setxattrat, EPERM),
    DENY(__NR_removexattr, EPERM), DENY(__NR_lremovexattr, EPERM), DENY(__NR_fremovexattr, EPERM),
    DENY(__NR_removexattrat, EPERM),
    DENY(__NR_setpriority, EPERM), DENY(__NR_ioprio_set, EPERM), DENY(__NR_sched_setscheduler, EPERM),
    DENY(__NR_sched_setparam, EPERM), DENY(__NR_sched_setattr, EPERM), DENY(__NR_migrate_pages, EPERM),
    DENY(__NR_move_pages, EPERM), DENY(__NR_process_madvise, EPERM), DENY(__NR_process_mrelease, EPERM),
    SELF_OR_ZERO(__NR_prlimit64, pid),
    LD_NR, SELF_OR_ZERO(__NR_sched_setaffinity, pid),
    LD_NR, SELF_ONLY(__NR_kill, pid),
    LD_NR, SELF_ONLY(__NR_tkill, pid),
    LD_NR, SELF_ONLY(__NR_tgkill, pid),
    LD_NR, DENY(__NR_rt_sigqueueinfo, EPERM), DENY(__NR_rt_tgsigqueueinfo, EPERM),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  struct sock_fprog prog = {sizeof f / sizeof f[0], f};
  return prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog);
}

#endif
=== runner/run.py (jail-related functions)
def train(c, t):
    """Jailed training; returns the weights' bytes."""
    with tempfile.TemporaryDirectory(prefix="exagium-rtrain-") as sb:
        shutil.copyfile(os.path.join(c["dir"], "train.py"), os.path.join(sb, "train.py"))
        shutil.copyfile(t["card_path"], os.path.join(sb, "card.json"))
        for f in ("train.in", "train.key"):
            shutil.copyfile(os.path.join(t["dir"], f), os.path.join(sb, f))
        r = J.jail([sys.executable, "-s", "-P", "train.py", "card.json", "train.in", "train.key", "weights.bin"],
                   [(sb, "/w", True)], L.TRAIN_TIMEOUT)
        if r.returncode:
            raise Failed(f"{c['name']} on {t['name']}: train.py exited {r.returncode}:\n{r.stderr.decode(errors='replace')[-2000:]}")
        w = os.path.join(sb, "weights.bin")
        if os.path.islink(w) or not os.path.isfile(w):
            raise Failed(f"{c['name']} on {t['name']}: train.py wrote no weights.bin")
        return open(w, "rb").read()


def timed(c, inp, n, timer, cpu):
    """One jailed run under loop's timer: (output bytes, ns)."""
    with tempfile.TemporaryDirectory(prefix="exagium-rrun-") as w:
        shutil.copy2(c["exe"], os.path.join(w, "infer"))
        shutil.copy2(c["weights_path"], os.path.join(w, "weights.bin"))
        shutil.copy2(timer, os.path.join(w, "timer"))
        shutil.copy2(inp, os.path.join(w, "in.rec"))
        # the timer forks, so it runs without the filter and adds it to its child, in both jail modes (Q10)
        r = J.jail(["./timer", "--seccomp", str(cpu), "./infer", "weights.bin", "in.rec", str(n)], [(w, "/w", False)],
                   1800, seccomp=False)
    out, sep, tail = r.stdout.rpartition(b"\nexagium-timer ")
    f = tail.split()
    if r.returncode or not sep or len(f) != 4 or not tail.endswith(b"\n") or not all(x.isdigit() for x in f[:3]) \
            or not (f[3].isdigit() or f[3] == b"-1"):
        raise Failed(f"{c['name']}: timer failed (exit {r.returncode}): {r.stderr.decode(errors='replace')[-300:]}")
    ns, status, cpu_ns, pids = (int(x) for x in f)
    if status:
        raise Failed(f"{c['name']}: infer exited {status}")
    if pids != -1:   # -1: the child ran under the seccomp filter, which refuses every new process or thread
        raise Failed(f"{c['name']}: the timer did not filter its child (pids {pids})")
    if cpu_ns > L.CPU_SLACK[0] * ns + L.CPU_SLACK[1]:
        raise Failed(f"{c['name']}: used more than one CPU ({cpu_ns} ns CPU in {ns} ns wall)")
    return out, ns


def count(c, inp, n, outdir):
    """Counter record for one jailed run (loop.count's pattern with record files): returns Ir."""
    os.makedirs(outdir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exagium-rcnt-") as tool:
        for src, name in ((c["exe"], "infer"), (c["weights_path"], "weights.bin"), (L.COUNT, "count.py"), (inp, "in.rec")):
            shutil.copy2(src, os.path.join(tool, name))
        tail = ["--reps", "2", "--input", "/tool/infer", "--input", "/tool/weights.bin", "--input", "/tool/in.rec", "--",
                "./infer", "weights.bin", "in.rec", str(n)]
        if L.JAIL["mode"] == "confine":
            ro = [x for flag, p in L.read_allow() for x in (("--confine-ro", p) if flag == "--ro" else ("--confine-ls", p))]
            with tempfile.TemporaryDirectory(prefix="exagium-rcnt-cwd-") as cwd:
                r = subprocess.run([sys.executable, "-I", os.path.join(tool, "count.py"), "--stdout-record",
                                    "--confine", L.JAIL["confine"], *ro, *[x.replace("/tool", tool, 1) for x in tail]],
                                   capture_output=True, timeout=3600, cwd=cwd, env=dict(L.ENV, HOME=cwd, TMPDIR=cwd))
        else:
            # count.py forks valgrind, so no filter here: rlimits only (runner/jail.py, "Gaps")
            r = J.jail([sys.executable, "-I", "/tool/count.py", "--stdout-record", "--single-process", *tail],
                       [(tool, "/tool", False)], 3600, cwd="/tmp", seccomp=False)
    if r.returncode not in (0, 1) or r.stdout.count(b"\n") != 1:
        raise Failed(f"{c['name']}: counter failed (exit {r.returncode}):\n{r.stderr.decode(errors='replace')[-1000:]}")
    rec = json.loads(r.stdout)
    body = {k: v for k, v in rec.items() if k != "digest_sha256"}
    if L.sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()) != rec["digest_sha256"]:
        raise Failed(f"{c['name']}: counter record digest does not match")
    with open(os.path.join(outdir, f"{c['name']}-n{n}-{rec['digest_sha256'][:16]}.json"), "wb") as f:
        f.write(r.stdout)
    return rec["reps"][0]["counts"]["Ir"], r.returncode


def check(t, split, out):
