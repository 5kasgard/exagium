// The seccomp half of the jail's fallback (see confine.c), shared by confine.c and timer.c:
// no new processes or threads (fork, vfork, clone, clone3), no sockets, no keyring, no ptrace, no process_vm_*,
// no pidfd, no namespaces or mounts, no io_uring, bpf, perf_event_open or userfaultfd, and kill/tkill/tgkill only
// to the process itself. Call after PR_SET_NO_NEW_PRIVS; returns 0 on success.
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

#define LD_NR BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr))
#define DENY(nr, err) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 1), BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (err))
// signals: allowed only when args[0] (low 32 bits; the arches here are little-endian) is our own pid
#define SELF_ONLY(nr, pid) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 4), \
  BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[0])), \
  BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (pid), 1, 0), \
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
#endif
    DENY(__NR_clone, EPERM), DENY(__NR_clone3, ENOSYS),
    DENY(__NR_socket, EPERM), DENY(__NR_socketpair, EPERM),
    DENY(__NR_keyctl, EPERM), DENY(__NR_add_key, EPERM), DENY(__NR_request_key, EPERM),
    DENY(__NR_ptrace, EPERM), DENY(__NR_process_vm_readv, EPERM), DENY(__NR_process_vm_writev, EPERM),
    DENY(__NR_pidfd_open, EPERM), DENY(__NR_pidfd_getfd, EPERM),
    DENY(__NR_unshare, EPERM), DENY(__NR_setns, EPERM), DENY(__NR_mount, EPERM),
    DENY(__NR_io_uring_setup, EPERM), DENY(__NR_bpf, EPERM), DENY(__NR_perf_event_open, EPERM),
    DENY(__NR_userfaultfd, EPERM),
    SELF_ONLY(__NR_kill, pid),
    LD_NR, SELF_ONLY(__NR_tkill, pid),
    LD_NR, SELF_ONLY(__NR_tgkill, pid),
    LD_NR, DENY(__NR_rt_sigqueueinfo, EPERM), DENY(__NR_rt_tgsigqueueinfo, EPERM),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  struct sock_fprog prog = {sizeof f / sizeof f[0], f};
  return prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog);
}

#endif
