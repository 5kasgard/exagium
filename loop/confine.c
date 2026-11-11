// confine: the jail's fallback where bubblewrap can't run (containers that refuse unprivileged /proc mounts).
// Needs no privilege and no mounts:
//   - Landlock: the process may read and execute only the --ro paths and read/write only the --rw paths;
//     TCP bind/connect is refused (ABI >= 4), and signals and abstract unix sockets can't leave the domain (ABI >= 6).
//     A Landlocked process also can't ptrace (or open /proc/<pid>/mem of) anything outside its domain.
//   - seccomp: no new processes or threads (fork, vfork, clone, clone3), no sockets, no keyring, no ptrace,
//     no process_vm_*, no pidfd, no namespaces or mounts, no io_uring, bpf, perf_event_open or userfaultfd,
//     and kill/tkill/tgkill only to itself.
// Then it execs the command, keeping its pid. Exit 125 if the confinement can't be applied (fail closed).
//   confine [--no-seccomp] [--ro PATH]... [--ls PATH]... [--rw PATH]... -- command args...
// --no-seccomp applies Landlock only: for the trusted timer, which forks and adds the seccomp filter in its child.
// --ls grants listing only (names, not contents): for directories carved around a hidden file.
// A path that doesn't exist is skipped; a file gets only the rights that apply to files.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "seccomp_filter.h"

#ifndef __NR_landlock_create_ruleset
#define __NR_landlock_create_ruleset 444
#define __NR_landlock_add_rule 445
#define __NR_landlock_restrict_self 446
#endif

#define FS_EXECUTE (1ULL << 0)
#define FS_WRITE_FILE (1ULL << 1)
#define FS_READ_FILE (1ULL << 2)
#define FS_READ_DIR (1ULL << 3)
#define FS_TRUNCATE (1ULL << 14)
#define FS_IOCTL_DEV (1ULL << 15)
#define NET_BIND_TCP (1ULL << 0)
#define NET_CONNECT_TCP (1ULL << 1)
#define SCOPE_ABSTRACT_UNIX (1ULL << 0)
#define SCOPE_SIGNAL (1ULL << 1)

struct ruleset_attr { uint64_t fs, net, scoped; };
struct path_beneath { uint64_t allowed; int32_t parent_fd; } __attribute__((packed));

static int fail(const char *what) {
  fprintf(stderr, "confine: %s: %s\n", what, strerror(errno));
  return 125;
}

static uint64_t fs_handled(int abi) {
  uint64_t all = (1ULL << 13) - 1;  // ABI 1: execute .. make_sym
  if (abi >= 2) all |= 1ULL << 13;  // refer
  if (abi >= 3) all |= FS_TRUNCATE;
  if (abi >= 5) all |= FS_IOCTL_DEV;
  return all;
}

static int add(int ruleset, const char *path, uint64_t want, uint64_t handled) {
  int fd = open(path, O_PATH | O_CLOEXEC);
  if (fd < 0) return errno == ENOENT ? 0 : -1;
  struct stat st;
  if (fstat(fd, &st)) { close(fd); return -1; }
  uint64_t allowed = want & handled;
  if (!S_ISDIR(st.st_mode)) allowed &= FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE | FS_IOCTL_DEV;
  struct path_beneath pb = {allowed, fd};
  long r = syscall(__NR_landlock_add_rule, ruleset, 1 /* LANDLOCK_RULE_PATH_BENEATH */, &pb, 0);
  close(fd);
  return r ? -1 : 0;
}

int main(int argc, char **argv) {
  int i = 1, seccomp = 1;
  if (argc > 1 && !strcmp(argv[1], "--no-seccomp")) { seccomp = 0; i = 2; }
  long abi = syscall(__NR_landlock_create_ruleset, NULL, 0, 1 /* LANDLOCK_CREATE_RULESET_VERSION */);
  if (abi < 1) return fail("Landlock is not available (needs Linux 5.13+ with Landlock enabled)");
  struct ruleset_attr ra = {fs_handled((int)abi), abi >= 4 ? NET_BIND_TCP | NET_CONNECT_TCP : 0,
                            abi >= 6 ? SCOPE_ABSTRACT_UNIX | SCOPE_SIGNAL : 0};
  size_t size = abi >= 6 ? 24 : abi >= 4 ? 16 : 8;
  int rs = (int)syscall(__NR_landlock_create_ruleset, &ra, size, 0);
  if (rs < 0) return fail("landlock_create_ruleset");
  for (; i < argc && strcmp(argv[i], "--"); i += 2) {
    if (i + 1 >= argc) { fprintf(stderr, "confine: %s needs a path\n", argv[i]); return 125; }
    uint64_t want;
    if (!strcmp(argv[i], "--ro")) want = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR;
    else if (!strcmp(argv[i], "--ls")) want = FS_READ_DIR;
    else if (!strcmp(argv[i], "--rw")) want = ~0ULL;
    else { fprintf(stderr, "confine: unknown option %s\n", argv[i]); return 125; }
    if (add(rs, argv[i + 1], want, ra.fs)) return fail(argv[i + 1]);
  }
  if (i >= argc - 1) { fprintf(stderr, "usage: confine [--no-seccomp] [--ro PATH]... [--ls PATH]... [--rw PATH]... -- command args...\n"); return 125; }
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) return fail("no_new_privs");
  if (syscall(__NR_landlock_restrict_self, rs, 0)) return fail("landlock_restrict_self");
  close(rs);
  if (seccomp && exagium_seccomp((uint32_t)getpid())) return fail("seccomp");
  execv(argv[i + 1], argv + i + 1);
  return fail(argv[i + 1]);
}
