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
