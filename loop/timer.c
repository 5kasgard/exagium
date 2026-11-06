// Trusted timer, run inside the candidate's jail: ./timer CPU PROG ARGS...
// Pins itself (and so the child) to one CPU, forks, runs PROG with stdin and stderr on /dev/null and
// stdout on a pipe, and measures CLOCK_MONOTONIC from just before fork to the child's exit.
// Prints the child's output, then a newline and one line "exagium-timer <ns> <exit status> <cpu ns>", where
// cpu ns is the user+system time of the child and everything it waited for (more than wall time = more than one CPU).
// It makes itself non-dumpable first, so the child can't reach its file descriptors through /proc.
#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc < 3) return 120;
  if (prctl(PR_SET_DUMPABLE, 0) != 0) return 121;
  cpu_set_t set; CPU_ZERO(&set); CPU_SET(atoi(argv[1]), &set);
  if (sched_setaffinity(0, sizeof set, &set) != 0) return 122;
  int p[2];
  if (pipe2(p, O_CLOEXEC) != 0) return 123;
  int devnull = open("/dev/null", O_RDWR | O_CLOEXEC);
  if (devnull < 0) return 124;
  size_t cap = 1 << 16, len = 0;
  char *buf = malloc(cap);
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  pid_t pid = fork();
  if (pid < 0) return 125;
  if (pid == 0) {
    dup2(devnull, 0); dup2(p[1], 1); dup2(devnull, 2);
    execv(argv[2], argv + 2);
    _exit(127);
  }
  close(p[1]);
  for (;;) {
    if (len == cap) { cap *= 2; buf = realloc(buf, cap); if (!buf) return 126; }
    ssize_t r = read(p[0], buf + len, cap - len);
    if (r <= 0) break;
    len += (size_t)r;
  }
  int status;
  struct rusage ru;
  if (wait4(pid, &status, 0, &ru) != pid) return 127;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  long long ns = (long long)(t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
  fwrite(buf, 1, len, stdout);
  long long cpu = ((long long)ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000000000LL
                + ((long long)ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1000LL;
  printf("\nexagium-timer %lld %d %lld\n", ns, WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status), cpu);
  return 0;
}
