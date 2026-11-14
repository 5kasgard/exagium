// One-shot probe: is a hardware PMU available to this (unprivileged) process in this VM?
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
static int open_one(__u64 config) {
  struct perf_event_attr a;
  memset(&a, 0, sizeof a);
  a.size = sizeof a;
  a.type = PERF_TYPE_HARDWARE;
  a.config = config;
  a.disabled = 0;
  a.exclude_kernel = 1;
  return (int)syscall(__NR_perf_event_open, &a, 0, -1, -1, 0);
}
int main(void) {
  int c = open_one(PERF_COUNT_HW_CPU_CYCLES);
  int i = open_one(PERF_COUNT_HW_INSTRUCTIONS);
  if (c < 0 || i < 0) {
    printf("NO HARDWARE COUNTERS: cycles=%d (%s), instructions=%d (%s)\n",
           c, strerror(errno), i, i < 0 ? strerror(errno) : "ok");
    printf("perf_event_paranoid=%d\n",
           ({ int v = 99; FILE *f = fopen("/proc/sys/kernel/perf_event_paranoid", "r");
              if (f) { if (fscanf(f, "%d", &v) != 1) v = 99; fclose(f); } v; }));
    return 1;
  }
  printf("HARDWARE COUNTERS AVAILABLE\n");
  long long oc = 0, oi = 0;
  ioctl(c, PERF_EVENT_IOC_RESET, 0); ioctl(i, PERF_EVENT_IOC_RESET, 0);
  for (volatile int k = 0; k < 1000000; k++) ;
  if (read(c, &oc, sizeof oc) == sizeof oc && read(i, &oi, sizeof oi) == sizeof oi)
    printf("sample: cycles=%lld instructions=%lld\n", oc, oi);
  close(c); close(i);
  return 0;
}
