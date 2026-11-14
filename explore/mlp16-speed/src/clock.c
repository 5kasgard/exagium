#define _POSIX_C_SOURCE 199309L
// Clock calibration without hardware counters: a chain of dependent integer adds retires one add per cycle on any
// core of the last 20 years (1-cycle latency), so adds per ns = the core clock in GHz while this runs.
// ./clock [millions of iterations, default 200]  -> prints GHz (median of 9 runs). Pin it like the bench:
// taskset -c N ./clock. Build: gcc -O2 -o bin/clock src/clock.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
int main(int argc, char **argv) {
  long iters = (argc > 1 ? atol(argv[1]) : 200) * 1000000L;
  double r[9];
  unsigned long x = 1, y = 3;
  for (int k = 0; k < 9; k++) {
    double t0 = now();
    for (long i = 0; i < iters; i++) {
#if defined(__x86_64__)
      __asm__ volatile("add %1,%0\n\tadd %1,%0\n\tadd %1,%0\n\tadd %1,%0\n\tadd %1,%0\n\t"
                       "add %1,%0\n\tadd %1,%0\n\tadd %1,%0\n\tadd %1,%0\n\tadd %1,%0" : "+r"(x) : "r"(y));
#elif defined(__aarch64__)
      __asm__ volatile("add %0,%0,%1\n\tadd %0,%0,%1\n\tadd %0,%0,%1\n\tadd %0,%0,%1\n\tadd %0,%0,%1\n\t"
                       "add %0,%0,%1\n\tadd %0,%0,%1\n\tadd %0,%0,%1\n\tadd %0,%0,%1\n\tadd %0,%0,%1" : "+r"(x) : "r"(y));
#else
#error "x86_64 or aarch64 only"
#endif
    }
    r[k] = 10.0 * iters / (now() - t0);
  }
  qsort(r, 9, sizeof r[0], cmp);
  printf("%.3f GHz (median of 9; min %.3f max %.3f) x=%lu\n", r[4], r[0], r[8], x);
  return 0;
}
