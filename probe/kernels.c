// forge-probe measurement kernels. One measurement, one repetition, per invocation.
// Prints one JSON object on stdout. The driver (probe.py) repeats and interleaves.
//
//   kernels timer
//   kernels chase  <bytes> <line_bytes> <min_ms> <hugepage 0|1> [seed] -> ns per dependent load
//   kernels stream <bytes> <threads> <min_ms> <hugepage 0|1>       -> GB/s read
//   kernels flops  <scalar_f64|vec_f64|vec_f32> <threads> <min_ms> -> GFLOP/s (FMA = 2 flops)
//
// Everything here is timing only (meter tier 2). No counters are read.
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *big_alloc(size_t bytes, int huge) {
  size_t align = 2u << 20;
  size_t sz = (bytes + align - 1) / align * align;
  void *p = aligned_alloc(align, sz);
  if (!p) { fprintf(stderr, "alloc failed\n"); exit(2); }
#ifdef MADV_HUGEPAGE
  if (huge) madvise(p, sz, MADV_HUGEPAGE);
#endif
  return p;
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t xs(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

// ---------- timer ----------
static int cmd_timer(void) {
  // resolution: smallest nonzero step between consecutive reads; overhead: mean cost of a read
  double best = 1e9, t0 = now();
  for (int i = 0; i < 100000; i++) {
    double a = now(), b = now();
    while (b == a) b = now();
    if (b - a < best) best = b - a;
  }
  int n = 1000000; double s = now();
  for (int i = 0; i < n; i++) (void)now();
  double ovh = (now() - s) / n;
  struct timespec res; clock_getres(CLOCK_MONOTONIC, &res);
  // min_step_ns: smallest observable gap between two reads (bounded below by read cost);
  // getres_ns: what the kernel reports as the clock's resolution.
  printf("{\"kind\":\"timer\",\"min_step_ns\":%.3f,\"read_overhead_ns\":%.3f,\"getres_ns\":%.3f,\"seconds\":%.6f}\n",
         best * 1e9, ovh * 1e9, res.tv_sec * 1e9 + res.tv_nsec, now() - t0);
  return 0;
}

// ---------- latency: dependent pointer chase over a random single cycle of cache lines ----------
static int cmd_chase(size_t bytes, size_t line, double min_ms, int huge, uint64_t seed) {
  if (seed) rng = seed;
  if (line < sizeof(void *)) line = sizeof(void *);
  size_t n = bytes / line;
  if (n < 2) n = 2;
  char *buf = big_alloc(n * line, huge);
  size_t *perm = malloc(n * sizeof(size_t));
  for (size_t i = 0; i < n; i++) perm[i] = i;
  // Sattolo's algorithm: uniform random single cycle, so every line is visited once per lap
  for (size_t i = n - 1; i > 0; i--) { size_t j = xs() % i; size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
  for (size_t i = 0; i < n; i++) *(void **)(buf + i * line) = buf + perm[i] * line;
  free(perm);
  void **p = (void **)buf;
  for (size_t k = 0; k < n; k++) p = (void **)*p;  // warm: one full lap
  size_t steps = 1 << 16; double dt = 0;
  for (;;) {
    double t0 = now();
    for (size_t k = 0; k < steps; k++) p = (void **)*p;
    dt = now() - t0;
    if (dt * 1e3 >= min_ms) break;
    steps *= 2;
  }
  printf("{\"kind\":\"chase\",\"bytes\":%zu,\"line\":%zu,\"seed\":%llu,\"steps\":%zu,\"seconds\":%.9f,\"ns_per_load\":%.4f,\"sink\":%d}\n",
         n * line, line, (unsigned long long)seed, steps, dt, dt / steps * 1e9, p == NULL);
  free(buf);
  return 0;
}

// ---------- bandwidth: streaming read with vector accumulators ----------
typedef double v8d __attribute__((vector_size(64)));
typedef float v16f __attribute__((vector_size(64)));

typedef struct {
  const char *base; size_t bytes; size_t passes; pthread_barrier_t *bar; double sink;
} stream_arg;

static void *stream_worker(void *vp) {
  stream_arg *a = vp;
  const v8d *v = (const v8d *)a->base; size_t nv = a->bytes / sizeof(v8d);
  // 8 independent accumulators: enough to cover vector-add latency, so L1-resident
  // reads are limited by loads, not by the add chain (4 was add-bound: ~1 vector/cycle).
  v8d s0 = {0}, s1 = {0}, s2 = {0}, s3 = {0}, s4 = {0}, s5 = {0}, s6 = {0}, s7 = {0};
  pthread_barrier_wait(a->bar);
  for (size_t r = 0; r < a->passes; r++)
    for (size_t i = 0; i + 8 <= nv; i += 8) {
      s0 += v[i]; s1 += v[i + 1]; s2 += v[i + 2]; s3 += v[i + 3];
      s4 += v[i + 4]; s5 += v[i + 5]; s6 += v[i + 6]; s7 += v[i + 7];
    }
  pthread_barrier_wait(a->bar);
  v8d s = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7; double t = 0;
  for (int k = 0; k < 8; k++) t += s[k];
  a->sink = t;
  return NULL;
}

static double stream_once(char *buf, size_t bytes, int th, size_t passes, double *sink) {
  pthread_t tid[256]; stream_arg arg[256]; pthread_barrier_t bar;
  pthread_barrier_init(&bar, NULL, th + 1);
  size_t chunk = bytes / th / 512 * 512;
  for (int t = 0; t < th; t++) {
    arg[t] = (stream_arg){buf + t * chunk, chunk, passes, &bar, 0};
    pthread_create(&tid[t], NULL, stream_worker, &arg[t]);
  }
  pthread_barrier_wait(&bar); double t0 = now();
  pthread_barrier_wait(&bar); double dt = now() - t0;
  for (int t = 0; t < th; t++) { pthread_join(tid[t], NULL); *sink += arg[t].sink; }
  pthread_barrier_destroy(&bar);
  return dt;
}

static int cmd_stream(size_t bytes, int th, double min_ms, int huge) {
  if (th < 1 || th > 256) th = 1;
  char *buf = big_alloc(bytes, huge);
  double *d = (double *)buf;
  for (size_t i = 0; i < bytes / sizeof(double); i++) d[i] = (double)(i & 1023) * 0.5;
  size_t chunk = bytes / th / 512 * 512, passes = 1; double sink = 0, dt;
  stream_once(buf, bytes, th, 1, &sink);  // warm
  for (;;) {
    dt = stream_once(buf, bytes, th, passes, &sink);
    if (dt * 1e3 >= min_ms) break;
    passes *= 2;
  }
  double moved = (double)chunk * th * passes;
  printf("{\"kind\":\"stream\",\"bytes\":%zu,\"threads\":%d,\"passes\":%zu,\"seconds\":%.9f,\"GBps\":%.4f,\"sink\":%d}\n",
         chunk * th, th, passes, dt, moved / dt / 1e9, sink == 1.2345);
  free(buf);
  return 0;
}

// ---------- compute: independent FMA chains, enough of them to hide FMA latency ----------
#define NACC 16
__attribute__((optimize("no-tree-vectorize", "no-tree-slp-vectorize")))
static double flops_scalar(size_t it) {
  double x[NACC]; for (int j = 0; j < NACC; j++) x[j] = 1.0 + j * 1e-3;
  const double m = 0.9999999, c = 1e-9;
  for (size_t i = 0; i < it; i++)
#pragma GCC unroll 16
    for (int j = 0; j < NACC; j++) x[j] = x[j] * m + c;
  double s = 0; for (int j = 0; j < NACC; j++) s += x[j];
  return s;
}
static double flops_vf64(size_t it) {
  v8d x[NACC]; for (int j = 0; j < NACC; j++) for (int k = 0; k < 8; k++) x[j][k] = 1.0 + j * 1e-3 + k * 1e-4;
  const v8d m = {0.9999999,0.9999999,0.9999999,0.9999999,0.9999999,0.9999999,0.9999999,0.9999999};
  const v8d c = {1e-9,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9};
  for (size_t i = 0; i < it; i++)
#pragma GCC unroll 16
    for (int j = 0; j < NACC; j++) x[j] = x[j] * m + c;
  double s = 0; for (int j = 0; j < NACC; j++) for (int k = 0; k < 8; k++) s += x[j][k];
  return s;
}
static double flops_vf32(size_t it) {
  v16f x[NACC]; for (int j = 0; j < NACC; j++) for (int k = 0; k < 16; k++) x[j][k] = 1.0f + j * 1e-3f + k * 1e-4f;
  v16f m, c; for (int k = 0; k < 16; k++) { m[k] = 0.9999999f; c[k] = 1e-9f; }
  for (size_t i = 0; i < it; i++)
#pragma GCC unroll 16
    for (int j = 0; j < NACC; j++) x[j] = x[j] * m + c;
  double s = 0; for (int j = 0; j < NACC; j++) for (int k = 0; k < 16; k++) s += x[j][k];
  return s;
}

typedef struct { int kind; size_t it; pthread_barrier_t *bar; double sink; } flops_arg;
static void *flops_worker(void *vp) {
  flops_arg *a = vp;
  pthread_barrier_wait(a->bar);
  a->sink = a->kind == 0 ? flops_scalar(a->it) : a->kind == 1 ? flops_vf64(a->it) : flops_vf32(a->it);
  pthread_barrier_wait(a->bar);
  return NULL;
}
static double flops_once(int kind, int th, size_t it, double *sink) {
  pthread_t tid[256]; flops_arg arg[256]; pthread_barrier_t bar;
  pthread_barrier_init(&bar, NULL, th + 1);
  for (int t = 0; t < th; t++) { arg[t] = (flops_arg){kind, it, &bar, 0}; pthread_create(&tid[t], NULL, flops_worker, &arg[t]); }
  pthread_barrier_wait(&bar); double t0 = now();
  pthread_barrier_wait(&bar); double dt = now() - t0;
  for (int t = 0; t < th; t++) { pthread_join(tid[t], NULL); *sink += arg[t].sink; }
  pthread_barrier_destroy(&bar);
  return dt;
}
static int cmd_flops(const char *k, int th, double min_ms) {
  int kind = !strcmp(k, "scalar_f64") ? 0 : !strcmp(k, "vec_f64") ? 1 : !strcmp(k, "vec_f32") ? 2 : -1;
  if (kind < 0) { fprintf(stderr, "unknown flops kind\n"); return 1; }
  if (th < 1 || th > 256) th = 1;
  int lanes = kind == 0 ? 1 : kind == 1 ? 8 : 16;
  size_t it = 1 << 12; double sink = 0, dt;
  flops_once(kind, th, it, &sink);  // warm (also lets clocks settle)
  for (;;) {
    dt = flops_once(kind, th, it, &sink);
    if (dt * 1e3 >= min_ms) break;
    it *= 2;
  }
  double fl = 2.0 * lanes * NACC * (double)it * th;
  printf("{\"kind\":\"flops\",\"op\":\"%s\",\"threads\":%d,\"lanes\":%d,\"chains\":%d,\"iters\":%zu,\"seconds\":%.9f,\"GFLOPs\":%.4f,\"sink\":%d}\n",
         k, th, lanes, NACC, it, dt, fl / dt / 1e9, sink == 1.2345);
  return 0;
}

// Which vector instructions this build can use for the 64-byte vector types, decided by the
// compiler from -march; recorded so a vector ceiling is never read as another ISA's.
static int cmd_isa(void) {
  const char *isa =
#if defined(__AVX512F__)
      "avx512";
#elif defined(__AVX2__)
      "avx2";
#elif defined(__AVX__)
      "avx";
#elif defined(__ARM_FEATURE_SVE)
      "sve";
#elif defined(__ARM_NEON)
      "neon";
#elif defined(__SSE2__)
      "sse2";
#else
      "scalar-emulated";
#endif
  printf("{\"kind\":\"isa\",\"vector_isa\":\"%s\"}\n", isa);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "isa")) return cmd_isa();
  if (argc >= 2 && !strcmp(argv[1], "timer")) return cmd_timer();
  if ((argc == 6 || argc == 7) && !strcmp(argv[1], "chase"))
    return cmd_chase(strtoull(argv[2], 0, 10), strtoull(argv[3], 0, 10), atof(argv[4]), atoi(argv[5]),
                     argc == 7 ? strtoull(argv[6], 0, 10) : 0);
  if (argc == 6 && !strcmp(argv[1], "stream"))
    return cmd_stream(strtoull(argv[2], 0, 10), atoi(argv[3]), atof(argv[4]), atoi(argv[5]));
  if (argc == 5 && !strcmp(argv[1], "flops")) return cmd_flops(argv[2], atoi(argv[3]), atof(argv[4]));
  fprintf(stderr, "usage: see header of kernels.c\n");
  return 1;
}
