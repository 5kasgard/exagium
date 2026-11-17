// Staging experiment (explore/mlp16-speed): the same null work as mlp16_null (validate every pixel, print one digit
// per image from its first byte), with the N needed images staged one of several ways. No classifier, so the time
// is staging + validation + output only.
//   STRAT 1  one fresh n*64 B buffer, one fread             (mlp16's way)
//   STRAT 2  CH images at a time, fread into one reused buffer
//   STRAT 3  mmap the file read-only, only the first n*64 B used (no copy)
//   STRAT 4  like 3 with MAP_POPULATE (map every page up front, in one call)
//   STRAT 5  like 1, but the buffer is 2 MB aligned and madvise(MADV_HUGEPAGE) asks for huge pages first
//   STRAT 6  like 2, with read(2) instead of stdio
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef STRAT
#define STRAT 1
#endif
#ifndef CH
#define CH 1024
#endif
typedef unsigned char v16u __attribute__((vector_size(16)));

static int check_and_emit(const unsigned char *x, long m, char *out) {  // returns 0 if every pixel <= 16
  v16u bad = {0}, lim;
  memset(&lim, 16, sizeof lim);
  for (size_t i = 0; i < (size_t)m * 4; i++) {
    v16u v;
    memcpy(&v, x + 16 * i, 16);
    bad |= (v16u)(v > lim);
  }
  for (int k = 0; k < 16; k++)
    if (bad[k]) return 1;
  for (long i = 0; i < m; i++) out[i] = (char)('0' + (x[64 * i] & 7));
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  FILE *f = fopen(argv[1], "rb");  // the weights file is only opened and closed (t(0) subtraction cancels this)
  if (!f) return 2;
  fclose(f);
  long n = atol(argv[3]);
  if (n < 0) return 2;
  char *out = malloc((size_t)n + 2);
  if (!out) return 2;
#if STRAT == 1 || STRAT == 5
  FILE *xf = fopen(argv[2], "rb");
  if (!xf) return 2;
#if STRAT == 1
  unsigned char *x = malloc((size_t)n * 64 + 1);
#else
  size_t sz = ((size_t)n * 64 + (2u << 20)) & ~((size_t)(2u << 20) - 1);
  unsigned char *x = aligned_alloc(2u << 20, sz ? sz : (2u << 20));
  if (x) madvise(x, sz, MADV_HUGEPAGE);
#endif
  if (!x) return 2;
  if (n && fread(x, 64, (size_t)n, xf) != (size_t)n) return 2;
  fclose(xf);
  if (check_and_emit(x, n, out)) return 2;
#elif STRAT == 2
  FILE *xf = fopen(argv[2], "rb");
  if (!xf) return 2;
  unsigned char *x = malloc((size_t)CH * 64 + 1);
  if (!x) return 2;
  for (long base = 0; base < n; base += CH) {
    long m = n - base < CH ? n - base : CH;
    if (fread(x, 64, (size_t)m, xf) != (size_t)m) return 2;
    if (check_and_emit(x, m, out + base)) return 2;
  }
  fclose(xf);
#elif STRAT == 3 || STRAT == 4
  int fd = open(argv[2], O_RDONLY);
  if (fd < 0) return 2;
  struct stat st;
  if (fstat(fd, &st) || (size_t)st.st_size < (size_t)n * 64) return 2;
  if (n) {
    const unsigned char *x = mmap(NULL, (size_t)n * 64, PROT_READ, MAP_PRIVATE
#if STRAT == 4
                                  | MAP_POPULATE
#endif
                                  , fd, 0);
    if (x == MAP_FAILED) return 2;
    if (check_and_emit(x, n, out)) return 2;
  }
  close(fd);
#elif STRAT == 6
  int fd = open(argv[2], O_RDONLY);
  if (fd < 0) return 2;
  unsigned char *x = malloc((size_t)CH * 64 + 1);
  if (!x) return 2;
  for (long base = 0; base < n; base += CH) {
    long m = n - base < CH ? n - base : CH;
    size_t want = (size_t)m * 64, got = 0;
    while (got < want) {
      ssize_t r = read(fd, x + got, want - got);
      if (r <= 0) return 2;
      got += (size_t)r;
    }
    if (check_and_emit(x, m, out + base)) return 2;
  }
  close(fd);
#endif
  out[n] = '\n';
  fwrite(out, 1, (size_t)n + 1, stdout);
  return 0;
}
