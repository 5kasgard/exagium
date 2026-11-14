// PROBE ceil (H6): how many instructions/ns does a trivially parallel loop reach on this host?
// Per image: 64 independent byte loads and 64 independent int32 adds in 8 parallel accumulator
// chains (8 accumulators fit in registers; the earlier 16-chain version spilled 3 to the stack
// and paid read-modify-write chains - both numbers are reported), then a 7-add fold:
//   out[i] = '0' + ((s0+..+s7) & 7).

// Same CLI as the others (WEIGHTS X N); argv[1] is accepted and ignored (no weights are read).
// Output is digits + newline, deterministic; differs from every other program by construction.
// This is a ceiling probe, not an mlp16 variant: report its Ir/ns next to linear's and mlp16's.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  long n = atol(argv[3]);
  if (n < 0) return 2;
  FILE *xf = fopen(argv[2], "rb");
  if (!xf) return 2;
  unsigned char *x = malloc((size_t)n * 64 + 1);
  char *out = malloc((size_t)n + 2);
  if (!x || !out) return 2;
  if (n && fread(x, 64, (size_t)n, xf) != (size_t)n) return 2;
  fclose(xf);
  for (long i = 0; i < n; i++) {
    const unsigned char *px = x + 64 * i;
    unsigned s[8] = {0};  /* CHANGED: 8 chains, register-clean */
    for (int p = 0; p < 64; p += 8) {
#pragma GCC unroll 8
      for (int q = 0; q < 8; q++) s[q] += px[p + q];
    }
    out[i] = (char)('0' + (int)((s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7]) & 7));
  }
  out[n] = '\n';
  fwrite(out, 1, (size_t)n + 1, stdout);
  return 0;
}
