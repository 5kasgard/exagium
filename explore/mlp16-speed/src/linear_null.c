// VARIANT linear_null of loop/baselines/linear/infer.c (explore/mlp16-speed): same per-image fread, no compute.
// Baseline 1 inference: linear on pixels/16. ./infer WEIGHTS X N -> N digits and a newline.
// The 1/16 scale is folded into the weights at load. Weights are transposed to pixel-major so the
// inner loop runs over the 10 classes, which the compiler can vectorise without reordering any sum.
#include <stdio.h>
#include <stdlib.h>
static float W[10][64], V[64][10], b[10];
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  FILE *f = fopen(argv[1], "rb");
  if (!f || fread(W, sizeof W, 1, f) != 1 || fread(b, sizeof b, 1, f) != 1) return 2;
  fclose(f);
  for (int c = 0; c < 10; c++) for (int p = 0; p < 64; p++) V[p][c] = W[c][p] / 16.0f;
  long n = atol(argv[3]);
  FILE *x = fopen(argv[2], "rb");
  if (!x) return 2;
  char *out = malloc(n + 2);
  unsigned char px[64];
  for (long i = 0; i < n; i++) {
    if (fread(px, 64, 1, x) != 1) return 2;
    out[i] = (char)('0' + (px[0] & 7));
  }
  out[n] = '\n';
  fwrite(out, 1, n + 1, stdout);
  return 0;
}
