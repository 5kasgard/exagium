// Baseline 2 inference: thermometer plus linear, run as the lookup table it equals.
// ./infer WEIGHTS X N -> N digits and a newline. Per image: 64 table rows of 10 adds.
#include <stdio.h>
#include <stdlib.h>
static float T[64][17][10], b[10];
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  FILE *f = fopen(argv[1], "rb");
  if (!f || fread(T, sizeof T, 1, f) != 1 || fread(b, sizeof b, 1, f) != 1) return 2;
  fclose(f);
  long n = atol(argv[3]);
  FILE *x = fopen(argv[2], "rb");
  if (!x) return 2;
  char *out = malloc(n + 2);
  unsigned char px[64];
  for (long i = 0; i < n; i++) {
    if (fread(px, 64, 1, x) != 1) return 2;
    float s[10];
    for (int c = 0; c < 10; c++) s[c] = b[c];
    for (int p = 0; p < 64; p++) {
      if (px[p] > 16) return 2;
      const float *row = T[p][px[p]];
      for (int c = 0; c < 10; c++) s[c] += row[c];
    }
    int best = 0;
    for (int c = 1; c < 10; c++) if (s[c] > s[best]) best = c;
    out[i] = (char)('0' + best);
  }
  out[n] = '\n';
  fwrite(out, 1, n + 1, stdout);
  return 0;
}
