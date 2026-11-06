// MLP 64 -> H -> 10 (ReLU), layer 1 run as int16 per-pixel lookup tables, layer 2 in float32.
// ./infer WEIGHTS X N -> N digits and a newline.
// Per image: 64 table rows of H int16 adds (vectorised with GCC vector extensions, which compile to SSE2 on
// x86-64 and NEON on arm64 without -march), ReLU, then one float32 multiply-add of 12 lanes per nonzero
// hidden unit. Loops are unrolled with #pragma GCC unroll so the accumulators stay in registers and table offsets
// are constants. All input is read with one fread. No overflow is possible: train.py scales each unit so
// |b1 + sum_p T| <= 32767 for any image with pixels 0..16; pixels > 16 are rejected.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H 32
typedef short v8s __attribute__((vector_size(16)));
typedef float v4f __attribute__((vector_size(16)));
typedef unsigned char v16u __attribute__((vector_size(16)));

static v8s T[64][17][H / 8];
static v8s B1[H / 8];
static v4f W2[H][3];
static v4f B2[3];

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  FILE *f = fopen(argv[1], "rb");
  if (!f || fread(T, sizeof T, 1, f) != 1 || fread(B1, sizeof B1, 1, f) != 1 || fread(W2, sizeof W2, 1, f) != 1 ||
      fread(B2, sizeof B2, 1, f) != 1 || fgetc(f) != EOF)
    return 2;
  fclose(f);
  long n = atol(argv[3]);
  if (n < 0) return 2;
  FILE *xf = fopen(argv[2], "rb");
  if (!xf) return 2;
  unsigned char *x = malloc((size_t)n * 64 + 1);
  char *out = malloc((size_t)n + 2);
  if (!x || !out) return 2;
  if (n && fread(x, 64, (size_t)n, xf) != (size_t)n) return 2;
  fclose(xf);
  // reject any pixel > 16, 16 bytes at a time
  v16u bad = {0}, lim;
  memset(&lim, 16, sizeof lim);
  for (size_t i = 0; i < (size_t)n * 4; i++) {
    v16u v;
    memcpy(&v, x + 16 * i, 16);
    bad |= (v16u)(v > lim);
  }
  for (int k = 0; k < 16; k++)
    if (bad[k]) return 2;
  for (long i = 0; i < n; i++) {
    const unsigned char *px = x + 64 * i;
    v8s acc[H / 8];
#pragma GCC unroll 8
    for (int k = 0; k < H / 8; k++) acc[k] = B1[k];
    for (int p = 0; p < 64; p += 8) {
      const char *tb = (const char *)T[p];
      const unsigned char *pp = px + p;
#pragma GCC unroll 8
      for (int q = 0; q < 8; q++) {
        const v8s *row = (const v8s *)(tb + q * sizeof T[0] + (size_t)pp[q] * sizeof T[0][0]);
#pragma GCC unroll 8
        for (int k = 0; k < H / 8; k++) acc[k] += row[k];
      }
    }
    short h[H];
#pragma GCC unroll 8
    for (int k = 0; k < H / 8; k++) {
      v8s a = acc[k];
      a &= (v8s)(a > (v8s){0});
      memcpy(h + 8 * k, &a, sizeof a);
    }
    v4f s0 = B2[0], s1 = B2[1], s2 = B2[2];
#pragma GCC unroll 64
    for (int j = 0; j < H; j++) {
      if (!h[j]) continue;
      float v = (float)h[j];
      v4f b = {v, v, v, v};
      s0 += b * W2[j][0];
      s1 += b * W2[j][1];
      s2 += b * W2[j][2];
    }
    float s[12];
    memcpy(s, &s0, 16);
    memcpy(s + 4, &s1, 16);
    memcpy(s + 8, &s2, 16);
    int best = 0;
#pragma GCC unroll 10
    for (int c = 1; c < 10; c++)
      if (s[c] > s[best]) best = c;
    out[i] = (char)('0' + best);
  }
  out[n] = '\n';
  fwrite(out, 1, (size_t)n + 1, stdout);
  return 0;
}
