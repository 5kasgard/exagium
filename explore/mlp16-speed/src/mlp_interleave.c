// MLP 64 -> H -> 10 (ReLU), mlp16's frozen form and weights, unchanged arithmetic per image.
// ./infer WEIGHTS X N -> N digits and a newline.
// Change from mlp16's infer.c: NB images are classified together, interleaved, so the NB independent chains of
// dependent adds (64 int16 table adds in layer 1, H float adds in layer 2) overlap instead of running one after
// another. Each image's own sums run in exactly the same order as before, so every prediction is bit-identical.
// BRANCHLESS=1 also drops the skip of zero hidden units: 0 * w = +-0 and s + (+-0) == s, so results are unchanged.
// N is padded up to a multiple of NB with zero images, whose outputs are discarded.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef H
#define H 16
#endif
#ifndef NB
#define NB 4
#endif
#ifndef BRANCHLESS
#define BRANCHLESS 0
#endif
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
  long np = (n + NB - 1) / NB * NB;
  unsigned char *x = calloc((size_t)np * 64 + 1, 1);
  char *out = malloc((size_t)np + 2);
  if (!x || !out) return 2;
  if (n && fread(x, 64, (size_t)n, xf) != (size_t)n) return 2;
  fclose(xf);
  v16u bad = {0}, lim;
  memset(&lim, 16, sizeof lim);
  for (size_t i = 0; i < (size_t)n * 4; i++) {
    v16u v;
    memcpy(&v, x + 16 * i, 16);
    bad |= (v16u)(v > lim);
  }
  for (int k = 0; k < 16; k++)
    if (bad[k]) return 2;
  for (long i = 0; i < np; i += NB) {
    v8s acc[NB][H / 8];
#pragma GCC unroll 64
    for (int b = 0; b < NB; b++)
#pragma GCC unroll 8
      for (int k = 0; k < H / 8; k++) acc[b][k] = B1[k];
    const unsigned char *px[NB];
#pragma GCC unroll 64
    for (int b = 0; b < NB; b++) px[b] = x + 64 * (i + b);
    for (int p = 0; p < 64; p += 8) {  // same table addressing as mlp16's infer.c, NB images side by side
      const char *tb = (const char *)T[p];
#pragma GCC unroll 8
      for (int q = 0; q < 8; q++) {
#pragma GCC unroll 64
        for (int b = 0; b < NB; b++) {
          const v8s *row = (const v8s *)(tb + q * sizeof T[0] + (size_t)px[b][p + q] * sizeof T[0][0]);
#pragma GCC unroll 8
          for (int k = 0; k < H / 8; k++) acc[b][k] += row[k];
        }
      }
    }
    short h[NB][H];
#pragma GCC unroll 64
    for (int b = 0; b < NB; b++)
#pragma GCC unroll 8
      for (int k = 0; k < H / 8; k++) {
        v8s a = acc[b][k];
        a &= (v8s)(a > (v8s){0});
        memcpy(h[b] + 8 * k, &a, sizeof a);
      }
    v4f s[NB][3];
#pragma GCC unroll 64
    for (int b = 0; b < NB; b++) s[b][0] = B2[0], s[b][1] = B2[1], s[b][2] = B2[2];
#pragma GCC unroll 64
    for (int j = 0; j < H; j++) {
#pragma GCC unroll 64
      for (int b = 0; b < NB; b++) {
#if !BRANCHLESS
        if (!h[b][j]) continue;
#endif
        float v = (float)h[b][j];
        v4f w = {v, v, v, v};
        s[b][0] += w * W2[j][0];
        s[b][1] += w * W2[j][1];
        s[b][2] += w * W2[j][2];
      }
    }
#pragma GCC unroll 64
    for (int b = 0; b < NB; b++) {
      float sc[12];
      memcpy(sc, &s[b][0], 16);
      memcpy(sc + 4, &s[b][1], 16);
      memcpy(sc + 8, &s[b][2], 16);
      int best = 0;
#pragma GCC unroll 10
      for (int c = 1; c < 10; c++)
        if (sc[c] > sc[best]) best = c;
      out[i + b] = (char)('0' + best);
    }
  }
  out[n] = '\n';
  fwrite(out, 1, (size_t)n + 1, stdout);
  return 0;
}
