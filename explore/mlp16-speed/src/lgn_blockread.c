// VARIANT lgn_d_blockread of pilots/lgn-20261115-2/lgn_d/infer.c (explore/mlp16-speed): reads one 1,024-image block at a time.
// Logic-gate network, run bit-sliced: 1,024 images per block, one bit per image in every signal.
// ./infer WEIGHTS X N -> N digits and a newline.
// weights.bin (uint16 little-endian): n_slots, n_groups, then per group: op, count, count x (a, b, out) slot
// indices, then 4 output slots (the bits of the predicted digit, LSB first). Slots 0..319 hold the 5 bit-planes of
// each pixel (slot 5p+k = bit k of pixel p), slot 320 is all zeros, 321 all ones. train.py compiled the whole
// classifier (threshold comparators, the learned gates, per-class popcounts, the argmax) into these gates; this
// program transposes pixels into bit-planes, runs the gates one group at a time (one op per group, so no per-gate
// dispatch) and reads the 4 answer bits back. A signal is 8 vectors of 2 x uint64 (GCC vector extensions: SSE2 on
// x86-64, NEON on arm64, no -march); image j of a block is bit j & 63 of uint64 number j >> 6.
// The slot array spans every uint16 index, so no index in weights.bin can reach outside it.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long u64;
typedef u64 vec __attribute__((vector_size(16)));
#define KV 8
#define BLK (128 * KV)
#define NIN 320

// one stage of an SWAR transpose: swap the bits of a selected by m << s with the bits of b selected by m
#define SWAP(a, b, m, s)                 \
  do {                                   \
    vec t_ = (((a) >> (s)) ^ (b)) & (m); \
    (b) ^= t_;                           \
    (a) ^= t_ << (s);                    \
  } while (0)

static vec S[65536][KV];
static unsigned short W[1 << 21];

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  FILE *f = fopen(argv[1], "rb");
  if (!f) return 2;
  size_t nw = fread(W, 2, sizeof W / 2, f);
  if (nw < 6 || !feof(f)) return 2;
  fclose(f);
  size_t ngroups = W[1], pos = 2;
  for (size_t g = 0; g < ngroups; g++) {  // structure only: every index is a uint16, so every slot exists
    if (pos + 2 > nw || W[pos] > 15) return 2;
    pos += 2 + 3 * (size_t)W[pos + 1];
  }
  if (pos + 4 != nw) return 2;
  const unsigned short *outs = W + pos;

  long n = atol(argv[3]);
  if (n < 0) return 2;
  FILE *xf = fopen(argv[2], "rb");
  if (!xf) return 2;
  long nb = (n + BLK - 1) / BLK;
  unsigned char *x = malloc((size_t)BLK * 64 + 16);  // one block of input, reused
  char *out = malloc((size_t)n + 2);
  if (!x || !out) return 2;
  const vec ones = {~0ULL, ~0ULL}, zero = {0, 0};
  for (int v = 0; v < KV; v++) S[NIN][v] = zero, S[NIN + 1][v] = ones;
  const vec m4 = {0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0F0F0F0F0FULL}, m2 = {0x3333333333333333ULL, 0x3333333333333333ULL},
            m1 = {0x5555555555555555ULL, 0x5555555555555555ULL}, b32 = {0x00000000FFFFFFFFULL, 0x00000000FFFFFFFFULL},
            b16 = {0x0000FFFF0000FFFFULL, 0x0000FFFF0000FFFFULL}, b8 = {0x00FF00FF00FF00FFULL, 0x00FF00FF00FF00FFULL},
            hi3 = {0xE0E0E0E0E0E0E0E0ULL, 0xE0E0E0E0E0E0E0E0ULL};
  vec bad = zero;

  for (long blk = 0; blk < nb; blk++) {
    {  // read only the images of this block; zero the tail of a partial block
      long m = n - blk * BLK < BLK ? n - blk * BLK : BLK;
      if (fread(x, 64, (size_t)m, xf) != (size_t)m) return 2;
      if (m < BLK) memset(x + (size_t)m * 64, 0, (size_t)(BLK - m) * 64);
    }
    const unsigned char *xb = x;
    for (int t = 0; t < BLK / 64; t++)  // 64 images: one uint64 of every input signal
      for (int P = 0; P < 4; P++) {     // 16 pixels: lane 0 holds pixels 16P..16P+7, lane 1 16P+8..16P+15
        vec m[5][8];                    // m[b][g]: byte q = pixel q of the lane, bit i = image 8g+i, plane b
        for (int g = 0; g < 8; g++) {
          vec w[8];
#pragma GCC unroll 8
          for (int i = 0; i < 8; i++) {
            memcpy(&w[i], xb + (size_t)(64 * t + 8 * g + i) * 64 + 16 * P, 16);
            bad |= w[i] & hi3;
          }
          // 8x8 bit transpose inside every byte: word i bit b -> word b bit i
#pragma GCC unroll 4
          for (int i = 0; i < 4; i++) SWAP(w[i], w[i + 4], m4, 4);
          SWAP(w[0], w[2], m2, 2); SWAP(w[1], w[3], m2, 2); SWAP(w[4], w[6], m2, 2); SWAP(w[5], w[7], m2, 2);
          SWAP(w[0], w[1], m1, 1); SWAP(w[2], w[3], m1, 1); SWAP(w[4], w[5], m1, 1); SWAP(w[6], w[7], m1, 1);
#pragma GCC unroll 5
          for (int b = 0; b < 5; b++) m[b][g] = w[b];
        }
#pragma GCC unroll 5
        for (int b = 0; b < 5; b++) {  // 8x8 byte transpose: word g byte q -> word q byte g
          vec *r = m[b];
#pragma GCC unroll 4
          for (int g = 0; g < 4; g++) SWAP(r[g], r[g + 4], b32, 32);
          SWAP(r[0], r[2], b16, 16); SWAP(r[1], r[3], b16, 16); SWAP(r[4], r[6], b16, 16); SWAP(r[5], r[7], b16, 16);
          SWAP(r[0], r[1], b8, 8); SWAP(r[2], r[3], b8, 8); SWAP(r[4], r[5], b8, 8); SWAP(r[6], r[7], b8, 8);
#pragma GCC unroll 8
          for (int q = 0; q < 8; q++) {
            ((u64 *)S[(16 * P + q) * 5 + b])[t] = r[q][0];
            ((u64 *)S[(16 * P + 8 + q) * 5 + b])[t] = r[q][1];
          }
        }
      }
    for (int p = 0; p < 64; p++)  // a pixel in 17..31 has bit 4 and some lower bit set
#pragma GCC unroll 8
      for (int v = 0; v < KV; v++)
        bad |= S[5 * p + 4][v] & (S[5 * p][v] | S[5 * p + 1][v] | S[5 * p + 2][v] | S[5 * p + 3][v]);
    if (bad[0] | bad[1]) return 2;

    const unsigned short *r = W + 2;
    for (size_t gi = 0; gi < ngroups; gi++) {
      int op = r[0];
      size_t cnt = r[1];
      r += 2;
#define RUN(EXPR)                                          \
  for (size_t j = 0; j < cnt; j++, r += 3) {               \
    const vec *A = S[r[0]], *B = S[r[1]];                  \
    vec *O = S[r[2]];                                      \
    _Pragma("GCC unroll 8") for (int v = 0; v < KV; v++) { \
      vec a = A[v], b = B[v];                              \
      (void)a; (void)b;                                    \
      O[v] = (EXPR);                                       \
    }                                                      \
  }                                                        \
  break;
      switch (op) {
        case 0: RUN(zero)
        case 1: RUN(a & b)
        case 2: RUN(a & ~b)
        case 3: RUN(a)
        case 4: RUN(~a & b)
        case 5: RUN(b)
        case 6: RUN(a ^ b)
        case 7: RUN(a | b)
        case 8: RUN(~(a | b))
        case 9: RUN(~(a ^ b))
        case 10: RUN(~b)
        case 11: RUN(a | ~b)
        case 12: RUN(~a)
        case 13: RUN(~a | b)
        case 14: RUN(~(a & b))
        default: RUN(ones)
      }
    }
    const u64 *o0 = (const u64 *)S[outs[0]], *o1 = (const u64 *)S[outs[1]], *o2 = (const u64 *)S[outs[2]],
              *o3 = (const u64 *)S[outs[3]];
    long base = blk * BLK, lim = n - base < BLK ? n - base : BLK;
    for (long j = 0; j < lim; j++) {
      int w = (int)(j >> 6), s = (int)(j & 63);
      int d = (int)(((o0[w] >> s) & 1) | (((o1[w] >> s) & 1) << 1) | (((o2[w] >> s) & 1) << 2) | (((o3[w] >> s) & 1) << 3));
      out[base + j] = (char)('0' + d);
    }
  }
  fclose(xf);
  out[n] = '\n';
  fwrite(out, 1, (size_t)n + 1, stdout);
  return 0;
}
