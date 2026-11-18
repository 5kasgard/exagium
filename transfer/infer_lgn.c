/* Logic-gate networks of the transfer plan: ./infer_lgn weights x.bin N > pred.bin
 * Bit-sliced: 64 items per uint64 word. Input bits are built from the bytes (v >= t or v == c), the gates run
 * per word, each class's group is counted with bit-sliced counters, and the argmax (first max wins) is
 * taken per item. Prints N class indices as raw bytes. N = 0 is start-up only. Format: see families.py.
 * Compile: gcc -O2 -std=c11 -ffp-contract=off infer_lgn.c -o infer_lgn */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

static inline uint64_t op(int k, uint64_t a, uint64_t b) {
    switch (k) {
    case 0: return 0;
    case 1: return a & b;
    case 2: return a & ~b;
    case 3: return a;
    case 4: return ~a & b;
    case 5: return b;
    case 6: return a ^ b;
    case 7: return a | b;
    case 8: return ~(a | b);
    case 9: return ~(a ^ b);
    case 10: return ~b;
    case 11: return a | ~b;
    case 12: return ~a;
    case 13: return ~a | b;
    case 14: return ~(a & b);
    default: return ~(uint64_t)0;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer_lgn weights x.bin N\n"); return 2; }
    long nw, nx, N = atol(argv[3]);
    int32_t *w = slurp(argv[1], &nw);
    if (nw < 24 || w[0] != 0x4C474E31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = w[1], C = w[2], g = w[3], U = w[4], L = w[5];
    if (C < 1 || C > 255 || g < 1 || g > 255) { fprintf(stderr, "bad header\n"); return 3; }
    int32_t *bits = w + 6, *p = bits + 3 * U;
    int32_t **ga = malloc(sizeof(int32_t *) * L);
    int *wid = malloc(sizeof(int) * L), maxw = U;
    for (int l = 0; l < L; l++) {
        wid[l] = *p++; ga[l] = p; p += 3 * wid[l];
        if (wid[l] > maxw) maxw = wid[l];
        if ((char *)p > (char *)w + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
    }
    if ((char *)p != (char *)w + nw || wid[L - 1] != C * g) { fprintf(stderr, "bad weights size\n"); return 3; }
    uint8_t *x = N ? slurp(argv[2], &nx) : NULL;
    if (N < 0 || (N && nx < N * d)) { fprintf(stderr, "bad N\n"); return 2; }
    uint64_t *bufA = malloc(8 * (size_t)maxw), *bufB = malloc(8 * (size_t)maxw);
    int planes = 0; while ((1 << planes) <= g) planes++;
    uint64_t *cnt = malloc(8 * (size_t)C * planes);
    uint8_t *out = malloc(N ? N : 1);
    for (long base = 0; base < N; base += 64) {
        int lanes = N - base < 64 ? (int)(N - base) : 64;
        const uint8_t *xb = x + base * d;
        for (int u = 0; u < U; u++) {
            int j = bits[3 * u], kind = bits[3 * u + 1], v = bits[3 * u + 2];
            uint64_t word = 0;
            for (int i = 0; i < lanes; i++) {
                int bit = kind ? xb[(long)i * d + j] == v : xb[(long)i * d + j] >= v;
                word |= (uint64_t)bit << i;
            }
            bufA[u] = word;
        }
        uint64_t *in = bufA, *o = bufB;
        for (int l = 0; l < L; l++) {
            const int32_t *G = ga[l];
            for (int k = 0; k < wid[l]; k++) o[k] = op(G[3 * k + 2], in[G[3 * k]], in[G[3 * k + 1]]);
            uint64_t *t = in; in = o; o = t;
        }
        for (int c = 0; c < C; c++) {
            uint64_t *pl = cnt + (size_t)c * planes;
            for (int q = 0; q < planes; q++) pl[q] = 0;
            for (int k = 0; k < g; k++) {
                uint64_t carry = in[c * g + k];
                for (int q = 0; q < planes && carry; q++) { uint64_t t = pl[q] & carry; pl[q] ^= carry; carry = t; }
            }
        }
        for (int i = 0; i < lanes; i++) {
            int best = 0, bv = -1;
            for (int c = 0; c < C; c++) {
                const uint64_t *pl = cnt + (size_t)c * planes;
                int v = 0;
                for (int q = 0; q < planes; q++) v |= (int)((pl[q] >> i) & 1) << q;
                if (v > bv) { bv = v; best = c; }
            }
            out[base + i] = (uint8_t)best;
        }
    }
    fwrite(out, 1, N, stdout);
    return 0;
}
