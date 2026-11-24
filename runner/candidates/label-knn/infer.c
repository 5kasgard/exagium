/* k nearest neighbours for label tasks (queue P3 step 1): ./infer weights.bin in.rec N
 * Integer distances from per-column cost tables; neighbours by distance, ties to the lower training index; the vote's
 * ties go to the tied class met first in neighbour order. Must agree with train.py's predict(). Format: see train.py. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    int32_t hd[5];
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    memcpy(hd, w, 20);
    int d = hd[1], C = hd[3], k = hd[4];
    long n = hd[2];
    if (hd[0] != 0x314E4E4B || d < 1 || n < 1 || C < 1 || C > 256 || k < 1 || k > 64 ||
        nw != 20 + 2L * d * 65536 + n * d + n) { fprintf(stderr, "bad weights size\n"); return 3; }
    if (k > n) k = n;                                    /* train.py's predict() uses min(k, n) */
    const uint16_t *cost = (const uint16_t *)(w + 20);   /* little-endian host assumed (x86, arm64) */
    const unsigned char *X = w + 20 + 2L * d * 65536, *y = X + n * d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    int64_t bd[64]; long bi[64]; int counts[256], first[256];
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *q = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        int m = 0;
        for (long t = 0; t < n; t++) {
            const unsigned char *r = X + t * d;
            int64_t s = 0;
            for (int j = 0; j < d; j++) s += cost[((long)j * 256 + q[j]) * 256 + r[j]];
            if (m == k && s >= bd[k - 1]) continue;          /* equal distance: the earlier index stays */
            int p = m < k ? m++ : k - 1;
            while (p > 0 && bd[p - 1] > s) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p--; }
            bd[p] = s; bi[p] = t;
        }
        for (int c = 0; c < C; c++) { counts[c] = 0; first[c] = k; }
        for (int p = 0; p < k; p++) { int c = y[bi[p]]; counts[c]++; if (first[c] == k) first[c] = p; }
        int best = 0;
        for (int c = 1; c < C; c++)
            if (counts[c] > counts[best] || (counts[c] == counts[best] && first[c] < first[best])) best = c;
        unsigned char o = (unsigned char)best;
        put_rec(stdout, &o, 1);
    }
    return 0;
}
