/* WiSARD n-tuple classifier for label tasks (queue P3 step 1, the lookup-table family for Q54):
 * ./infer weights.bin in.rec N. Bits from per-column thresholds or values, tuple addresses through a fixed bit map,
 * per-class counts looked up and bleached exactly as train.py's predict(). Format: see train.py. */
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
    int32_t hd[6];
    if (nw < 24) { fprintf(stderr, "bad weights\n"); return 3; }
    memcpy(hd, w, 24);
    int d = hd[1], T = hd[2], n = hd[3], ntup = hd[4], C = hd[5];
    if (hd[0] != 0x31445357 || d < 1 || T < 1 || T > 255 || n < 1 || n > 16 || ntup < 1 || C < 1 || C > 256) {
        fprintf(stderr, "bad weights\n"); return 3;
    }
    long A = 1L << n;
    if (nw != 24 + 2L * d + (long)d * T + 2L * ntup * n + 2L * C * ntup * A) { fprintf(stderr, "bad weights size\n"); return 3; }
    const unsigned char *nv = w + 24, *kind = nv + d, *val = kind + d;
    const uint16_t *map = (const uint16_t *)(val + (long)d * T);   /* offset is even: 24 + 2d + dT is even when dT is */
    const uint16_t *cnt = map + (long)ntup * n;
    if (((24 + 2L * d + (long)d * T) & 1) != 0) { fprintf(stderr, "odd table offset\n"); return 3; }
    for (long i = 0; i < (long)ntup * n; i++) if (map[i] >= d * T) { fprintf(stderr, "bad map\n"); return 3; }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *bit = malloc((size_t)d * T);
    uint16_t *g = malloc(sizeof(uint16_t) * (size_t)C * ntup);
    int *r = malloc(sizeof(int) * C);
    unsigned char *win = malloc(C);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *q = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        for (int j = 0; j < d; j++)
            for (int t = 0; t < T; t++) {
                unsigned char v = val[j * T + t];
                bit[j * T + t] = t < nv[j] && (kind[j] ? q[j] == v : q[j] > v);
            }
        for (int u = 0; u < ntup; u++) {
            long a = 0;
            for (int k = 0; k < n; k++) a |= (long)bit[map[u * n + k]] << k;
            for (int c = 0; c < C; c++) g[c * ntup + u] = cnt[((long)c * ntup + u) * A + a];
        }
        /* bleaching: b = 1 sets the winners; raise b while more than one wins and the top response is above 0 */
        int b = 1, top = 0, nwin = 0;
        for (int c = 0; c < C; c++) { int s = 0; for (int u = 0; u < ntup; u++) s += g[c * ntup + u] >= b; r[c] = s; if (s > top) top = s; }
        for (int c = 0; c < C; c++) { win[c] = r[c] == top; nwin += win[c]; }
        while (nwin > 1 && top > 0) {
            b++;
            int t2 = 0;
            for (int c = 0; c < C; c++) { int s = 0; for (int u = 0; u < ntup; u++) s += g[c * ntup + u] >= b; r[c] = s; if (s > t2) t2 = s; }
            if (t2 == 0) break;
            top = t2; nwin = 0;
            for (int c = 0; c < C; c++) { win[c] = r[c] == top; nwin += win[c]; }
        }
        unsigned char best = 0;
        while (!win[best]) best++;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
