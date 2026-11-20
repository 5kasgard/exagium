/* WalkSAT (Selman, Kautz and Cohen 1994): ./infer weights.bin in.rec N
 * weights: uint32 noise in 1/1000, uint32 max flips per instance. From a random start, repeat: if all clauses hold,
 * stop; pick an unsatisfied clause (the first-found one in a random scan); with probability noise flip a random
 * variable of it, else the one whose flip breaks the fewest satisfied clauses. Deterministic: xorshift seeded per
 * instance by its index. Output: n bytes 0/1 (the best assignment seen if no solution is found). */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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

static uint64_t rs;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw), *x = N ? slurp(argv[2], &nx) : NULL;
    if (nw != 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t noise = w[0] | w[1] << 8 | w[2] << 16 | (uint32_t)w[3] << 24;
    uint32_t maxf = w[4] | w[5] << 8 | w[6] << 16 | (uint32_t)w[7] << 24;
    for (long it = 0; it < N; it++) {
        uint32_t k; const unsigned char *r = next_rec(x, nx, &pos, &k);
        if (k < 3) { fprintf(stderr, "bad instance\n"); return 3; }
        int n = r[0], m = r[1] | r[2] << 8;
        if ((long)k != 3 + 3L * m || n < 1) { fprintf(stderr, "bad instance\n"); return 3; }
        const signed char *L = (const signed char *)(r + 3);
        for (int i = 0; i < 3 * m; i++) {
            int l = L[i];
            if (abs(l) > n) { fprintf(stderr, "bad instance\n"); return 3; }
        }
        unsigned char a[256], best[256];
        int *nt = calloc(m, sizeof(int));  /* true literals per clause */
        rs = 0x9E3779B97F4A7C15ULL ^ (uint64_t)(it + 1) * 0xBF58476D1CE4E5B9ULL;
        for (int v = 0; v < n; v++) a[v] = rnd() & 1;
        int unsat = 0, bestu = m + 1;
        for (int c = 0; c < m; c++) {
            for (int j = 0; j < 3; j++) { int l = L[3 * c + j]; nt[c] += (a[abs(l) - 1] == 1) == (l > 0); }
            unsat += nt[c] == 0;
        }
        for (uint32_t f = 0; ; f++) {
            if (unsat < bestu) { bestu = unsat; memcpy(best, a, n); }
            if (!unsat || f >= maxf) break;
            int start = rnd() % m, c = -1;
            for (int s = 0; s < m; s++) { int cc = (start + s) % m; if (!nt[cc]) { c = cc; break; } }
            int pick;
            if (rnd() % 1000 < noise) pick = abs(L[3 * c + rnd() % 3]) - 1;
            else {
                int bb = 1 << 30; pick = abs(L[3 * c]) - 1;
                for (int j = 0; j < 3; j++) {
                    int v = abs(L[3 * c + j]) - 1, brk = 0;
                    for (int d = 0; d < m; d++)
                        for (int q = 0; q < 3; q++) {
                            int l = L[3 * d + q];
                            if (abs(l) - 1 == v && nt[d] == 1 && (a[v] == 1) == (l > 0)) brk++;
                        }
                    if (brk < bb) { bb = brk; pick = v; }
                }
            }
            for (int d = 0; d < m; d++)
                for (int q = 0; q < 3; q++) {
                    int l = L[3 * d + q];
                    if (abs(l) - 1 != pick) continue;
                    int was = nt[d];
                    nt[d] += ((a[pick] == 1) == (l > 0)) ? -1 : 1;
                    unsat += (nt[d] == 0) - (was == 0);
                }
            a[pick] ^= 1;
        }
        free(nt);
        put_rec(stdout, best, n);
    }
    return 0;
}
