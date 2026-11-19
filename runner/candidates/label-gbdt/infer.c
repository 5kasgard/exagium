/* Gradient-boosted trees for label tasks (queue P3 step 1): ./infer weights.bin in.rec N
 * Sum of the base score and every tree's leaf vector; output the argmax class (first max wins). Format: see train.py. */
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

typedef struct { int16_t f; uint8_t thr, pad; int32_t l, r; } Node;   /* 12 bytes, matches "<hBBii" */

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    int32_t hd[5];
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    memcpy(hd, w, 20);
    int C = hd[1], nt = hd[2], nn = hd[3], nl = hd[4];
    if (hd[0] != 0x31544447 || C < 1 || C > 256 || nt < 0 || nn < 1 || nl < 1 ||
        nw != 20 + 4L * C + 12L * nn + 4L * nt + 4L * nl * C) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *base = malloc(4L * C), *leaf = malloc(4L * nl * C), *s = malloc(4L * C);
    Node *node = malloc(sizeof(Node) * nn);
    int32_t *root = malloc(4L * (nt ? nt : 1));
    long o = 20;
    memcpy(base, w + o, 4L * C); o += 4L * C;
    for (int k = 0; k < nn; k++, o += 12) {
        memcpy(&node[k].f, w + o, 2); node[k].thr = w[o + 2]; memcpy(&node[k].l, w + o + 4, 4); memcpy(&node[k].r, w + o + 8, 4);
        if (node[k].f >= 0 ? (node[k].l < 0 || node[k].l >= nn || node[k].r < 0 || node[k].r >= nn) : (node[k].l < 0 || node[k].l >= nl)) {
            fprintf(stderr, "bad node %d\n", k); return 3;
        }
    }
    memcpy(root, w + o, 4L * nt); o += 4L * nt;
    memcpy(leaf, w + o, 4L * nl * C);
    for (int t = 0; t < nt; t++) if (root[t] < 0 || root[t] >= nn) { fprintf(stderr, "bad root\n"); return 3; }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *q = next_rec(x, nx, &pos, &len);
        for (int c = 0; c < C; c++) s[c] = base[c];
        for (int t = 0; t < nt; t++) {
            int k = root[t], steps = 0;
            while (node[k].f >= 0) {
                if (node[k].f >= (int)len || ++steps > 64) { fprintf(stderr, "bad tree or item %ld\n", i); return 3; }
                k = q[node[k].f] <= node[k].thr ? node[k].l : node[k].r;
            }
            const float *v = leaf + (long)node[k].l * C;
            for (int c = 0; c < C; c++) s[c] += v[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
