/* Table families (linear, mlp*) of the transfer plan: ./infer_table weights x.bin N > pred.bin
 * x.bin holds N items of d bytes; prints N class indices as raw bytes. N = 0 is start-up only.
 * Format: see families.py. Sums run in feature order, then hidden-unit order, in float32, exactly as
 * families.py's table_predict, so the predictions must agree byte for byte.
 * Compile: gcc -O2 -std=c11 -ffp-contract=off infer_table.c -o infer_table */
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer_table weights x.bin N\n"); return 2; }
    long nw, nx, N = atol(argv[3]);
    int32_t *hd = slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    long want = 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0));
    if (nw != want || C < 1 || C > 255) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W, *b2 = W2 + (long)H * C;
    uint8_t *x = N ? slurp(argv[2], &nx) : NULL;
    if (N < 0 || (N && nx < N * d)) { fprintf(stderr, "bad N\n"); return 2; }
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    uint8_t *out = malloc(N ? N : 1);
    for (long i = 0; i < N; i++) {
        const uint8_t *xi = x + i * d;
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        int best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = c;
        out[i] = (uint8_t)best;
    }
    fwrite(out, 1, N, stdout);
    return 0;
}
