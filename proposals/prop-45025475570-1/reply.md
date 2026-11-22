```hypothesis
Family: label-poly (per-pixel quadratic logistic regression).  
Why: Optdigits pixels are ordinal 0..16; a linear model on z-scored values ignores curvature (e.g., mid-range intensities may be more discriminative). Adding v^2 per pixel gives 128 features, still tiny, and lets the linear decision boundary curve in the original space.  
Failure signal: dev accuracy ≤ label-linear (0.9613) or instruction count > 2× label-linear.
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_bytes.shape[1]
    rng = np.random.default_rng(SEED)

    # Polynomial features: for each pixel, standardized v and v^2
    X = X_bytes.astype(np.float64)
    means = np.zeros(2 * d, np.float64)
    stds = np.zeros(2 * d, np.float64)
    for j in range(d):
        v = X[:, j]
        v2 = v * v
        means[2 * j] = v.mean()
        stds[2 * j] = v.std() or 1.0
        means[2 * j + 1] = v2.mean()
        stds[2 * j + 1] = v2.std() or 1.0

    Z = np.zeros((X.shape[0], 2 * d), np.float32)
    for j in range(d):
        v = X[:, j]
        v2 = v * v
        Z[:, 2 * j] = (v - means[2 * j]) / stds[2 * j]
        Z[:, 2 * j + 1] = (v2 - means[2 * j + 1]) / stds[2 * j + 1]

    n, k = Z.shape  # k = 2*d = 128
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            S = z @ W + b
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Save: magic, d, C; then per-pixel mu1,sd1,mu2,sd2 (float32); then W (2*d x C), b (C)
    magic = 0x504F4C59  # 'POLY'
    out = struct.pack("<3i", magic, d, C)
    for j in range(d):
        out += struct.pack("<4f", means[2 * j], stds[2 * j], means[2 * j + 1], stds[2 * j + 1])
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x504F4C59) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (d != 64 || C != 10) { fprintf(stderr, "unexpected d or C\n"); return 3; }
    long expected = 12L + (long)d * 16 + 2L * d * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    float *norm = (float *)(wb + 12);
    float *W = (float *)(wb + 12 + d * 16);
    float *b = (float *)(wb + 12 + d * 16 + 2 * d * C * 4);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            float v = xi[j];
            float v2 = v * v;
            float mu1 = norm[j * 4 + 0];
            float sd1 = norm[j * 4 + 1];
            float mu2 = norm[j * 4 + 2];
            float sd2 = norm[j * 4 + 3];
            float f1 = (v - mu1) / sd1;
            float f2 = (v2 - mu2) / sd2;
            for (int c = 0; c < C; c++) {
                scores[c] += f1 * W[(2 * j) * C + c] + f2 * W[(2 * j + 1) * C + c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```