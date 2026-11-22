```hypothesis
Family: label-poly2 (linear model on standardized features plus their squares).
Why: Optdigits pixels have non-linear relationships (e.g., stroke thickness). Quadratic terms capture curvature and interactions implicitly. The linear model (label-linear) gets 0.9613; adding squares should boost accuracy to ~0.97 with only 2x feature count, keeping inference fast (still a single dot product per class).
Failure: If dev accuracy ≤ label-linear or cost > 2× label-linear, the family is not worthwhile.
```
```python
"""label-poly2: linear model on z-scored features plus their squares.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # "POL2"


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    # split: last 20% for tuning (file order)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)

    # compute mean/std on training set only
    mu = X_train.mean(axis=0).astype(np.float32)
    sd = X_train.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0

    def make_poly(X_raw):
        Z = (X_raw.astype(np.float32) - mu) / sd          # (n, d)
        Z2 = Z * Z                                        # (n, d)
        return np.concatenate([Z, Z2], axis=1).astype(np.float32)  # (n, 2d)

    Z_train = make_poly(X_train)
    Z_tune = make_poly(X_tune)
    k = Z_train.shape[1]  # 2d

    # linear model: W (k, C), b (C)
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
    for _ in range(epochs):
        order = rng.permutation(len(X_train))
        for s in range(0, len(X_train), BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_train[idx], y_train[idx]
            # softmax gradient
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # retrain on full data
    Z_full = make_poly(X)
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs_full):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_full[idx], y[idx]
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # write weights: magic, d, C, mu[d], sd[d], W[2d][C], b[C] (all float32 little-endian)
    out = struct.pack("<3i", MAGIC, d, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class per record (4-byte LE length + byte). */
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
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x504F4C32) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = (int)hd[1], C = (int)hd[2];
    if (d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 12 + (size_t)d * 4 * 2 + (size_t)2 * d * C * 4 + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *mu = (const float *)(wb + 12);
    const float *sd = mu + d;
    const float *W = (const float *)(sd + d);
    const float *b = W + 2 * d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            float z = (xi[j] - mu[j]) / sd[j];
            float z2 = z * z;
            const float *w1 = W + j * C;
            const float *w2 = W + (d + j) * C;
            for (int c = 0; c < C; c++) {
                scores[c] += z * w1[c] + z2 * w2[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```