```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression).
Reason: Optdigits pixels exhibit local correlations (strokes, loops) that linear models miss; pairwise products capture these interactions. With only 64 input features, the expanded 2080 features are still tiny, enabling exact inference with a simple double loop and no approximation overhead.
Failure signal: Dev accuracy ≤ 0.9613 (label-linear) or instruction count > 2× label-linear.
```

```python
"""Degree-2 polynomial logistic regression for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x504F4C32, d, C, D; float32 means[d], stds[d], W[D][C], b[C]."""
import json, math, struct, sys
from itertools import combinations_with_replacement

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
    D = d * (d + 1) // 2

    rng = np.random.default_rng(SEED)

    # z-score per feature
    mean = X.mean(axis=0).astype(np.float64)
    std = X.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0
    Z = ((X - mean) / std).astype(np.float32)  # (n, d)

    # polynomial features degree 2
    P = np.empty((X.shape[0], D), dtype=np.float32)
    col = 0
    for i, j in combinations_with_replacement(range(d), 2):
        P[:, col] = Z[:, i] * Z[:, j]
        col += 1

    n, _ = P.shape
    # init weights
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            p_batch, y_batch = P[idx], y[idx]
            logits = p_batch @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(y_batch)), y_batch] -= 1.0
            grad = (probs / len(y_batch)).astype(np.float32)  # (batch, C)

            gW = p_batch.T @ grad + L2 * W
            gb = grad.sum(axis=0)

            t += 1
            for param, gr, (m, v) in zip(params, [gW, gb], state):
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    out = struct.pack("<4i", MAGIC, d, C, D)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Degree-2 polynomial logistic regression inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class 0..9). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], D = hd[3];
    if (d <= 0 || C <= 0 || C > 256 || D != d * (d + 1) / 2) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * d + 4L * d + 4L * D * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *W    = std + d;
    const float *b    = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *norm = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) norm[j] = (xi[j] - mean[j]) / std[j];

        for (int c = 0; c < C; c++) scores[c] = b[c];

        int idx = 0;
        for (int i1 = 0; i1 < d; i1++) {
            for (int i2 = i1; i2 < d; i2++) {
                float prod = norm[i1] * norm[i2];
                const float *wrow = W + (long)idx * C;
                for (int c = 0; c < C; c++) scores[c] += prod * wrow[c];
                idx++;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```