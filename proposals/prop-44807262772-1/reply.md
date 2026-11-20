```hypothesis
Family: label-poly2 (polynomial logistic regression degree 2).  
Why: Optdigits pixels exhibit local correlations; quadratic features capture pairwise interactions without the overhead of trees or deep nets. Expected to beat label-linear (0.9613) and approach label-mlp16 (0.9770) at a fraction of GBDT's instruction cost.  
Failure signal: dev accuracy ≤ 0.9613 or instructions/item > 50 000 (indicating excessive compute for the gain).
```

```python
"""label-poly2: polynomial logistic regression (degree 2) for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x504F4C32, d, num_feat, C; float32 mean[d], std[d], W[num_feat*C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.05
L2 = 1e-4
BATCH = 64
EPOCHS = 200


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    # Normalize per feature (z-score)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std

    # Build polynomial features up to degree 2 (including squares)
    n = X_norm.shape[0]
    num_quad = d * (d + 1) // 2
    num_feat = d + num_quad
    Phi = np.zeros((n, num_feat), dtype=np.float32)
    Phi[:, :d] = X_norm
    idx = d
    for i in range(d):
        for j in range(i, d):
            Phi[:, idx] = X_norm[:, i] * X_norm[:, j]
            idx += 1

    # Train logistic regression with SGD
    rng = np.random.default_rng(SEED)
    W = np.zeros((num_feat, C), dtype=np.float32)
    bias = np.zeros(C, dtype=np.float32)
    lr = LR

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            Xb = Phi[idx_batch]
            yb = y[idx_batch]
            scores = Xb @ W + bias
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_W = (Xb.T @ probs) / len(yb) + L2 * W
            grad_b = probs.mean(axis=0)
            W -= lr * grad_W
            bias -= lr * grad_b
        lr *= 0.99

    # Save weights
    magic = 0x504F4C32  # "POL2"
    out = struct.pack("<4i", magic, d, num_feat, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output record: one byte class index.
 * Weights format: magic 0x504F4C32, d, num_feat, C; float32 mean[d], std[d], W[num_feat*C], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], num_feat = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (2L * d + (long)num_feat * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *W = std + d;
    const float *bias = W + (long)num_feat * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *norm = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) norm[j] = (xi[j] - mean[j]) / std[j];

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        /* linear terms */
        for (int j = 0; j < d; j++) {
            float xj = norm[j];
            const float *w = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += xj * w[c];
        }

        /* quadratic terms (i <= j) */
        long base = d;
        for (int i1 = 0; i1 < d; i1++) {
            float xi = norm[i1];
            for (int i2 = i1; i2 < d; i2++) {
                float prod = xi * norm[i2];
                const float *w = W + (long)base * C;
                for (int c = 0; c < C; c++) scores[c] += prod * w[c];
                base++;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```