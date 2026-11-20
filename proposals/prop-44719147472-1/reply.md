```hypothesis
We propose "label-lda", a Linear Discriminant Analysis classifier with shrinkage regularization. LDA models each class as a Gaussian with a shared covariance matrix, yielding a linear decision boundary. On optdigits the 64 pixel counts are roughly Gaussian per class, so LDA should outperform logistic regression (label-linear) while keeping the same inference cost: one matrix-vector multiply (64×10) and a bias add. The shrinkage intensity is chosen on the tuning split (last 20% of train.in) to regularize the covariance estimate. If dev accuracy does not exceed label-linear (0.9613), the hypothesis is falsified.
```

```python
"""Linear Discriminant Analysis with shrinkage for label tasks (fixed-size byte items).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: fixed seed, single thread, no randomness except fixed grid search on tuning split.
Weights format: magic 0x4C444131 ('LDA1'), int32 d, int32 C, float32 W[C][d] (row-major), float32 b[C]."""
import json, math, struct, sys

import numpy as np

# Shrinkage grid for tuning (last 20% of train.in in file order)
SHRINKAGE_GRID = [0.01, 0.05, 0.1, 0.2, 0.5]
SEED = 1  # for any potential randomness (none used, but kept for consistency)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_lda(X, y, C, shrinkage):
    """Return W (C, d), b (C) for LDA with given shrinkage."""
    n, d = X.shape
    # Class priors and means
    priors = np.zeros(C, dtype=np.float64)
    means = np.zeros((C, d), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        count = mask.sum()
        priors[c] = count / n
        means[c] = X[mask].mean(axis=0)
    # Pooled covariance
    Sigma = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        diff = X[y == c] - means[c]
        Sigma += diff.T @ diff
    Sigma /= (n - C)
    # Shrinkage: (1-lambda)*Sigma + lambda * (tr(Sigma)/d) * I
    tr = np.trace(Sigma)
    target = (tr / d) * np.eye(d, dtype=np.float64)
    Sigma_shrunk = (1.0 - shrinkage) * Sigma + shrinkage * target
    # Invert
    Sigma_inv = np.linalg.inv(Sigma_shrunk)
    # Weights and biases
    W = (Sigma_inv @ means.T).T  # (C, d)
    b = -0.5 * np.sum(means * (means @ Sigma_inv.T), axis=1) + np.log(priors)
    return W, b


def accuracy(W, b, X, y):
    scores = X @ W.T + b
    preds = np.argmax(scores, axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20% in file order
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]
    # Grid search on tuning split
    best_acc = -1.0
    best_shrink = SHRINKAGE_GRID[0]
    for shrink in SHRINKAGE_GRID:
        W, b = compute_lda(train_X, train_y, C, shrink)
        acc = accuracy(W, b, tune_X, tune_y)
        if acc > best_acc:
            best_acc = acc
            best_shrink = shrink
    # Retrain on all data with best shrinkage
    W_final, b_final = compute_lda(X, y, C, best_shrink)
    # Convert to float32 for storage
    W_f32 = W_final.astype(np.float32)
    b_f32 = b_final.astype(np.float32)
    # Write weights.bin
    magic = 0x4C444131  # 'LDA1'
    header = struct.pack("<3i", magic, d, C)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(W_f32.tobytes())   # row-major: C * d
        f.write(b_f32.tobytes())   # C


if __name__ == "__main__":
    main()
```

```c
/* Linear Discriminant Analysis inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic=0x4C444131, int32 d, int32 C, float32 W[C][d] (row-major), float32 b[C]. */
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x4C444131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 12L + 4L * ((long)C * d + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *W = (const float *)(hd + 3);          // size C*d
    const float *b = W + (long)C * d;                  // size C
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            float xv = (float)xi[j];
            for (int c = 0; c < C; c++) scores[c] += xv * W[(long)c * d + j];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```