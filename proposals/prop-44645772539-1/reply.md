```hypothesis
Propose "label-lda": Linear Discriminant Analysis with shared covariance and per-feature z-scoring.
LDA models each class as Gaussian with a common covariance, yielding a linear classifier that exploits feature correlations.
It is deterministic, has one regularization hyperparameter (tuned on the last 20% of train), and inference is a single matrix-vector multiply per item (64*10 FLOPs).
Expected to beat label-linear (logistic regression) because the shared covariance better separates the digit classes.
Failure: dev accuracy ≤ label-linear (0.9613) or CPU cost per item significantly higher.
```
```python
import json, math, struct, sys

import numpy as np

# hyperparameter grid for regularization (lambda = reg * trace/d)
REG_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SEED = 0


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def standardize(X, mean, std):
    return (X - mean) / std


def lda_train(X, y, C, reg):
    """X: n x d standardized, y: n labels. Returns w (C x d), bias (C)."""
    n, d = X.shape
    # class priors
    priors = np.bincount(y, minlength=C) / n
    log_priors = np.log(priors)
    # class means
    means = np.zeros((C, d), dtype=np.float64)
    for c in range(C):
        means[c] = X[y == c].mean(axis=0)
    # pooled covariance
    cov = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        Xc = X[y == c] - means[c]
        cov += Xc.T @ Xc
    cov /= (n - C)
    # regularization
    trace = np.trace(cov)
    cov += reg * (trace / d) * np.eye(d)
    # invert
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    # weights and biases
    w = means @ inv_cov  # C x d
    bias = -0.5 * np.sum(means * w, axis=1) + log_priors
    return w.astype(np.float32), bias.astype(np.float32)


def accuracy(X, y, w, bias):
    scores = X @ w.T + bias
    pred = np.argmax(scores, axis=1)
    return np.mean(pred == y)


def main():
    np.random.seed(SEED)
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # split: last 20% for tuning
    n = X.shape[0]
    tune_size = int(0.2 * n)
    train_size = n - tune_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_tune, y_tune = X[train_size:], y[train_size:]

    # compute mean/std on training split
    mean = X_train.mean(axis=0).astype(np.float64)
    std = X_train.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0

    X_train_s = standardize(X_train.astype(np.float64), mean, std)
    X_tune_s = standardize(X_tune.astype(np.float64), mean, std)

    best_reg = REG_GRID[0]
    best_acc = -1.0
    for reg in REG_GRID:
        w, bias = lda_train(X_train_s, y_train, C, reg)
        acc = accuracy(X_tune_s, y_tune, w, bias)
        if acc > best_acc:
            best_acc = acc
            best_reg = reg

    # retrain on all data with best_reg
    mean_all = X.mean(axis=0).astype(np.float64)
    std_all = X.std(axis=0).astype(np.float64)
    std_all[std_all == 0] = 1.0
    X_all_s = standardize(X.astype(np.float64), mean_all, std_all)
    w, bias = lda_train(X_all_s, y, C, best_reg)

    # pack weights: magic, d, C, mean[d], std[d], w[C][d], bias[C]
    magic = 0x3141444C  # "LDA1" little-endian
    out = struct.pack("<3i", magic, d, C)
    out += mean_all.astype("<f4").tobytes()
    out += std_all.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x3141444C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (d != 64 || C != 10) { fprintf(stderr, "unexpected d=%d C=%d\n", d, C); return 3; }
    long expected = 12L + 4L * d + 4L * d + 4L * C * d + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 3);
    const float *std  = mean + d;
    const float *w    = std + d;        // size C * d
    const float *bias = w + (long)C * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            float z = (xi[j] - mean[j]) / std[j];
            if (z != z) z = 0.0f; // guard against NaN
            for (int c = 0; c < C; c++) {
                scores[c] += z * w[(long)c * d + j];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```