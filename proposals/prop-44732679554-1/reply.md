```hypothesis
Family: label-lda (Linear Discriminant Analysis with shrinkage).
Why: Optdigits features are low-dimensional (64) and roughly Gaussian per class; LDA is a strong linear baseline that often matches logistic regression with fewer hyperparameters. Shrinkage stabilizes covariance estimation on small datasets.
Failure: If dev accuracy <= 0.9613 (label-linear) or inference cost exceeds label-linear significantly.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
ALPHAS = np.logspace(-6, 2, 9)  # shrinkage grid


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def lda_train(X, y, alpha, C):
    """X: (n, d) float64, y: (n,) int. Returns W (C, d), b (C) in original feature space."""
    n, d = X.shape
    # class stats
    priors = np.zeros(C, np.float64)
    means = np.zeros((C, d), np.float64)
    for c in range(C):
        mask = (y == c)
        priors[c] = mask.mean()
        means[c] = X[mask].mean(axis=0)
    # pooled covariance
    centered = X - means[y]
    S = (centered.T @ centered) / (n - C)
    # shrinkage: S_reg = S + alpha * I
    S_reg = S + alpha * np.eye(d)
    # solve for each class: W_lda = S_reg^{-1} @ means.T  -> (d, C)
    W_lda = np.linalg.solve(S_reg, means.T).T  # (C, d)
    # biases in standardized space: b_lda = -0.5 * diag(means @ W_lda.T) + log(priors)
    b_lda = -0.5 * np.sum(means * W_lda, axis=1) + np.log(priors)
    return W_lda, b_lda


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # split: last 20% for tuning
    n_tune = int(0.2 * n)
    n_train = n - n_tune
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]

    # standardize on training portion
    mu = X_train.mean(axis=0).astype(np.float64)
    sigma = X_train.std(axis=0).astype(np.float64)
    sigma[sigma == 0] = 1.0
    Z_train = (X_train - mu) / sigma
    Z_tune = (X_tune - mu) / sigma

    # tune alpha
    best_alpha = ALPHAS[0]
    best_acc = -1.0
    for alpha in ALPHAS:
        W_lda, b_lda = lda_train(Z_train, y_train, alpha, C)
        # convert to original space
        W = W_lda / sigma  # (C, d)
        b = b_lda - mu @ W.T
        # evaluate on tune
        scores = Z_tune @ W.T + b_lda  # using standardized features
        preds = np.argmax(scores, axis=1)
        acc = (preds == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # retrain on full data with best_alpha
    mu_full = X.mean(axis=0).astype(np.float64)
    sigma_full = X.std(axis=0).astype(np.float64)
    sigma_full[sigma_full == 0] = 1.0
    Z_full = (X - mu_full) / sigma_full
    W_lda, b_lda = lda_train(Z_full, y, best_alpha, C)
    W = (W_lda / sigma_full).astype(np.float32)  # (C, d)
    b = (b_lda - mu_full @ W_lda.T / sigma_full).astype(np.float32)  # (C,)

    # write weights: magic, d, C, reserved; then W (C*d), then b (C)
    magic = 0x4C444131  # "LDA1"
    out = struct.pack("<4i", magic, d, C, 0)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* LDA family for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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
    if (nw < 16 || hd[0] != 0x4C444131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)C * d + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)C * d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *wc = W + (long)c * d;
            for (int j = 0; j < d; j++) sum += xi[j] * wc[j];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```