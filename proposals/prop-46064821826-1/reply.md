```hypothesis
Family: label-lda (Linear Discriminant Analysis with shrinkage regularization).
Why: Optdigits features are ordinal pixel counts; LDA models class-conditional Gaussians with shared covariance, which suits this data well. Shrinkage stabilizes the covariance estimate for the modest sample size. The resulting linear classifier requires only O(d·C) operations per item (64·10=640 multiply-adds), making it faster than label-mlp16 while typically exceeding label-linear accuracy.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or inference cost > 5058 Ir/item (label-linear), the model does not meet its goals.
```

```python
"""Linear Discriminant Analysis with shrinkage for label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (no randomness), single thread, stdlib + numpy only.
Exports: magic, d, C; float32 mean[d], std[d], W[d*C], b[C]."""
import json, math, struct, sys

import numpy as np


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

    # Tuning split: last 20% in file order
    val_size = max(1, n // 5)
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Standardize on training portion
    mu = X_train.mean(axis=0).astype(np.float64)
    sigma = X_train.std(axis=0).astype(np.float64)
    sigma[sigma == 0] = 1.0

    def standardize(X_raw):
        return (X_raw.astype(np.float64) - mu) / sigma

    X_train_std = standardize(X_train)
    X_val_std = standardize(X_val)

    # Grid search for shrinkage
    best_shrink = 0.0
    best_acc = -1.0
    for shrink in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Class means
        class_means = np.zeros((C, d), dtype=np.float64)
        priors = np.zeros(C, dtype=np.float64)
        for c in range(C):
            mask = (y_train == c)
            if not np.any(mask):
                continue
            class_means[c] = X_train_std[mask].mean(axis=0)
            priors[c] = mask.sum()
        priors /= priors.sum()

        # Pooled covariance
        centered = X_train_std - class_means[y_train]
        Sigma = centered.T @ centered / (len(y_train) - C)
        # Shrinkage
        Sigma_shrunk = (1 - shrink) * Sigma + shrink * np.diag(np.diag(Sigma))
        # Invert
        try:
            Sigma_inv = np.linalg.inv(Sigma_shrunk)
        except np.linalg.LinAlgError:
            continue

        # Discriminant coefficients
        W = Sigma_inv @ class_means.T  # d x C
        # Bias: -0.5 * mu_c^T Sigma^{-1} mu_c + log(prior)
        quad = np.einsum('ci,ij,cj->c', class_means, Sigma_inv, class_means)
        b = -0.5 * quad + np.log(priors)

        # Validate
        scores = X_val_std @ W + b
        preds = scores.argmax(axis=1)
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_shrink = shrink

    # Retrain on full data with best shrinkage
    mu_full = X.mean(axis=0).astype(np.float64)
    sigma_full = X.std(axis=0).astype(np.float64)
    sigma_full[sigma_full == 0] = 1.0
    X_std_full = (X.astype(np.float64) - mu_full) / sigma_full

    class_means = np.zeros((C, d), dtype=np.float64)
    priors = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        class_means[c] = X_std_full[mask].mean(axis=0)
        priors[c] = mask.sum()
    priors /= priors.sum()

    centered = X_std_full - class_means[y]
    Sigma = centered.T @ centered / (n - C)
    Sigma_shrunk = (1 - best_shrink) * Sigma + best_shrink * np.diag(np.diag(Sigma))
    Sigma_inv = np.linalg.inv(Sigma_shrunk)

    W = Sigma_inv @ class_means.T  # d x C
    quad = np.einsum('ci,ij,cj->c', class_means, Sigma_inv, class_means)
    b = -0.5 * quad + np.log(priors)

    # Export as float32 little-endian
    MAGIC = 0x4C444131  # "LDA1"
    out = struct.pack("<4i", MAGIC, d, C, 0)
    out += mu_full.astype("<f4").tobytes()
    out += sigma_full.astype("<f4").tobytes()
    # W is d x C, stored column-major (class-contiguous) for cache-friendly inference
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Linear Discriminant Analysis inference: ./infer weights.bin in.rec N
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
    if (nw < 16 || hd[0] != 0x4C444131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * (2L * d + (long)d * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *W    = std + d;        // size d*C, column-major: W[j + c*d]
    const float *b    = W + (long)d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_std = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            float val = xi[j];
            float s = std[j];
            x_std[j] = (val - mean[j]) / (s != 0.0f ? s : 1.0f);
        }

        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *wc = W + (long)c * d;
            for (int j = 0; j < d; j++) sum += x_std[j] * wc[j];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```