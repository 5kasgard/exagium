```hypothesis
Family: label-lda (Linear Discriminant Analysis with shrinkage).
Why: LDA models each class as Gaussian with a shared covariance, yielding a linear classifier that often excels on small, approximately Gaussian-distributed datasets like optdigits. With a tuned shrinkage parameter on the tuning split, it can surpass logistic regression (label-linear) while keeping the same O(d·C) inference cost.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or inference instructions per item exceed label-linear by >10%.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 0.1, 1.0]


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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    # Use raw pixel values (0..16) as floats
    X = X_bytes.astype(np.float64)

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    best_lambda = LAMBDA_GRID[0]
    best_acc = -1.0

    for lam in LAMBDA_GRID:
        # Class means
        mu = np.zeros((C, d), dtype=np.float64)
        for c in range(C):
            mu[c] = X_tr[y_tr == c].mean(axis=0) if np.any(y_tr == c) else 0.0

        # Pooled covariance
        Sigma = np.zeros((d, d), dtype=np.float64)
        for c in range(C):
            Xc = X_tr[y_tr == c]
            if len(Xc) > 1:
                diff = Xc - mu[c]
                Sigma += diff.T @ diff
        Sigma /= (len(X_tr) - C)
        # Shrinkage
        Sigma += lam * np.eye(d)

        # Regularized inverse
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            continue

        # Discriminant weights and biases
        W = mu @ Sigma_inv.T  # C x d
        bias = -0.5 * np.sum(mu * W, axis=1) + np.log(np.bincount(y_tr, minlength=C) / len(X_tr))

        # Validation accuracy
        scores_va = X_va @ W.T + bias
        pred_va = scores_va.argmax(axis=1)
        acc = (pred_va == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_lambda = lam

    # Retrain on all data with best lambda
    mu = np.zeros((C, d), dtype=np.float64)
    for c in range(C):
        mu[c] = X[y == c].mean(axis=0) if np.any(y == c) else 0.0

    Sigma = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        Xc = X[y == c]
        if len(Xc) > 1:
            diff = Xc - mu[c]
            Sigma += diff.T @ diff
    Sigma /= (n - C)
    Sigma += best_lambda * np.eye(d)

    Sigma_inv = np.linalg.inv(Sigma)
    W = mu @ Sigma_inv.T
    bias = -0.5 * np.sum(mu * W, axis=1) + np.log(np.bincount(y, minlength=C) / n)

    # Export weights: magic, d, C, then W (C*d float32), bias (C float32)
    magic = 0x4C444131  # "LDA1"
    out = struct.pack("<3i", magic, d, C)
    out += W.astype(np.float32).tobytes()
    out += bias.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Linear Discriminant Analysis inference: ./infer weights.bin in.rec N */
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
    if (nw < 12 || hd[0] != 0x4C444131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 12 + 4L * ((long)C * d + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 3);
    const float *bias = W + (long)C * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char *inbuf = malloc(d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            float xj = (float)xi[j];
            for (int c = 0; c < C; c++) scores[c] += xj * W[(long)c * d + j];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```