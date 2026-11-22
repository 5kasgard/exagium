```hypothesis
Family: label-lda (Linear Discriminant Analysis with shrinkage).
Why it should do well: Optdigits pixels are ordinal counts; LDA assumes Gaussian class-conditionals with shared covariance, a good match. Shrinkage regularizes the covariance for the small training set (2867 samples), improving over unregularized LDA and logistic regression. Inference is a single linear layer (same 64×10 multiply-add as label-linear) so CPU cost stays minimal.
Failure signal: Dev accuracy ≤ 0.9613 (label-linear) or instructions/item > 5600 (10% above label-linear).
```

```python
import json, math, struct, sys

import numpy as np

SHRINKAGE_GRID = [0.0, 0.1, 0.5, 1.0, 10.0, 100.0]
SEED = 1


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
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]

    # Split: last 20% for tuning (file order)
    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_shrinkage = 0.0
    best_acc = -1.0

    for shrinkage in SHRINKAGE_GRID:
        # Scale on training split
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd[sd == 0] = 1.0
        X_tr_scaled = (X_train - mu) / sd
        X_tu_scaled = (X_tune - mu) / sd

        # Class means
        means = np.zeros((C, d), dtype=np.float64)
        for c in range(C):
            means[c] = X_tr_scaled[y_train == c].mean(axis=0)

        # Pooled covariance
        cov = np.zeros((d, d), dtype=np.float64)
        for c in range(C):
            Xc = X_tr_scaled[y_train == c]
            diff = Xc - means[c]
            cov += diff.T @ diff
        cov /= (len(X_train) - C)

        # Shrinkage
        if shrinkage > 0:
            trace = np.trace(cov)
            cov = (1 - shrinkage) * cov + shrinkage * (trace / d) * np.eye(d)

        # Solve for weights
        W = np.linalg.solve(cov, means.T)  # d x C
        prior = np.bincount(y_train, minlength=C) / len(X_train)
        bias = -0.5 * np.diag(means @ W) + np.log(prior)

        # Evaluate
        scores = X_tu_scaled @ W + bias
        pred = scores.argmax(axis=1)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_shrinkage = shrinkage

    # Retrain on full data
    mu_full = X.mean(axis=0)
    sd_full = X.std(axis=0)
    sd_full[sd_full == 0] = 1.0
    X_scaled = (X - mu_full) / sd_full

    means = np.zeros((C, d), dtype=np.float64)
    for c in range(C):
        means[c] = X_scaled[y == c].mean(axis=0)

    cov = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        Xc = X_scaled[y == c]
        diff = Xc - means[c]
        cov += diff.T @ diff
    cov /= (n - C)

    if best_shrinkage > 0:
        trace = np.trace(cov)
        cov = (1 - best_shrinkage) * cov + best_shrinkage * (trace / d) * np.eye(d)

    W = np.linalg.solve(cov, means.T)
    prior = np.bincount(y, minlength=C) / n
    bias = -0.5 * np.diag(means @ W) + np.log(prior)

    # Build lookup tables T[j][v][c] = scaled(v) * W[j][c]
    T = np.zeros((d, 256, C), dtype=np.float32)
    for j in range(d):
        scaled_vals = (np.arange(256, dtype=np.float32) - mu_full[j]) / sd_full[j]
        T[j, :, :] = scaled_vals[:, None] * W[j, :].astype(np.float32)

    b1 = bias.astype(np.float32)

    # Write weights.bin: magic 0x4C444131 ("LDA1"), d, H=0, C
    magic = 0x4C444131
    H = 0
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) acc[c] = b1[c];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```