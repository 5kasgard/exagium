```hypothesis
We propose "label-lda", a Linear Discriminant Analysis classifier with shrinkage regularization.
LDA is well-suited for this task because the 8×8 pixel counts are approximately Gaussian per class
and share a common covariance structure. By using the same per-byte lookup table format as
label-linear, inference cost is identical (same number of operations), but the generative modeling
of class-conditional densities often yields higher accuracy than discriminative logistic regression
on small datasets. We select the shrinkage intensity on the held-out tuning split (last 20% of train).
If dev accuracy does not exceed label-linear's 0.9613, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
SHRINKAGE_GRID = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]


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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # compute class statistics on training portion
    class_counts = np.bincount(y_train, minlength=C)
    class_means = np.zeros((C, d), dtype=np.float64)
    scatter = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        Xc = X_train[y_train == c]
        mu_c = Xc.mean(axis=0)
        class_means[c] = mu_c
        scatter += (Xc - mu_c).T @ (Xc - mu_c)

    pooled_cov = scatter / (split - C)  # unbiased pooled covariance
    diag_cov = np.diag(np.diag(pooled_cov))

    best_lambda = 0.0
    best_acc = -1.0
    best_W = None
    best_b = None

    for lam in SHRINKAGE_GRID:
        Sigma_reg = (1.0 - lam) * pooled_cov + lam * diag_cov
        Sigma_reg += 1e-10 * np.eye(d)  # numerical stability
        try:
            Sigma_inv = np.linalg.inv(Sigma_reg)
        except np.linalg.LinAlgError:
            continue
        W = Sigma_inv @ class_means.T  # d x C
        priors = class_counts / split
        b = np.zeros(C, dtype=np.float64)
        for c in range(C):
            b[c] = -0.5 * class_means[c] @ W[:, c] + math.log(priors[c])

        scores = X_tune.astype(np.float64) @ W + b
        pred = np.argmax(scores, axis=1)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_lambda = lam
            best_W = W
            best_b = b

    # retrain on all data with best lambda
    class_counts = np.bincount(y, minlength=C)
    class_means = np.zeros((C, d), dtype=np.float64)
    scatter = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        class_means[c] = mu_c
        scatter += (Xc - mu_c).T @ (Xc - mu_c)

    pooled_cov = scatter / (n - C)
    diag_cov = np.diag(np.diag(pooled_cov))
    Sigma_reg = (1.0 - best_lambda) * pooled_cov + best_lambda * diag_cov
    Sigma_reg += 1e-10 * np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma_reg)
    W = Sigma_inv @ class_means.T  # d x C
    priors = class_counts / n
    b = np.zeros(C, dtype=np.float64)
    for c in range(C):
        b[c] = -0.5 * class_means[c] @ W[:, c] + math.log(priors[c])

    # build per-byte lookup tables: T[j][v][c] = v * W[j, c] for v in 0..255
    T = np.zeros((d, 256, C), dtype=np.float32)
    for j in range(d):
        T[j, :, :] = (np.arange(256, dtype=np.float32)[:, None] * W[j, :].astype(np.float32))

    b1 = b.astype(np.float32)

    # write weights in the same format as label-linear (magic, d, H=0, C)
    out = struct.pack("<4i", 0x54424C31, d, 0, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```