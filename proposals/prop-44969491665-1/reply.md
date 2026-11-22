```hypothesis
We propose "label-lda", a Linear Discriminant Analysis model with per-feature standardization and ridge regularization. LDA assumes Gaussian class-conditional densities with a shared covariance matrix, yielding linear decision boundaries. This generative approach is known to work well on optical digit recognition (e.g., UCI optdigits) and provides a distinct inductive bias from logistic regression. By tuning the ridge parameter on the held-out tuning split (last 20% of train.in), we expect to match or exceed label-linear's dev accuracy (0.9613) while keeping identical inference cost (same linear table lookup). If accuracy does not improve, it would indicate the Gaussian assumption is too restrictive for this dataset.
```

```python
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
    C = card["classes"]
    d = X.shape[1]

    # Split into train (first 80%) and tune (last 20%) in file order
    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    train_idx, tune_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Precompute standardization parameters on training portion
    mu = X_train.mean(axis=0).astype(np.float64)
    sd = X_train.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0

    def standardize(X_raw):
        return (X_raw.astype(np.float64) - mu) / sd

    X_train_std = standardize(X_train)
    X_tune_std = standardize(X_tune)

    # Grid search over ridge regularization
    lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_acc = -1.0
    best_lambda = lambdas[0]

    for lam in lambdas:
        # Compute class means
        class_means = np.zeros((C, d), dtype=np.float64)
        priors = np.zeros(C, dtype=np.float64)
        for c in range(C):
            mask = (y_train == c)
            priors[c] = mask.mean()
            if mask.any():
                class_means[c] = X_train_std[mask].mean(axis=0)

        # Pooled covariance (MLE)
        Sigma = np.zeros((d, d), dtype=np.float64)
        for c in range(C):
            mask = (y_train == c)
            if mask.any():
                centered = X_train_std[mask] - class_means[c]
                Sigma += centered.T @ centered
        Sigma /= n_train = X_train_std.shape[0]

        # Ridge regularization
        if lam > 0:
            Sigma += lam * np.eye(d)
        else:
            # Add tiny ridge for numerical stability when lam=0
            Sigma += 1e-10 * np.eye(d)

        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        # Discriminant weights and biases
        W = Sigma_inv @ class_means.T  # (d, C)
        b = np.zeros(C, dtype=np.float64)
        for c in range(C):
            b[c] = -0.5 * class_means[c] @ Sigma_inv @ class_means[c] + math.log(priors[c] + 1e-12)

        # Evaluate on tune set
        scores = X_tune_std @ W + b
        preds = scores.argmax(axis=1)
        acc = (preds == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_lambda = lam

    # Retrain on all data with best lambda
    mu_all = X.mean(axis=0).astype(np.float64)
    sd_all = X.std(axis=0).astype(np.float64)
    sd_all[sd_all == 0] = 1.0
    X_all_std = (X.astype(np.float64) - mu_all) / sd_all

    class_means_all = np.zeros((C, d), dtype=np.float64)
    priors_all = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        priors_all[c] = mask.mean()
        if mask.any():
            class_means_all[c] = X_all_std[mask].mean(axis=0)

    Sigma_all = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if mask.any():
            centered = X_all_std[mask] - class_means_all[c]
            Sigma_all += centered.T @ centered
    Sigma_all /= X_all_std.shape[0]

    if best_lambda > 0:
        Sigma_all += best_lambda * np.eye(d)
    else:
        Sigma_all += 1e-10 * np.eye(d)

    try:
        Sigma_inv_all = np.linalg.inv(Sigma_all)
    except np.linalg.LinAlgError:
        Sigma_inv_all = np.linalg.pinv(Sigma_all)

    W_all = Sigma_inv_all @ class_means_all.T  # (d, C)
    b_all = np.zeros(C, dtype=np.float64)
    for c in range(C):
        b_all[c] = -0.5 * class_means_all[c] @ Sigma_inv_all @ class_means_all[c] + math.log(priors_all[c] + 1e-12)

    # Build lookup tables T[j, v, c] = ((v - mu_j)/sd_j) * W[j, c]
    T = np.zeros((d, 256, C), dtype=np.float32)
    for j in range(d):
        scale = 1.0 / sd_all[j]
        offset = -mu_all[j] * scale
        for v in range(256):
            z = (v + offset) * scale  # (v - mu)/sd
            T[j, v, :] = z * W_all[j, :]

    b1 = b_all.astype(np.float32)

    # Write weights.bin (table family format with H=0)
    magic = 0x54424C31
    H = 0
    out = struct.pack("<4i", magic, d, H, C)
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