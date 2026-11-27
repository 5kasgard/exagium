```hypothesis
Family: label-rbf (Radial Basis Function network with k-means centers).
Why: Optdigits pixels have spatial structure; RBF kernels capture non-linear boundaries better than linear models. K=64 centers match input dimension, gamma tuned on 20% holdout. Inference cost: K*d=4096 mult-adds + K exps + K*C=640 mult-adds, far below label-mlp16.
Failure: Dev accuracy ≤ 0.9613 (label-linear) shows RBF adds no value here.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
K = 64
GAMMA_CANDIDATES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
LAMBDA = 1e-4
MAX_KMEANS_ITER = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, rng, max_iter=100):
    n, d = X.shape
    centers = np.zeros((K, d), dtype=X.dtype)
    idx = rng.integers(n)
    centers[0] = X[idx]
    for i in range(1, K):
        diff = X[:, None, :] - centers[:i][None, :, :]
        dists = np.sum(diff * diff, axis=2)
        min_dists = np.min(dists, axis=1)
        probs = min_dists / min_dists.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
    for _ in range(max_iter):
        diff = X[:, None, :] - centers[None, :, :]
        dists = np.sum(diff * diff, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        counts = np.zeros(K, dtype=int)
        for j in range(n):
            c = labels[j]
            new_centers[c] += X[j]
            counts[c] += 1
        for k in range(K):
            if counts[k] > 0:
                new_centers[k] /= counts[k]
            else:
                new_centers[k] = X[rng.integers(n)]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers


def rbf_features(X, centers, gamma):
    diff = X[:, None, :] - centers[None, :, :]
    dists = np.sum(diff * diff, axis=2)
    return np.exp(-gamma * dists, dtype=np.float32)


def ridge_regression(Phi, Y, lam):
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1], dtype=np.float32)
    B = Phi.T @ Y
    return np.linalg.solve(A, B)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n, d = X.shape

    split_idx = int(n * 0.8)
    idx = np.arange(n)
    train_idx, tune_idx = idx[:split_idx], idx[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    rng = np.random.default_rng(SEED)

    # Standardize on train_80
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_train_s = (X_train - mean) / std
    X_tune_s = (X_tune - mean) / std

    Y_train = np.eye(C, dtype=np.float32)[y_train]
    Y_tune = np.eye(C, dtype=np.float32)[y_tune]

    centers = kmeans(X_train_s.astype(np.float32), K, rng, MAX_KMEANS_ITER)

    best_gamma = GAMMA_CANDIDATES[0]
    best_acc = -1.0
    for gamma in GAMMA_CANDIDATES:
        Phi_train = rbf_features(X_train_s.astype(np.float32), centers, gamma)
        Phi_tune = rbf_features(X_tune_s.astype(np.float32), centers, gamma)
        Phi_train_b = np.hstack([Phi_train, np.ones((Phi_train.shape[0], 1), dtype=np.float32)])
        Phi_tune_b = np.hstack([Phi_tune, np.ones((Phi_tune.shape[0], 1), dtype=np.float32)])
        W_full = ridge_regression(Phi_train_b, Y_train, LAMBDA)
        scores = Phi_tune_b @ W_full
        preds = np.argmax(scores, axis=1)
        acc = np.mean(preds == y_tune)
        if acc > best_acc:
            best_acc = acc
            best_gamma = gamma

    # Retrain on all data
    mean_all = X.mean(axis=0).astype(np.float32)
    std_all = X.std(axis=0).astype(np.float32)
    std_all[std_all == 0] = 1.0
    X_all_s = (X - mean_all) / std_all
    Y_all = np.eye(C, dtype=np.float32)[y]

    centers = kmeans(X_all_s.astype(np.float32), K, rng, MAX_KMEANS_ITER)
    Phi_all = rbf_features(X_all_s.astype(np.float32), centers, best_gamma)
    Phi_all_b = np.hstack([Phi_all, np.ones((n, 1), dtype=np.float32)])
    W_full = ridge_regression(Phi_all_b, Y_all, LAMBDA)
    W = W_full[:-1, :]
    b = W_full[-1, :]

    magic = 0x52424631
    out = struct.pack("<4i", magic, d, K, C)
    out += mean_all.astype("<f4").tobytes()
    out += std_all.astype("<f4").tobytes()
    out += centers.astype("<f4").tobytes()
    out += struct.pack("<f", best_gamma)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x52424631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 64 || C != 10 || K != 64) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L + 4L * (2L * d + (long)K * d + 1 + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *mean = p; p += d;
    const float *std = p; p += d;
    const float *centers = p; p += (long)K * d;
    const float gamma = *p++; 
    const float *W = p; p += (long)K * C;
    const float *b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *s = malloc(sizeof(float) * C);
    float *phi = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int u = 0; u < K; u++) {
            float dist2 = 0.0f;
            const float *c = centers + (long)u * d;
            for (int j = 0; j < d; j++) {
                float xn = (xi[j] - mean[j]) / std[j];
                float diff = xn - c[j];
                dist2 += diff * diff;
            }
            phi[u] = expf(-gamma * dist2);
        }

        for (int c = 0; c < C; c++) s[c] = b[c];
        for (int u = 0; u < K; u++) {
            float ph = phi[u];
            const float *wu = W + (long)u * C;
            for (int c = 0; c < C; c++) s[c] += ph * wu[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```