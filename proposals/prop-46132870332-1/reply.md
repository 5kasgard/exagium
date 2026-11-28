```hypothesis
Family: label-lvq (Learning Vector Quantization). It learns a small set of prototypes per class (K tuned on the tuning split) using LVQ1 on z-scored ordinal features. This should beat label-linear on optdigits because digits form compact, non-linear clusters in pixel space that a few prototypes can capture better than a single hyperplane per class. If it fails to beat linear, it suggests the class-conditional densities are not well-separated by Euclidean distance after normalization, or LVQ gets stuck in poor local optima.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
K_CHOICES = [3, 5, 7]
LR_CHOICES = [0.05, 0.1]
EPOCHS_CHOICES = [10, 15]
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def zscore_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def zscore_apply(X, mean, std):
    return (X - mean) / std

def kmeans_init(X, K, rng, max_iter=10):
    n = X.shape[0]
    if n <= K:
        centroids = X[:K]
        if n < K:
            centroids = np.vstack([centroids, rng.uniform(X.min(), X.max(), (K - n, X.shape[1]))])
        return centroids
    idx = rng.choice(n, K, replace=False)
    centroids = X[idx].copy()
    for _ in range(max_iter):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K)
        for i in range(K):
            mask = labels == i
            if np.any(mask):
                new_centroids[i] = X[mask].mean(axis=0)
                counts[i] = mask.sum()
            else:
                new_centroids[i] = centroids[i]
        empty = counts == 0
        if np.any(empty):
            new_centroids[empty] = X[rng.choice(n, empty.sum(), replace=False)]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids

def lvq_train(X, y, K, C, lr, epochs, rng):
    d = X.shape[1]
    prototypes = np.zeros((C * K, d), dtype=np.float32)
    proto_class = np.zeros(C * K, dtype=np.int64)
    for c in range(C):
        class_mask = y == c
        if np.any(class_mask):
            proto = kmeans_init(X[class_mask], K, rng)
        else:
            proto = rng.uniform(-1, 1, (K, d))
        prototypes[c*K:(c+1)*K] = proto
        proto_class[c*K:(c+1)*K] = c

    n = X.shape[0]
    for epoch in range(epochs):
        lr_epoch = lr * (1 - epoch / epochs)
        order = rng.permutation(n)
        for i in order:
            x = X[i]
            yi = y[i]
            dists = np.sum((prototypes - x) ** 2, axis=1)
            best = np.argmin(dists)
            if proto_class[best] == yi:
                prototypes[best] += lr_epoch * (x - prototypes[best])
            else:
                prototypes[best] -= lr_epoch * (x - prototypes[best])
    return prototypes, proto_class

def evaluate(X, y, prototypes, proto_class):
    dists = np.sum((X[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
    preds = proto_class[np.argmin(dists, axis=1)]
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_all.shape[1]

    n = X_all.shape[0]
    n_tune = n // 5
    X_train, X_tune = X_all[:-n_tune], X_all[-n_tune:]
    y_train, y_tune = y_all[:-n_tune], y_all[-n_tune:]

    rng = np.random.default_rng(SEED)

    best_acc = -1.0
    best_params = None
    best_mean = best_std = best_protos = best_pclass = None

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for epochs in EPOCHS_CHOICES:
                mean, std = zscore_fit(X_train)
                X_tr_norm = zscore_apply(X_train, mean, std).astype(np.float32)
                X_tu_norm = zscore_apply(X_tune, mean, std).astype(np.float32)

                protos, pclass = lvq_train(X_tr_norm, y_train, K, C, lr, epochs, rng)
                acc = evaluate(X_tu_norm, y_tune, protos, pclass)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K, lr, epochs)
                    best_mean, best_std = mean, std
                    best_protos, best_pclass = protos, pclass

    K, lr, epochs = best_params
    mean, std = zscore_fit(X_all)
    X_all_norm = zscore_apply(X_all, mean, std).astype(np.float32)
    protos, pclass = lvq_train(X_all_norm, y_all, K, C, lr, epochs, rng)

    magic = 0x4C565131  # "LVQ1"
    out = struct.pack("<4i", magic, d, K, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 256 || d < 1 || d > 1024) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16 + 4L * d * 2 + 4L * (long)K * C * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *protos = std + d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in = malloc(sizeof(float) * d);
    float *dists = malloc(sizeof(float) * K * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            in[j] = (xi[j] - mean[j]) / std[j];
        }
        for (int p = 0; p < K * C; p++) {
            const float *proto = protos + (long)p * d;
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = in[j] - proto[j];
                sum += diff * diff;
            }
            dists[p] = sum;
        }
        int best_p = 0;
        for (int p = 1; p < K * C; p++) {
            if (dists[p] < dists[best_p]) best_p = p;
        }
        unsigned char best_class = best_p / K;
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```