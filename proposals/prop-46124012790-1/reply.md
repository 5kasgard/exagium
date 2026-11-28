```hypothesis
Family: label-proto (prototype-based classifier with per-class k-means centroids).
Why: Digits have distinct visual prototypes; multiple centroids per class capture intra-class variation (e.g., different writing styles) while keeping inference fast (distance to 10*K vectors). Z-scored pixels make Euclidean distance meaningful.
Failure: If dev accuracy ≤ label-linear (0.9613) or inference cost exceeds label-linear significantly.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
K_CANDIDATES = [1, 2, 3, 4, 5]
MAX_KMEANS_ITER = 100
TUNING_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def zscore_params(X):
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    return mu, sd

def apply_zscore(X, mu, sd):
    return (X.astype(np.float32) - mu) / sd

def kmeans_class(Xc, K, rng):
    n, d = Xc.shape
    if n == 0:
        return np.zeros((K, d), np.float32)
    if n <= K:
        cents = Xc.astype(np.float32)
        if n < K:
            cents = np.vstack([cents, np.zeros((K - n, d), np.float32)])
        return cents
    idx = rng.choice(n, K, replace=False)
    cents = Xc[idx].astype(np.float32).copy()
    for _ in range(MAX_KMEANS_ITER):
        dists = ((Xc[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        assign = dists.argmin(axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(K, np.int32)
        for i in range(n):
            c = assign[i]
            new_cents[c] += Xc[i]
            counts[c] += 1
        for k in range(K):
            if counts[k] > 0:
                new_cents[k] /= counts[k]
            else:
                new_cents[k] = Xc[rng.integers(n)]
        if np.allclose(cents, new_cents):
            cents = new_cents
            break
        cents = new_cents
    return cents.astype(np.float32)

def train_prototypes(X, y, C, K, rng):
    d = X.shape[1]
    protos = np.zeros((C, K, d), np.float32)
    for c in range(C):
        Xc = X[y == c]
        protos[c] = kmeans_class(Xc, K, rng)
    return protos

def predict(X, protos):
    n, d = X.shape
    C, K, _ = protos.shape
    dists = np.zeros((n, C), np.float32)
    for c in range(C):
        for k in range(K):
            diff = X - protos[c, k]
            dists[:, c] = np.minimum(dists[:, c], (diff * diff).sum(axis=1))
    return dists.argmin(axis=1).astype(np.int64)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # split: last 20% for tuning
    split = int(n * (1 - TUNING_FRAC))
    idx = np.arange(n)
    train_idx, tune_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    mu, sd = zscore_params(X_train)
    X_train_z = apply_zscore(X_train, mu, sd)
    X_tune_z = apply_zscore(X_tune, mu, sd)

    best_K, best_acc = 1, -1.0
    for K in K_CANDIDATES:
        protos = train_prototypes(X_train_z, y_train, C, K, rng)
        pred = predict(X_tune_z, protos)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc, best_K = acc, K

    # retrain on all data with best_K
    mu_all, sd_all = zscore_params(X)
    X_all_z = apply_zscore(X, mu_all, sd_all)
    protos = train_prototypes(X_all_z, y, C, best_K, rng)

    # write weights.bin
    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<4i", magic, d, C, best_K)
    out += mu_all.astype("<f4").tobytes()
    out += sd_all.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (d <= 0 || C <= 0 || K <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * d * 2 + 4L * C * K * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *protos = sd + d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(d * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mu[j]) / sd[j];

        float best_dist = 1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                const float *p = protos + ((long)c * K + kk) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = z[j] - p[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_class = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```