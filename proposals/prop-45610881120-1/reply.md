```hypothesis
Family: label-proto (prototype-based classifier with per-class k-means).
Why: Optdigits forms tight clusters per digit in the 64D z-scored space; a few prototypes per class capture non-linear boundaries better than a linear model while keeping inference extremely fast (30 prototypes × 64 dims ≈ 1920 FLOPs/item).
Failure: Dev accuracy ≤ 0.9613 (label-linear) or instruction count > 10 000 Ir/item.
```
```python
"""label-proto: prototype classifier (per-class k-means) for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encodes ordinal features as z-scores (mu,sd per byte). Learns M prototypes/class via k-means++
on the tuning split (last 20% of train), then retrains on all data.
Weights format: int32 magic=0x50524F54, d, C, M; float32 mu[d], sd[d], proto[M*C][d]."""
import json, math, struct, sys

import numpy as np

SEED = 1
KM_ITERS = 20
M_CANDIDATES = [1, 2, 3, 4, 5]  # prototypes per class to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, k, rng):
    """X: (n,d) float32, returns (k,d) centroids. Deterministic k-means++ + Lloyd."""
    n, d = X.shape
    cents = np.zeros((k, d), np.float32)
    # k-means++
    idx = rng.integers(n)
    cents[0] = X[idx]
    for i in range(1, k):
        # squared distances to nearest existing centroid
        diff = X[:, None, :] - cents[:i][None, :, :]  # (n,i,d)
        dist2 = np.sum(diff * diff, axis=2).min(axis=1)  # (n,)
        if dist2.sum() == 0:
            idx = rng.integers(n)
        else:
            probs = dist2 / dist2.sum()
            idx = rng.choice(n, p=probs)
        cents[i] = X[idx]
    # Lloyd
    for _ in range(KM_ITERS):
        diff = X[:, None, :] - cents[None, :, :]  # (n,k,d)
        dist2 = np.sum(diff * diff, axis=2)  # (n,k)
        labels = np.argmin(dist2, axis=1)
        new_cents = np.zeros_like(cents)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_cents[j] = X[mask].mean(axis=0)
            else:
                new_cents[j] = cents[j]
        if np.allclose(cents, new_cents):
            break
        cents = new_cents
    return cents


def evaluate(X, y, cents, M, C):
    """X: (n,d), cents: (M*C,d). Return accuracy."""
    n = X.shape[0]
    # Compute squared distances to all prototypes
    # Use ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    x_norm2 = np.sum(X * X, axis=1)  # (n,)
    c_norm2 = np.sum(cents * cents, axis=1)  # (M*C,)
    dots = X @ cents.T  # (n, M*C)
    scores = -2 * dots + c_norm2[None, :]  # (n, M*C), ||x||^2 constant omitted
    pred_proto = np.argmin(scores, axis=1)
    pred_class = pred_proto // M
    return np.mean(pred_class == y)


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(y)

    # Split: last 20% for tuning (file order)
    split = int(n * 0.8)
    X_tr, y_tr = X_bytes[:split], y[:split]
    X_tu, y_tu = X_bytes[split:], y[split:]

    # Feature scaling on TRAIN only
    mu = X_tr.mean(axis=0).astype(np.float64)
    sd = X_tr.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    def scale(X):
        return ((X.astype(np.float64) - mu) / sd).astype(np.float32)

    Z_tr = scale(X_tr)
    Z_tu = scale(X_tu)

    rng = np.random.default_rng(SEED)
    best_M, best_acc = 1, -1.0
    for M in M_CANDIDATES:
        cents_list = []
        for c in range(C):
            Xc = Z_tr[y_tr == c]
            if len(Xc) == 0:
                cents_list.append(np.zeros((M, d), np.float32))
            else:
                cents_list.append(kmeans(Xc, M, rng))
        cents = np.vstack(cents_list)  # (M*C, d)
        acc = evaluate(Z_tu, y_tu, cents, M, C)
        if acc > best_acc:
            best_acc, best_M = acc, M

    # Retrain on ALL data with best_M
    mu_all = X_bytes.mean(axis=0).astype(np.float64)
    sd_all = X_bytes.std(axis=0).astype(np.float64)
    sd_all[sd_all == 0] = 1.0
    Z_all = ((X_bytes.astype(np.float64) - mu_all) / sd_all).astype(np.float32)

    rng = np.random.default_rng(SEED)
    cents_list = []
    for c in range(C):
        Xc = Z_all[y == c]
        cents_list.append(kmeans(Xc, best_M, rng))
    cents = np.vstack(cents_list)  # (best_M*C, d)

    # Write weights
    magic = 0x50524F54  # 'PROT'
    hdr = struct.pack("<4i", magic, d, C, best_M)
    out = hdr + mu_all.astype("<f4").tobytes() + sd_all.astype("<f4").tobytes() + cents.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-proto infer: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output record: 1 byte class (0..C-1).
 * Weights: int32 magic=0x50524F54, d, C, M; float32 mu[d], sd[d], proto[M*C][d]. */
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
    int d = hd[1], C = hd[2], M = hd[3];
    if (d <= 0 || C <= 0 || M <= 0 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * (2L * d + (long)M * C * d);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *proto = sd + d;                 // [M*C][d]
    long n_proto = (long)M * C;

    // Precompute prototype squared norms for fast distance: ||x-c||^2 = ||x||^2 + ||c||^2 - 2<x,c>
    float *proto_norm2 = malloc(sizeof(float) * n_proto);
    for (long i = 0; i < n_proto; i++) {
        float s = 0.0f;
        const float *p = proto + i * d;
        for (int j = 0; j < d; j++) s += p[j] * p[j];
        proto_norm2[i] = s;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *dots = malloc(sizeof(float) * n_proto);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Scale to z = (x - mu) / sd
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mu[j]) * (1.0f / sd[j]);

        // Compute dot products with all prototypes
        for (long p = 0; p < n_proto; p++) {
            float sum = 0.0f;
            const float *w = proto + p * d;
            for (int j = 0; j < d; j++) sum += z[j] * w[j];
            dots[p] = sum;
        }

        // Find prototype minimizing -2*dot + ||proto||^2 (||z||^2 constant)
        long best_p = 0;
        float best_score = -2.0f * dots[0] + proto_norm2[0];
        for (long p = 1; p < n_proto; p++) {
            float score = -2.0f * dots[p] + proto_norm2[p];
            if (score < best_score) { best_score = score; best_p = p; }
        }
        unsigned char cls = (unsigned char)(best_p / M);
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```