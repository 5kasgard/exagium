```hypothesis
Family: label-proto (prototype-based classifier with per-class k-means on z-scored pixels).
Why: Optdigits has strong spatial structure; prototypes capture nonlinear class boundaries better than a linear model while staying tiny (few KB) and fast (O(C*K*d) ops). Beats label-linear (96.1%) if dev accuracy ≥ 96.5% at lower Ir/item.
Failure: Dev accuracy < 96.1% or Ir/item > label-linear.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
K_CANDIDATES = [1, 2, 3, 4, 5, 8, 10]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, K, rng, max_iter=100):
    n = X.shape[0]
    if n <= K:
        return X[:K].copy()
    centroids = np.zeros((K, X.shape[1]), dtype=np.float32)
    idx = rng.integers(n)
    centroids[0] = X[idx]
    dist2 = np.full(n, np.inf, dtype=np.float32)
    for k in range(1, K):
        d = ((X - centroids[k-1])**2).sum(axis=1)
        np.minimum(dist2, d, out=dist2)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centroids[k] = X[idx]
    for _ in range(max_iter):
        diff = X[:, None, :] - centroids[None, :, :]
        dist2 = (diff**2).sum(axis=2)
        assign = dist2.argmin(axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.int32)
        for i in range(n):
            a = assign[i]
            new_centroids[a] += X[i]
            counts[a] += 1
        for k in range(K):
            if counts[k] > 0:
                new_centroids[k] /= counts[k]
            else:
                new_centroids[k] = X[rng.integers(n)]
        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_K = 1
    best_acc = -1.0
    for K in K_CANDIDATES:
        mu = X_train.mean(axis=0).astype(np.float32)
        sd = X_train.std(axis=0).astype(np.float32)
        sd[sd == 0] = 1.0
        Z_train = ((X_train - mu) / sd).astype(np.float32)
        Z_tune = ((X_tune - mu) / sd).astype(np.float32)
        protos = np.zeros((C, K, d), dtype=np.float32)
        for c in range(C):
            mask = (y_train == c)
            if mask.sum() > 0:
                protos[c] = kmeans(Z_train[mask], K, rng)
        protos_flat = protos.reshape(C*K, d)
        diff = Z_tune[:, None, :] - protos_flat[None, :, :]
        dist2 = (diff**2).sum(axis=2)
        assign = dist2.argmin(axis=1)
        pred = assign // K
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_K = K
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)
    protos = np.zeros((C, best_K, d), dtype=np.float32)
    for c in range(C):
        mask = (y == c)
        if mask.sum() > 0:
            protos[c] = kmeans(Z[mask], best_K, rng)
    magic = 0x544F5250  # "PROT"
    out = struct.pack("<4i", magic, d, best_K, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x544F5250) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64 || K < 1) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * d * 2 + 4L * C * K * d;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }
    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *protos = sd + d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *znorm = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) znorm[j] = (xi[j] - mu[j]) / sd[j];
        float best_dist = 1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                const float *p = protos + ((long)c * K + kk) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = znorm[j] - p[j];
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