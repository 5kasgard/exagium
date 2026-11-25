```hypothesis
Family: label-proto (prototype-based classifier with per-class k-means).
Why: Optdigits pixels have clear class-conditional clusters; prototypes capture non-linear boundaries with few parameters. Tuning K (1–8) on the last 20% of train.in selects model complexity. Inference is O(C·K·d) distance computations—fast, no branching, cache-friendly.
Failure sign: Dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 5000.
```
```python
import json, struct, sys
import numpy as np

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, K, rng, max_iter=100):
    n, d = X.shape
    if n <= K:
        centroids = np.zeros((K, d), dtype=np.float32)
        centroids[:n] = X.astype(np.float32)
        return centroids
    centroids = np.zeros((K, d), dtype=np.float32)
    idx = rng.integers(n)
    centroids[0] = X[idx].astype(np.float32)
    for k in range(1, K):
        dists = np.full(n, np.inf, dtype=np.float32)
        for j in range(k):
            diff = X - centroids[j]
            d2 = np.sum(diff * diff, axis=1)
            np.minimum(dists, d2, out=dists)
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centroids[k] = X[idx].astype(np.float32)
    for _ in range(max_iter):
        diffs = X[:, None, :] - centroids[None, :, :]
        dists2 = np.sum(diffs * diffs, axis=2)
        labels = np.argmin(dists2, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.int32)
        for i in range(n):
            c = labels[i]
            new_centroids[c] += X[i]
            counts[c] += 1
        for k in range(K):
            if counts[k] > 0:
                new_centroids[k] /= counts[k]
            else:
                idx = rng.integers(n)
                new_centroids[k] = X[idx].astype(np.float32)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    
    rng = np.random.default_rng(SEED)
    
    n_tune = n // 5
    train_idx = np.arange(n - n_tune)
    tune_idx = np.arange(n - n_tune, n)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]
    
    best_K = 1
    best_acc = -1.0
    for K in range(1, 9):
        prototypes = np.zeros((C, K, d), dtype=np.float32)
        for c in range(C):
            Xc = X_train[y_train == c].astype(np.float32)
            if len(Xc) > 0:
                prototypes[c] = kmeans(Xc, K, rng)
        correct = 0
        for i in range(len(X_tune)):
            x = X_tune[i].astype(np.float32)
            diffs = prototypes.reshape(C * K, d) - x
            dists = np.sum(diffs * diffs, axis=1)
            pred = np.argmin(dists) // K
            if pred == y_tune[i]:
                correct += 1
        acc = correct / len(X_tune)
        if acc > best_acc:
            best_acc = acc
            best_K = K
    
    prototypes = np.zeros((C, best_K, d), dtype=np.float32)
    for c in range(C):
        Xc = X[y == c].astype(np.float32)
        if len(Xc) > 0:
            prototypes[c] = kmeans(Xc, best_K, rng)
    
    out = struct.pack("<4i", 0x544F5250, d, best_K, C)
    out += prototypes.astype("<f4").tobytes()
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
    long expected = 16 + 4L * (long)C * K * d;
    if (C < 1 || C > 256 || K < 1 || K > 256 || d < 1 || d > 1024 || nw != expected) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *protos = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *dists = malloc(sizeof(float) * C * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                const float *proto = protos + ((long)c * K + p) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = (float)xi[j] - proto[j];
                    dist += diff * diff;
                }
                dists[c * K + p] = dist;
            }
        }
        unsigned char best_c = 0;
        float best_dist = dists[0];
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                float dist = dists[c * K + p];
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best_c, 1);
    }
    return 0;
}
```