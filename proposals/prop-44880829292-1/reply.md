```hypothesis
label-proto: prototype-based classifier (k-means centroids per class). Optdigits pixels are ordinal 0–16; Euclidean distance on standardized pixels captures stroke similarity well. Few prototypes/class (3–5) yield non-linear boundaries faster than MLP (no matmul, just distance). If dev accuracy ≤0.9613 at ≥5000 Ir/item, linear separability after z-scoring suffices.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
K_CHOICES = [1, 2, 3, 4, 5, 8, 10]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, K, rng, max_iter=100):
    n, d = X.shape
    centroids = np.zeros((K, d), dtype=np.float32)
    idx = rng.integers(n)
    centroids[0] = X[idx]
    for k in range(1, K):
        dist2 = np.min(np.sum((X[:, None] - centroids[:k])**2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centroids[k] = X[idx]
    for _ in range(max_iter):
        dist2 = np.sum((X[:, None] - centroids)**2, axis=2)
        labels = np.argmin(dist2, axis=1)
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
                new_centroids[k] = X[rng.integers(n)]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X_bytes.shape[0]

    mu = X_bytes.mean(axis=0).astype(np.float32)
    sd = X_bytes.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X = ((X_bytes - mu) / sd).astype(np.float32)

    n_tune = int(0.2 * n)
    idx = np.arange(n)
    tune_idx = idx[-n_tune:]
    train_idx = idx[:-n_tune]

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    rng = np.random.default_rng(SEED)

    best_K = 1
    best_acc = -1

    for K in K_CHOICES:
        prototypes = np.zeros((C * K, d), dtype=np.float32)
        proto_labels = np.zeros(C * K, dtype=np.int32)
        for c in range(C):
            Xc = X_train[y_train == c]
            if len(Xc) == 0:
                centroids = np.tile(X_train.mean(axis=0), (K, 1))
            else:
                centroids = kmeans(Xc, K, rng)
            prototypes[c*K:(c+1)*K] = centroids
            proto_labels[c*K:(c+1)*K] = c

        dist2 = np.sum((X_tune[:, None] - prototypes)**2, axis=2)
        pred = proto_labels[np.argmin(dist2, axis=1)]
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_K = K

    prototypes = np.zeros((C * best_K, d), dtype=np.float32)
    proto_labels = np.zeros(C * best_K, dtype=np.int32)
    for c in range(C):
        Xc = X[y == c]
        if len(Xc) == 0:
            centroids = np.tile(X.mean(axis=0), (best_K, 1))
        else:
            centroids = kmeans(Xc, best_K, rng)
        prototypes[c*best_K:(c+1)*best_K] = centroids
        proto_labels[c*best_K:(c+1)*best_K] = c

    out = struct.pack("<4i", 0x50524F54, d, C, best_K)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (d < 1 || d > 256 || C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)d * 4 * 2 + (long)C * K * d * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *mean = (float *)(hd + 4);
    float *std = mean + d;
    float *proto = std + d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *buf = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) buf[j] = (xi[j] - mean[j]) / std[j];
        float best_dist = 1e30f;
        int best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                float *p = proto + ((long)c * K + kk) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = buf[j] - p[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_class = c;
                }
            }
        }
        unsigned char out = (unsigned char)best_class;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```