```hypothesis
Family: label-proto (prototype classifier with 3 prototypes per class via k-means on z-scored pixels).
Why: Optdigits classes have non-linear boundaries in pixel space; a nearest-centroid model captures this better than linear while being far cheaper than k-NN (30 prototypes vs 2867 training points). The tiny weight file (8 KB) and simple distance calculations yield very low CPU cost.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear), the classes are essentially linearly separable after z-scoring and prototypes add no value.
```
```python
import json, struct, sys
import numpy as np

K = 3
SEED = 12345

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, k, rng, max_iter=20):
    n, d = X.shape
    centroids = np.zeros((k, d), dtype=np.float32)
    idx = rng.integers(n)
    centroids[0] = X[idx]
    for i in range(1, k):
        diff = X[:, None, :] - centroids[:i]
        dist2 = np.sum(diff * diff, axis=2)
        min_dist2 = np.min(dist2, axis=1)
        if min_dist2.sum() == 0:
            idx = rng.integers(n)
        else:
            probs = min_dist2 / min_dist2.sum()
            idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
    for _ in range(max_iter):
        diff = X[:, None, :] - centroids
        dist2 = np.sum(diff * diff, axis=2)
        labels = np.argmin(dist2, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int32)
        for j in range(n):
            c = labels[j]
            new_centroids[c] += X[j]
            counts[c] += 1
        for i in range(k):
            if counts[i] > 0:
                new_centroids[i] /= counts[i]
            else:
                new_centroids[i] = X[rng.integers(n)]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    
    rng = np.random.default_rng(SEED)
    
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Z = ((X - mean) / std).astype(np.float32)
    
    prototypes = np.zeros((C * K, d), dtype=np.float32)
    for c in range(C):
        Zc = Z[y == c]
        if len(Zc) == 0:
            prototypes[c*K:(c+1)*K] = rng.uniform(-1, 1, (K, d)).astype(np.float32)
        else:
            prototypes[c*K:(c+1)*K] = kmeans(Zc, K, rng)
    
    magic = 0x50524F54
    out = struct.pack("<4i", magic, d, C, K)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (d != 64 || C != 10 || K != 3) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * (2L * d + (long)C * K * d);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    
    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *prototypes = std + d;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *norm = malloc(sizeof(float) * d);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            norm[j] = (xi[j] - mean[j]) / std[j];
        }
        float best_dist = 1e30f;
        int best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                const float *proto = prototypes + ((long)c * K + p) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = norm[j] - proto[j];
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