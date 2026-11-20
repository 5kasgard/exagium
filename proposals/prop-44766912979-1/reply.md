```hypothesis
Family: label-proto (per-class k-means prototypes on z-scored pixels).
Why: optdigits has strong per-class clusters in pixel space; 3 prototypes/class (30 total) captures non-linear boundaries better than 1 centroid (linear) while staying extremely fast: inference is one matrix multiply (64→30) plus 30 squared distances. No ReLU, no softmax, no trees.
Fail signal: dev accuracy ≤ 0.9613 (label-linear) or instructions/item ≥ label-linear (5059).
```

```python
"""label-proto: per-class k-means prototypes on z-scored ordinal features.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Weights format (little-endian):
  magic 0x50524F54 ('PROT'), d=64, K=30, C=10, k_per_class=3
  float32 mean[64], std[64] (std clamped to >=1.0)
  float32 proto[30][64] (k_per_class * C prototypes, row-major)
  uint8   proto_class[30] (class index for each prototype)
Inference: z-score input, compute squared L2 distance to each prototype, pick class of nearest.
"""
import json, struct, sys
import numpy as np

SEED = 1
K_PER_CLASS = 3
MAX_ITER = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans_pp_init(X, k, rng):
    """k-means++ initialization (deterministic given rng)."""
    n, d = X.shape
    centroids = np.empty((k, d), dtype=np.float32)
    idx = rng.integers(n)
    centroids[0] = X[idx]
    for i in range(1, k):
        # compute squared distances to nearest centroid
        dist2 = np.full(n, np.inf, dtype=np.float32)
        for j in range(i):
            diff = X - centroids[j]
            d2 = np.sum(diff * diff, axis=1)
            np.minimum(dist2, d2, out=dist2)
        # sample proportional to dist2
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
    return centroids

def kmeans_fit(X, k, rng, max_iter=100):
    """Lloyd's algorithm, deterministic given rng."""
    centroids = kmeans_pp_init(X, k, rng)
    n, d = X.shape
    for _ in range(max_iter):
        # assignment
        dist2 = np.empty((n, k), dtype=np.float32)
        for j in range(k):
            diff = X - centroids[j]
            dist2[:, j] = np.sum(diff * diff, axis=1)
        labels = np.argmin(dist2, axis=1)
        # update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int32)
        for i in range(n):
            c = labels[i]
            new_centroids[c] += X[i]
            counts[c] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                # reinitialize empty cluster to random point
                new_centroids[j] = X[rng.integers(n)]
        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
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

    # z-score per pixel (ordinal features only; all are ordinal per card)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd < 1.0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    # per-class k-means
    prototypes = []
    proto_classes = []
    for c in range(C):
        Xc = Z[y == c]
        if len(Xc) == 0:
            # fallback: use global mean
            proto = np.tile(Z.mean(axis=0), (K_PER_CLASS, 1))
        else:
            proto = kmeans_fit(Xc, K_PER_CLASS, rng, MAX_ITER)
        prototypes.append(proto)
        proto_classes.extend([c] * K_PER_CLASS)

    prototypes = np.vstack(prototypes).astype(np.float32)  # (K, d) where K = C * K_PER_CLASS
    proto_classes = np.array(proto_classes, dtype=np.uint8)
    K = prototypes.shape[0]

    # weights: magic, d, K, C, K_PER_CLASS, mean[d], std[d], proto[K][d], proto_class[K]
    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<5i", magic, d, K, C, K_PER_CLASS)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += prototypes.astype("<f4").tobytes()
    out += proto_classes.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output record: 1 byte class 0..9.
 * Weights format (little-endian):
 *   int32 magic=0x50524F54, d, K, C, K_PER_CLASS
 *   float32 mean[d], std[d]
 *   float32 proto[K][d]
 *   uint8   proto_class[K]
 * Inference: z-score input, find nearest prototype (squared L2), output its class.
 */
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
    if (nw < 20 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3], K_PER_CLASS = hd[4];
    if (d != 64 || C != 10 || K != C * K_PER_CLASS || K_PER_CLASS != 3) {
        fprintf(stderr, "unexpected dims\n"); return 3;
    }
    size_t expected = 20 + (size_t)d * 2 * 4 + (size_t)K * d * 4 + (size_t)K;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 5);
    const float *std  = mean + d;
    const float *proto = (const float *)(std + d);
    const uint8_t *proto_class = (const uint8_t *)(proto + (size_t)K * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *dist2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // z-score
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mean[j]) / std[j];
        }

        // squared distances to prototypes
        for (int p = 0; p < K; p++) {
            float sum = 0.0f;
            const float *w = proto + (size_t)p * d;
            for (int j = 0; j < d; j++) {
                float diff = z[j] - w[j];
                sum += diff * diff;
            }
            dist2[p] = sum;
        }

        // nearest prototype
        int best_p = 0;
        for (int p = 1; p < K; p++) {
            if (dist2[p] < dist2[best_p]) best_p = p;
        }
        unsigned char best = proto_class[best_p];
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```