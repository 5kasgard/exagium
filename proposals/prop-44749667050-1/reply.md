```hypothesis
Family: label-proto (prototype classifier with 2 centroids per class via seeded k-means).
Why: Optdigits has multiple modes per digit (writing styles); 2 prototypes capture this non-linearity better than linear, while inference (20×64 distance ops) is cheaper than MLP16 (16×64+16×10) and much cheaper than GBDT/KNN.
Failure: If dev accuracy ≤ 0.961 (linear) or Ir/item ≥ 8000.
```
```python
"""Prototype family for label tasks: 2 centroids per class via seeded k-means.
python3 -s -P train.py card.json train.in train.key weights.bin"""
import json, struct, sys
import numpy as np

SEED = 12345
K_PER_CLASS = 2
MAX_ITER = 30

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans_class(Xc, k, rng):
    """Xc: (n, d) float32, returns (k, d) centroids."""
    n, d = Xc.shape
    if n <= k:
        # duplicate if too few
        centroids = np.vstack([Xc, np.zeros((k - n, d), dtype=np.float32)])
        return centroids[:k]
    # random initialization from data points
    idx = rng.choice(n, size=k, replace=False)
    centroids = Xc[idx].copy()
    for _ in range(MAX_ITER):
        # assign
        diff = Xc[:, None, :] - centroids[None, :, :]  # (n, k, d)
        dist2 = np.sum(diff * diff, axis=2)            # (n, k)
        assign = np.argmin(dist2, axis=1)              # (n,)
        # update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int32)
        for i in range(n):
            c = assign[i]
            new_centroids[c] += Xc[i]
            counts[c] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                # reinitialize empty cluster to a random point
                new_centroids[j] = Xc[rng.integers(n)]
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
    Xf = X.astype(np.float32)
    rng_master = np.random.default_rng(SEED)
    all_centroids = []
    all_labels = []
    for c in range(C):
        mask = (y == c)
        Xc = Xf[mask]
        # per-class seeded RNG
        rng = np.random.default_rng(SEED + c + 1)
        cents = kmeans_class(Xc, K_PER_CLASS, rng)
        all_centroids.append(cents)
        all_labels.extend([c] * K_PER_CLASS)
    centroids = np.vstack(all_centroids).astype(np.float32)  # (K*C, d)
    labels = np.array(all_labels, dtype=np.uint8)            # (K*C,)
    K_total = centroids.shape[0]
    # weights format: magic, d, K_total, C, centroids flat (K_total*d float32), labels (K_total uint8)
    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<4i", magic, d, K_total, C)
    out += centroids.astype("<f4").tobytes()
    out += labels.astype("<u1").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Prototype classifier: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class of nearest centroid. */
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (d <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (long)K * d + (long)K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *centroids = (const float *)(hd + 4);
    const unsigned char *labels = (const unsigned char *)(centroids + (long)K * d);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *diff = (float *)malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f;
        unsigned char best_label = 0;
        for (int cidx = 0; cidx < K; cidx++) {
            const float *cent = centroids + (long)cidx * d;
            float dist2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float df = (float)xi[j] - cent[j];
                dist2 += df * df;
            }
            if (dist2 < best_dist) {
                best_dist = dist2;
                best_label = labels[cidx];
            }
        }
        put_rec(stdout, &best_label, 1);
    }
    free(diff);
    return 0;
}
```