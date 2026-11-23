```hypothesis
Family: label-proto (prototype-based classifier with 2 prototypes per class).
Why: Optdigits has 10 well-separated classes; a few prototypes per class capture non-linear boundaries better than a linear model.
With 20 prototypes x 64 dims, inference is ~1280 multiply-adds (far cheaper than label-mlp16). If dev accuracy ≤ 0.9613, the family does not beat label-linear.
```
```python
"""label-proto: prototype classifier (2 prototypes per class) for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format: magic 0x50524F54, uint32 num_classes, uint32 prototypes_per_class, uint32 input_dim,
then float32 centroids[num_classes * prototypes_per_class * input_dim] (row-major)."""
import json, struct, sys
import numpy as np

SEED = 1
PROTOS_PER_CLASS = 2
MAX_ITER = 20

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, k, rng):
    """X: (n, d) float32. Return centroids (k, d)."""
    n = X.shape[0]
    if n <= k:
        cents = X[:k].copy()
        if n < k:
            cents = np.vstack([cents, np.tile(X.mean(0), (k - n, 1))])
        return cents
    # k-means++ initialization (deterministic via rng)
    cents = np.empty((k, X.shape[1]), dtype=np.float32)
    idx = rng.integers(n)
    cents[0] = X[idx]
    for i in range(1, k):
        dist2 = ((X[:, None, :] - cents[:i]) ** 2).sum(2).min(1)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        cents[i] = X[idx]
    # Lloyd's algorithm
    for _ in range(MAX_ITER):
        dist2 = ((X[:, None, :] - cents) ** 2).sum(2)
        assign = dist2.argmin(1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(k, dtype=np.int32)
        for j in range(n):
            c = assign[j]
            new_cents[c] += X[j]
            counts[c] += 1
        for c in range(k):
            if counts[c]:
                new_cents[c] /= counts[c]
            else:
                new_cents[c] = cents[c]
        if np.allclose(new_cents, cents):
            break
        cents = new_cents
    return cents

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)
    Xf = X.astype(np.float32)
    centroids = []
    for c in range(C):
        Xc = Xf[y == c]
        cents = kmeans(Xc, PROTOS_PER_CLASS, rng)
        centroids.append(cents)
    centroids = np.stack(centroids)  # (C, P, d)
    magic = 0x50524F54  # 'PROT'
    hdr = struct.pack("<4I", magic, C, PROTOS_PER_CLASS, d)
    body = centroids.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + body)

if __name__ == "__main__":
    main()
```
```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (argmin Euclidean distance to prototypes). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t C = hd[1], P = hd[2], d = hd[3];
    if (C == 0 || P == 0 || d == 0 || nw != 16 + 4L * C * P * d) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *centroids = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        float best_dist = 1e30f; uint32_t best_class = 0;
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t p = 0; p < P; p++) {
                const float *cent = centroids + (c * P + p) * d;
                float dist = 0.0f;
                for (uint32_t j = 0; j < d; j++) {
                    float diff = (float)xi[j] - cent[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_class = c;
                }
            }
        }
        put_rec(stdout, (const unsigned char *)&best_class, 1);
    }
    return 0;
}
```