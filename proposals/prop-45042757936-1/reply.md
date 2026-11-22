```hypothesis
Prototype classifier (label-proto): per-class k-means centroids (K=5) as prototypes.
8x8 digits have multi-modal within-class distributions (stroke variations) that linear boundaries
cannot capture; 50 prototypes (5/class) model this with tiny 12KB weights and ~3200 ops/item.
If dev accuracy ≤ 0.9613 (label-linear) or latency ≥ 2× label-linear, the approach fails.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
K_PER_CLASS = 5
MAX_ITER = 100
TOL = 1e-4

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans_class(X, K, rng):
    n = X.shape[0]
    if n <= K:
        cents = X[:K].copy()
        if n < K:
            cents = np.vstack([cents, rng.normal(size=(K - n, X.shape[1]))])
        return cents
    idx = rng.choice(n, K, replace=False)
    cents = X[idx].copy().astype(np.float64)
    for _ in range(MAX_ITER):
        diff = X[:, None, :] - cents[None, :, :]
        dist2 = (diff * diff).sum(axis=2)
        assign = dist2.argmin(axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(K, dtype=np.int64)
        for i in range(n):
            c = assign[i]
            new_cents[c] += X[i]
            counts[c] += 1
        for k in range(K):
            if counts[k] > 0:
                new_cents[k] /= counts[k]
            else:
                new_cents[k] = rng.normal(size=X.shape[1])
        shift = np.sqrt(((new_cents - cents) ** 2).sum(axis=1)).max()
        cents = new_cents
        if shift < TOL:
            break
    return cents.astype(np.float32)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"]).astype(np.float32)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)
    all_cents = []
    for c in range(C):
        Xc = X[y == c]
        cents = kmeans_class(Xc, K_PER_CLASS, rng)
        all_cents.append(cents)
    cents = np.vstack(all_cents)  # (C*K, d)
    magic = 0x50524F54  # 'PROT'
    hdr = struct.pack("<4i", magic, d, C, K_PER_CLASS)
    open(sys.argv[4], "wb").write(hdr + cents.astype("<f4").tobytes())

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
    int P = C * K;
    if (C < 1 || C > 256 || K < 1 || d < 1 || nw != 16 + 4L * (long)P * d) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *cents = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f; int best_idx = 0;
        for (int p = 0; p < P; p++) {
            float dist = 0.0f;
            const float *c = cents + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - c[j];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_idx = p; }
        }
        unsigned char cls = (unsigned char)(best_idx / K);
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```