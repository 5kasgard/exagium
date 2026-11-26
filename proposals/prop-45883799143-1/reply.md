```hypothesis
Family: label-proto (prototype-based classifier with per-class k-means).
Why: Optdigits classes are multi-modal (different writing styles); a few prototypes per class capture non-linear boundaries better than a single linear separator. Inference computes squared L2 distance to ~30 centroids (10 classes × 3 prototypes) in z-scored space, costing ~2000 FLOPs/item — comparable to label-linear but with higher representational power.
Failure signal: Dev accuracy ≤ 0.9613 (label-linear) or CPU cost > 2× label-linear.
```

```python
import json, struct, sys
import numpy as np

SEED = 1
K = 3          # prototypes per class
MAX_ITER = 20  # k-means iterations

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Z-score each feature (all ordinal for optdigits; general ordinal-only family)
    mu = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Xz = (X.astype(np.float32) - mu) / std

    centroids = []
    for c in range(C):
        Xc = Xz[y == c]
        n_c = Xc.shape[0]
        if n_c == 0:
            cents = np.zeros((K, d), dtype=np.float32)
        else:
            # Deterministic initialization: first K samples (shuffled by fixed seed)
            idx = rng.choice(n_c, size=min(K, n_c), replace=False)
            cents = Xc[idx].copy()
            if len(cents) < K:
                cents = np.vstack([cents, np.zeros((K - len(cents), d), dtype=np.float32)])
            # Lloyd's algorithm
            for _ in range(MAX_ITER):
                diff = Xc[:, None, :] - cents[None, :, :]
                dist = np.sum(diff * diff, axis=2)
                assign = np.argmin(dist, axis=1)
                new_cents = np.zeros_like(cents)
                counts = np.zeros(K, dtype=np.int32)
                for i in range(n_c):
                    k = assign[i]
                    new_cents[k] += Xc[i]
                    counts[k] += 1
                for k in range(K):
                    if counts[k] > 0:
                        new_cents[k] /= counts[k]
                    else:
                        new_cents[k] = Xc[rng.integers(n_c)]
                if np.allclose(cents, new_cents):
                    cents = new_cents
                    break
                cents = new_cents
        centroids.append(cents)
    centroids = np.stack(centroids)  # (C, K, d)

    # Write weights: magic, d, C, K, mu[d], std[d], centroids[C*K*d]
    magic = 0x50524F54  # "PROT"
    out = struct.pack("<4i", magic, d, C, K)
    out += mu.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += centroids.astype("<f4").tobytes()
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
    if (d <= 0 || C <= 0 || K <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 2L * d * 4 + (long)C * K * d * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }
    float *mu = (float *)(hd + 4);
    float *std = mu + d;
    float *cents = std + d;  // [C][K][d]
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xz = malloc(d * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xz[j] = (xi[j] - mu[j]) / std[j];
        float best_dist = 1e30f;
        int best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                const float *cent = cents + ((long)c * K + kk) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = xz[j] - cent[j];
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