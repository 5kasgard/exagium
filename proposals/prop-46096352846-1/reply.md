```hypothesis
Family: label-proto (prototype classifier with per-class k-means, implemented as linear lookup tables).
Why: Optdigits has 64 ordinal pixels; class-conditional densities are multi-modal. 3 prototypes/class (30 total) capture non-linear boundaries better than a single linear separator, while inference remains a single matrix-vector multiply (like label-linear) plus argmax over 30 scores → similar CPU cost, higher accuracy.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or inference cost > 2× label-linear.
```
```python
"""label-proto: prototype classifier (k-means per class) exported as linear lookup tables.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), stdlib+numpy, single thread."""
import json, math, struct, sys

import numpy as np

SEED = 1
K_PER_CLASS = 3          # prototypes per class
KMEANS_ITERS = 50
KMEANS_TOL = 1e-4


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, k, rng):
    """Deterministic k-means: init with first k points, then Lloyd."""
    n, d = X.shape
    if n <= k:
        cents = X.copy()
        if n < k:
            extra = rng.uniform(X.min(0), X.max(0), (k - n, d))
            cents = np.vstack([cents, extra])
        return cents.astype(np.float32)
    cents = X[:k].copy().astype(np.float64)
    for _ in range(KMEANS_ITERS):
        # assign
        diff = X[:, None, :] - cents[None, :, :]          # (n,k,d)
        dist2 = (diff * diff).sum(axis=2)                 # (n,k)
        assign = dist2.argmin(axis=1)                     # (n,)
        # update
        new_cents = np.zeros_like(cents)
        counts = np.zeros(k, dtype=np.int64)
        for i in range(n):
            c = assign[i]
            new_cents[c] += X[i]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                new_cents[c] /= counts[c]
            else:
                new_cents[c] = rng.uniform(X.min(0), X.max(0))
        if np.max(np.linalg.norm(new_cents - cents, axis=1)) < KMEANS_TOL:
            cents = new_cents
            break
        cents = new_cents
    return cents.astype(np.float32)


def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # per-feature z-score (all ordinal)
    mu = X_raw.mean(axis=0).astype(np.float64)
    sd = X_raw.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    X_enc = ((X_raw - mu) / sd).astype(np.float32)        # (n, d)

    # k-means per class
    prototypes = []       # list of (proto_vector, class_label)
    for c in range(C):
        Xc = X_enc[y == c]
        cents = kmeans(Xc, K_PER_CLASS, rng)
        for proto in cents:
            prototypes.append((proto, c))

    W = len(prototypes)                   # C * K_PER_CLASS
    proto_mat = np.stack([p for p, _ in prototypes], axis=0)   # (W, d)
    proto_class = np.array([c for _, c in prototypes], dtype=np.uint8)

    # Build lookup tables T[j][v][w] = v * weight_w[j]  where weight_w[j] = 2 * proto[w,j] / sd[j]
    # and bias[w] = -2 * sum_j mu[j] * proto[w,j] / sd[j] - ||proto[w]||^2
    weight = 2.0 * proto_mat / sd.astype(np.float32)           # (W, d)
    bias = -2.0 * np.sum(mu.astype(np.float32) * proto_mat / sd.astype(np.float32), axis=1) - np.sum(proto_mat * proto_mat, axis=1)

    # T shape: (d, 256, W)  float32 little-endian
    T = np.empty((d, 256, W), dtype=np.float32)
    for j in range(d):
        wj = weight[:, j]                       # (W,)
        T[j] = np.outer(np.arange(256, dtype=np.float32), wj)   # (256, W)

    # Pack weights: magic, d, K, C, T, bias, proto_class
    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<4i", magic, d, K_PER_CLASS, C)
    out += T.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    out += proto_class.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-proto inference: ./infer weights.bin in.rec N
 * Each input record: d bytes (pixels). Output: one byte class (0-9).
 * Weights: magic 0x50524F54, d, K, C; float32 T[d][256][W]; float32 bias[W]; uint8 proto_class[W] (W=C*K). */
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
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    int W = C * K;
    if (C < 1 || C > 256 || K < 1 || K > 256 || d < 1 || d > 1024) {
        fprintf(stderr, "bad header dims\n"); return 3;
    }
    long expected = 16L + 4L * d * 256 * W + 4L * W + W;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *bias = T + (long)d * 256 * W;
    const unsigned char *proto_class = (const unsigned char *)(bias + W);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * W);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int w = 0; w < W; w++) scores[w] = bias[w];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) scores[w] += tj[w];
        }
        int best = 0;
        for (int w = 1; w < W; w++) if (scores[w] > scores[best]) best = w;
        unsigned char cls = proto_class[best];
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```