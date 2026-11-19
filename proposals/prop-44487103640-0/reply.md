```hypothesis
Nearest Prototype classifier: learn 3 k-means prototypes per class (30 total) on z-scored pixels.
Digits have non-linear manifolds; prototypes capture multi-modal class structure better than a single
linear hyperplane, while inference is just 30 small Euclidean distances — faster than MLP. If dev
accuracy ≤ 96.13% (label-linear) or latency ≥ 5000 Ir/item, the family does not improve the frontier.
```
```python
"""Nearest Prototype family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Per-byte z-scoring like label-linear.
Learns K prototypes per class via k-means (fixed seed, 20 iters). Exports:
magic 0x50524F54, d, K, C; float32 mean[d], std[d]; float32 proto[C][K][d]."""
import json, math, struct, sys

import numpy as np

K = 3
SEED = 1
KMEANS_ITERS = 20


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
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # z-score each pixel (ordinal only in this task)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X.astype(np.float32) - mu) / sd)

    # k-means per class
    proto = np.zeros((C, K, d), dtype=np.float32)
    for c in range(C):
        Zc = Z[y == c]
        n = Zc.shape[0]
        if n == 0:
            continue
        # k-means++ init (deterministic via rng)
        cents = np.zeros((K, d), np.float32)
        cents[0] = Zc[rng.integers(n)]
        for k in range(1, K):
            dist2 = np.min(np.sum((Zc[:, None, :] - cents[:k])**2, axis=2), axis=1)
            probs = dist2 / dist2.sum()
            cents[k] = Zc[rng.choice(n, p=probs)]
        # Lloyd iterations
        for _ in range(KMEANS_ITERS):
            dist2 = np.sum((Zc[:, None, :] - cents)**2, axis=2)
            assign = np.argmin(dist2, axis=1)
            for k in range(K):
                mask = (assign == k)
                if mask.any():
                    cents[k] = Zc[mask].mean(axis=0)
        proto[c] = cents

    # write weights
    out = struct.pack("<4i", 0x50524F54, d, K, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += proto.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Nearest Prototype classifier: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (argmin squared Euclidean distance to prototypes). */
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
    long expected = 16L + 4L * d + 4L * d + 4L * (long)C * K * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *proto = sd + d;  // [C][K][d]

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
                const float *p = proto + ((long)c * K + kk) * d;
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