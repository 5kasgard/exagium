```hypothesis
Prototype classifier (label-proto): per-class k-means centroids (K=2) on z-scored pixels.
Captures multi-modal digit styles (e.g., crossed vs. uncrossed '7') with non-linear boundaries.
Inference is O(C*K*d) ~ 1280 FLOPs/item, competitive with linear.
Fails if dev accuracy <= 0.9613 (label-linear).
```
```python
import json, struct, sys

import numpy as np

SEED, K_PER_CLASS, MAX_ITER = 1, 2, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, k, rng, max_iter=100):
    n, d = X.shape
    if n <= k:
        cents = np.zeros((k, d), dtype=np.float32)
        cents[:n] = X
        return cents
    cents = np.zeros((k, d), dtype=np.float32)
    idx = rng.integers(n)
    cents[0] = X[idx]
    for i in range(1, k):
        dists = np.min(np.sum((X[:, None, :] - cents[:i]) ** 2, axis=2), axis=1)
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        cents[i] = X[idx]
    for _ in range(max_iter):
        diffs = X[:, None, :] - cents[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(k, dtype=np.int32)
        for j in range(n):
            c = labels[j]
            new_cents[c] += X[j]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                new_cents[c] /= counts[c]
            else:
                idx = rng.integers(n)
                new_cents[c] = X[idx]
        if np.allclose(cents, new_cents):
            cents = new_cents
            break
        cents = new_cents
    return cents


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    means = np.zeros(d, dtype=np.float32)
    stds = np.ones(d, dtype=np.float32)
    X = np.zeros_like(X_bytes, dtype=np.float32)
    for j in range(d):
        t = card["input"]["types"][j]
        v = X_bytes[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            means[j] = mu
            stds[j] = sd
            X[:, j] = (v - mu) / sd
        else:
            mu, sd = 127.5, 73.0
            means[j] = mu
            stds[j] = sd
            X[:, j] = (v - mu) / sd

    prototypes = np.zeros((C * K_PER_CLASS, d), dtype=np.float32)
    for c in range(C):
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        cents = kmeans(Xc, K_PER_CLASS, rng, MAX_ITER)
        prototypes[c * K_PER_CLASS:(c + 1) * K_PER_CLASS] = cents

    magic = 0x50524F54
    out = struct.pack("<4i", magic, d, K_PER_CLASS, C)
    out += means.astype("<f4").tobytes()
    out += stds.astype("<f4").tobytes()
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || d < 1) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * d * 2 + 4L * C * K * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *means = (const float *)(hd + 4);
    const float *stds = means + d;
    const float *protos = stds + d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *dists = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) z[j] = (xi[j] - means[j]) / stds[j];

        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                const float *p = protos + ((long)c * K + kk) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = z[j] - p[j];
                    dist += diff * diff;
                }
                dists[c * K + kk] = dist;
            }
        }

        unsigned char best_c = 0;
        float best_dist = dists[0];
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                float dist = dists[c * K + kk];
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