```hypothesis
Family: label-proto (prototype-based classifier with 3 prototypes per class).
Why: Optdigits classes form tight, non-linear clusters in z-scored pixel space; prototypes capture this structure better than a linear decision boundary at ~6× the inference cost of label-linear (still <30k Ir/item).
Failure: Dev accuracy ≤ 0.9613 (label-linear) or inference cost > 30k Ir/item.
```
```python
import json, struct, sys

import numpy as np

M = 3
SEED = 1


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
        centroids = X[:k].copy()
        if n < k:
            centroids[n:] = centroids[0]
        return centroids
    centroids = np.zeros((k, d), dtype=X.dtype)
    idx = rng.integers(n)
    centroids[0] = X[idx]
    for i in range(1, k):
        dist2 = np.min(np.sum((X[:, None] - centroids[:i]) ** 2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
    for _ in range(max_iter):
        dist2 = np.sum((X[:, None] - centroids) ** 2, axis=2)
        labels = np.argmin(dist2, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=int)
        for j in range(n):
            c = labels[j]
            new_centroids[c] += X[j]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                new_centroids[c] /= counts[c]
            else:
                new_centroids[c] = X[rng.integers(n)]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xz = (X.astype(np.float32) - mu) / sd

    prototypes = np.zeros((C * M, d), dtype=np.float32)
    for c in range(C):
        Xc = Xz[y == c]
        if len(Xc) == 0:
            prototypes[c * M:(c + 1) * M] = 0
        else:
            prototypes[c * M:(c + 1) * M] = kmeans(Xc, M, rng)

    magic = 0x544F5250  # 'PROT' little-endian
    out = struct.pack("<4i", magic, d, M, C)
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
    if (nw < 16 || hd[0] != 0x544F5250) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], M = hd[2], C = hd[3];
    if (d <= 0 || M <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * (long)C * M * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *proto = sd + d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xz = malloc(sizeof(float) * d);
    float *dist2 = malloc(sizeof(float) * C * M);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) xz[j] = (xi[j] - mu[j]) / sd[j];

        for (int c = 0; c < C; c++) {
            for (int m = 0; m < M; m++) {
                const float *p = proto + ((long)c * M + m) * d;
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = xz[j] - p[j];
                    sum += diff * diff;
                }
                dist2[c * M + m] = sum;
            }
        }

        int best_idx = 0;
        for (int i = 1; i < C * M; i++)
            if (dist2[i] < dist2[best_idx]) best_idx = i;
        unsigned char best_class = best_idx / M;
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```