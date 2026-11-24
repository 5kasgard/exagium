```hypothesis
Prototype-based classifier (label-proto): learn M centroids per class via k-means on the training data,
predict by nearest centroid. For optdigits' 8x8 ordinal pixels, a few prototypes capture each digit's
stroke variations better than a linear decision boundary. Tuning M on the held-out 20% should yield
>0.965 dev accuracy at ~3000 Ir/item (C*M*d = 10*4*64 mul-adds). If dev accuracy <= 0.961 (linear)
or cost exceeds 5000 Ir/item, the approach fails.
```
```python
"""Prototype family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Tunes prototypes-per-class M on last 20% of train, then trains on all.
Exports: magic 0x50524F54, d, M, C; float32 centroids[C][M][d] (row-major)."""
import json, math, struct, sys

import numpy as np

SEED = 1
CANDIDATE_M = [2, 3, 4, 5, 6, 8]
KMEANS_MAX_ITER = 100
KMEANS_TOL = 1e-4


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_class(X, M, rng):
    """X: (n, d), return (M, d) centroids. Deterministic init: random permutation."""
    n, d = X.shape
    if n <= M:
        # pad with duplicates if needed
        centroids = X[:M].copy()
        if n < M:
            centroids = np.vstack([centroids, np.tile(centroids[-1:], (M - n, 1))])
        return centroids.astype(np.float32)
    idx = rng.permutation(n)[:M]
    cents = X[idx].astype(np.float32).copy()
    Xf = X.astype(np.float32)
    for _ in range(KMEANS_MAX_ITER):
        # assign
        diff = Xf[:, None, :] - cents[None, :, :]  # (n, M, d)
        dist2 = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(M, dtype=np.int32)
        for i in range(n):
            c = assign[i]
            new_cents[c] += Xf[i]
            counts[c] += 1
        for m in range(M):
            if counts[m] > 0:
                new_cents[m] /= counts[m]
            else:
                # reinitialize empty cluster to a random point
                new_cents[m] = Xf[rng.integers(n)]
        shift = np.max(np.sqrt(np.sum((new_cents - cents) ** 2, axis=1)))
        cents = new_cents
        if shift < KMEANS_TOL:
            break
    return cents


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # split: last 20% for tuning
    split = int(n * 0.8)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_M, best_acc = CANDIDATE_M[0], -1.0

    for M in CANDIDATE_M:
        # train prototypes on training split
        cents_list = []
        for c in range(C):
            Xc = X_tr[y_tr == c]
            cents = kmeans_class(Xc, M, rng)
            cents_list.append(cents)
        cents_all = np.stack(cents_list)  # (C, M, d)

        # evaluate on validation
        correct = 0
        for i in range(len(X_va)):
            x = X_va[i].astype(np.float32)
            diff = cents_all - x[None, None, :]  # (C, M, d)
            dist2 = np.sum(diff * diff, axis=2)  # (C, M)
            min_idx = np.argmin(dist2)
            pred = min_idx // M
            if pred == y_va[i]:
                correct += 1
        acc = correct / len(X_va)
        if acc > best_acc:
            best_acc, best_M = acc, M

    # retrain on full data with best_M
    rng = np.random.default_rng(SEED)
    cents_list = []
    for c in range(C):
        Xc = X[y == c]
        cents = kmeans_class(Xc, best_M, rng)
        cents_list.append(cents)
    cents_all = np.stack(cents_list).astype(np.float32)  # (C, M, d)

    # write weights
    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<4i", magic, d, best_M, C)
    out += cents_all.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Prototype classifier (label-proto): ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, class of nearest centroid.
 * Weights format: int32 magic 0x50524F54, d, M, C; float32 centroids[C][M][d] row-major. */
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
    int d = hd[1], M = hd[2], C = hd[3];
    if (d <= 0 || M <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (long)C * M * d;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }
    const float *cents = (const float *)(hd + 4);  // [C][M][d]
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist2 = 1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int m = 0; m < M; m++) {
                const float *cent = cents + ((long)c * M + m) * d;
                float dist2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = (float)xi[j] - cent[j];
                    dist2 += diff * diff;
                }
                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    best_class = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```