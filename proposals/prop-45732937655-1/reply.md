```hypothesis
Propose "label-proto": a prototype-based classifier that learns M prototypes per class via k-means on the training data. Optdigits digits form tight clusters in 64D pixel space; multiple prototypes capture intra-class variation better than a single linear boundary. Inference computes squared Euclidean distance to all C*M prototypes (C=10, M≈5–10) and picks the nearest prototype's class. This should beat label-linear (0.9613) at similar or lower CPU cost (≈5000 FLOPs/item). If dev accuracy ≤ 0.9613, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# hyperparameter search space for M (prototypes per class)
M_CANDIDATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEED = 1
KMEANS_ITERS = 30
TUNE_FRACTION = 0.2  # last 20% for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_class(Xc, M, rng):
    """Xc: (n_c, d) float32. Return (M, d) centroids."""
    n_c, d = Xc.shape
    if n_c <= M:
        # not enough points: pad with zeros
        cents = np.zeros((M, d), dtype=np.float32)
        cents[:n_c] = Xc
        return cents
    # deterministic initialization: pick M random indices without replacement
    idx = rng.choice(n_c, size=M, replace=False)
    cents = Xc[idx].copy()
    for _ in range(KMEANS_ITERS):
        # assignment
        diff = Xc[:, None, :] - cents[None, :, :]  # (n_c, M, d)
        dist2 = np.sum(diff * diff, axis=2)  # (n_c, M)
        assign = np.argmin(dist2, axis=1)  # (n_c,)
        # update
        new_cents = np.zeros_like(cents)
        counts = np.zeros(M, dtype=np.int32)
        for i in range(n_c):
            c = assign[i]
            new_cents[c] += Xc[i]
            counts[c] += 1
        for m in range(M):
            if counts[m] > 0:
                new_cents[m] /= counts[m]
            else:
                # reinitialize empty cluster to a random point
                new_cents[m] = Xc[rng.integers(n_c)]
        if np.allclose(cents, new_cents):
            cents = new_cents
            break
        cents = new_cents
    return cents


def predict_proto(X, centroids, C, M, d):
    """X: (n, d) uint8. centroids: (C, M, d) float32. Return predicted class (n,) int."""
    n = X.shape[0]
    Xf = X.astype(np.float32)
    best_class = np.zeros(n, dtype=np.int64)
    best_dist = np.full(n, np.inf, dtype=np.float32)
    for c in range(C):
        for m in range(M):
            cent = centroids[c, m]  # (d,)
            diff = Xf - cent
            dist2 = np.sum(diff * diff, axis=1)
            better = dist2 < best_dist
            best_dist[better] = dist2[better]
            best_class[better] = c
    return best_class


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = X_all.shape[0]
    # split: last 20% for tuning
    n_tune = int(math.ceil(n_all * TUNE_FRACTION))
    n_train = n_all - n_tune
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_tune, y_tune = X_all[n_train:], y_all[n_train:]

    rng = np.random.default_rng(SEED)

    best_M = 1
    best_acc = -1.0

    for M in M_CANDIDATES:
        # train on train split
        centroids = np.zeros((C, M, d), dtype=np.float32)
        for c in range(C):
            Xc = X_train[y_train == c].astype(np.float32)
            # use a derived seed per class for reproducibility
            class_rng = np.random.default_rng(SEED + c + 1000 * M)
            centroids[c] = kmeans_class(Xc, M, class_rng)
        # evaluate on tune split
        pred = predict_proto(X_tune, centroids, C, M, d)
        acc = np.mean(pred == y_tune)
        if acc > best_acc:
            best_acc = acc
            best_M = M

    # retrain on all data with best_M
    M = best_M
    centroids = np.zeros((C, M, d), dtype=np.float32)
    for c in range(C):
        Xc = X_all[y_all == c].astype(np.float32)
        class_rng = np.random.default_rng(SEED + c + 1000 * M)
        centroids[c] = kmeans_class(Xc, M, class_rng)

    # write weights.bin
    # format: uint32 magic, d, C, M; then float32 centroids[C, M, d] in row-major
    magic = 0x544F5250  # "PROT" little-endian
    out = struct.pack("<4I", magic, d, C, M)
    out += centroids.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-proto inference: ./infer weights.bin in.rec N */
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
    if (nw < 16 || hd[0] != 0x544F5250) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], M = hd[3];
    if (C < 1 || C > 256 || M < 1 || d < 1 ||
        nw != 16 + 4L * (long)C * M * d) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *centroids = (const float *)(hd + 4);  // [C][M][d]
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char best;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f;
        best = 0;
        for (int c = 0; c < C; c++) {
            for (int m = 0; m < M; m++) {
                const float *cent = centroids + ((long)c * M + m) * d;
                float dist2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = (float)xi[j] - cent[j];
                    dist2 += diff * diff;
                }
                if (dist2 < best_dist) {
                    best_dist = dist2;
                    best = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```