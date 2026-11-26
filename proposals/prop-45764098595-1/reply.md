```hypothesis
Family: label-proto (prototype-based classifier). Learns a small set of prototypes per class via k-means on the tuning split, then classifies by nearest prototype (Euclidean distance). Should beat label-linear by capturing non-linear class boundaries while keeping inference extremely fast (O(C*P*d) operations). If dev accuracy does not exceed 0.9613, the family fails.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_ITER = 20
PROTO_RANGE = range(1, 11)  # prototypes per class to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_class(X, P, rng):
    """Deterministic k-means on a single class's data. Returns (P, d) prototypes."""
    n, d = X.shape
    if n <= P:
        # duplicate if needed
        idx = np.arange(n)
        idx = np.resize(idx, P)
        return X[idx].astype(np.float32)
    # deterministic initialization: shuffle with fixed seed and take first P
    order = rng.permutation(n)
    centroids = X[order[:P]].astype(np.float64)
    for _ in range(MAX_ITER):
        # assign
        diff = X[:, None, :] - centroids[None, :, :]  # (n, P, d)
        dist2 = np.sum(diff * diff, axis=2)
        labels = np.argmin(dist2, axis=1)
        # update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(P, dtype=np.int64)
        for i in range(n):
            c = labels[i]
            new_centroids[c] += X[i]
            counts[c] += 1
        for c in range(P):
            if counts[c] > 0:
                new_centroids[c] /= counts[c]
            else:
                # empty cluster: reinitialize to a random point
                new_centroids[c] = X[rng.integers(n)]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids.astype(np.float32)


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    # normalize to [0,1]
    X = X_bytes.astype(np.float32) / 16.0

    # split: last 20% for tuning
    n = len(X)
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    rng = np.random.default_rng(SEED)

    best_P = 1
    best_acc = -1.0
    for P in PROTO_RANGE:
        # train prototypes on train split
        protos = np.zeros((C, P, d), dtype=np.float32)
        for c in range(C):
            Xc = X_train[y_train == c]
            protos[c] = kmeans_class(Xc, P, rng)
        # evaluate on tune split
        # compute distances to all prototypes
        # tune: (n_tune, d), protos: (C, P, d) -> reshape to (C*P, d)
        flat_protos = protos.reshape(-1, d)
        proto_class = np.repeat(np.arange(C), P)
        # (n_tune, C*P)
        diff = X_tune[:, None, :] - flat_protos[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        pred = proto_class[np.argmin(dist2, axis=1)]
        acc = np.mean(pred == y_tune)
        if acc > best_acc:
            best_acc = acc
            best_P = P

    # retrain on all data with best_P
    P = best_P
    protos = np.zeros((C, P, d), dtype=np.float32)
    for c in range(C):
        Xc = X[y == c]
        protos[c] = kmeans_class(Xc, P, rng)

    # write weights.bin
    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<4i", magic, d, C, P)
    out += protos.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], P = hd[3];
    if (d <= 0 || C <= 0 || P <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (long)C * P * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *protos = (const float *)(hd + 4);  // shape (C, P, d) row-major
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // normalize input to [0,1]
        float xf[64];
        for (int j = 0; j < d; j++) xf[j] = xi[j] / 16.0f;

        // find nearest prototype
        float best_dist = 1e30f;
        int best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < P; p++) {
                const float *proto = protos + ((long)c * P + p) * d;
                float dist2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = xf[j] - proto[j];
                    dist2 += diff * diff;
                }
                if (dist2 < best_dist) {
                    best_dist = dist2;
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