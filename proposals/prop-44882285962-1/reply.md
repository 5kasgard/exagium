```hypothesis
Family: label-vq (vector quantization classifier). Learns a small set of prototypes per class via k-means on normalized pixel space; inference is nearest-prototype search. Should beat linear on optdigits because digits form tight clusters per class. If dev accuracy <= 0.9613, the family does not justify its extra complexity.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_ITER = 20
CANDIDATE_P = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(data, k, rng):
    """data: (n, d) float32, k: int. Returns (k, d) float32 centroids."""
    n, d = data.shape
    if n <= k:
        return data[:k].copy()
    idx = rng.choice(n, k, replace=False)
    centroids = data[idx].copy()
    for _ in range(MAX_ITER):
        # assignment
        diff = data[:, None, :] - centroids[None, :, :]  # (n, k, d)
        dist2 = (diff * diff).sum(axis=2)
        labels = dist2.argmin(axis=1)
        # update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        for i in range(n):
            c = labels[i]
            new_centroids[c] += data[i]
            counts[c] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                new_centroids[j] = data[rng.integers(n)]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = len(X)
    n_tune = n // 5
    idx_train = np.arange(n - n_tune)
    idx_tune = np.arange(n - n_tune, n)

    X_train, y_train = X[idx_train], y[idx_train]
    X_tune, y_tune = X[idx_tune], y[idx_tune]

    best_p = 1
    best_acc = -1.0

    for p in CANDIDATE_P:
        # compute mean/std on train split
        mu = X_train.mean(axis=0).astype(np.float32)
        sd = X_train.std(axis=0).astype(np.float32)
        sd[sd == 0] = 1.0
        Z_train = (X_train.astype(np.float32) - mu) / sd
        Z_tune = (X_tune.astype(np.float32) - mu) / sd

        # learn prototypes per class
        prototypes = []
        proto_labels = []
        for c in range(C):
            class_data = Z_train[y_train == c]
            if len(class_data) == 0:
                continue
            rng = np.random.default_rng(SEED + c * 1000 + p * 10000)
            centroids = kmeans(class_data, p, rng)
            prototypes.append(centroids)
            proto_labels.append(np.full(p, c, dtype=np.uint8))
        if not prototypes:
            continue
        prototypes = np.vstack(prototypes).astype(np.float32)  # (K, d)
        proto_labels = np.concatenate(proto_labels)  # (K,)
        K = prototypes.shape[0]

        # classify tune set
        # distance: (n_tune, K)
        diff = Z_tune[:, None, :] - prototypes[None, :, :]
        dist2 = (diff * diff).sum(axis=2)
        preds = proto_labels[dist2.argmin(axis=1)]
        acc = (preds == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_p = p

    # retrain on all data with best_p
    mu_all = X.mean(axis=0).astype(np.float32)
    sd_all = X.std(axis=0).astype(np.float32)
    sd_all[sd_all == 0] = 1.0
    Z_all = (X.astype(np.float32) - mu_all) / sd_all

    prototypes = []
    proto_labels = []
    for c in range(C):
        class_data = Z_all[y == c]
        if len(class_data) == 0:
            continue
        rng = np.random.default_rng(SEED + c * 1000 + best_p * 10000)
        centroids = kmeans(class_data, best_p, rng)
        prototypes.append(centroids)
        proto_labels.append(np.full(best_p, c, dtype=np.uint8))
    prototypes = np.vstack(prototypes).astype(np.float32)  # (K, d)
    proto_labels = np.concatenate(proto_labels)  # (K,)
    K = prototypes.shape[0]

    # write weights: magic, d, K, C; mean[d], std[d]; prototypes[K][d]; labels[K]
    magic = 0x56514C42  # 'VQLB' little-endian
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += mu_all.astype("<f4").tobytes()
    out += sd_all.astype("<f4").tobytes()
    out += prototypes.astype("<f4").tobytes()
    out += proto_labels.astype(np.uint8).tobytes()
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
    if (nw < 16 || hd[0] != 0x56514C42) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 64 || C != 10 || K <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * K * d + K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *prototypes = std + d;
    const unsigned char *proto_labels = (const unsigned char *)(prototypes + (long)K * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *znorm = (float *)malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            float v = (float)xi[j];
            float s = std[j];
            znorm[j] = (v - mean[j]) / (s != 0.0f ? s : 1.0f);
        }
        float best_dist = 1e30f;
        unsigned char best_label = 0;
        for (int p = 0; p < K; p++) {
            const float *proto = prototypes + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = znorm[j] - proto[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_label = proto_labels[p];
            }
        }
        put_rec(stdout, &best_label, 1);
    }
    return 0;
}
```