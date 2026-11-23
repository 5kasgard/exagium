```hypothesis
Family: label-proto (per-class prototype classifier with z-scored Euclidean distance).
Why: Optdigits has non-linear class boundaries; prototypes capture multi-modal per-class distributions better than a single linear separator. Tuning prototypes per class (1-5) on the 20% holdout adapts complexity. Inference is ~C*P*d FLOPs (e.g., 10*3*64=1920), comparable to label-linear.
Failure: If dev accuracy < 0.96, prototypes are insufficient for this task.
```

```python
import json, struct, sys
import numpy as np

SEED = 1
CANDIDATE_P = [1, 2, 3, 4, 5]
KMEANS_ITER = 100

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(data, k, rng, max_iter=100):
    n, d = data.shape
    if n <= k:
        centroids = np.zeros((k, d), dtype=data.dtype)
        centroids[:n] = data
        if n < k:
            centroids[n:] = data[0]
        return centroids
    centroids = np.zeros((k, d), dtype=data.dtype)
    idx = rng.integers(n)
    centroids[0] = data[idx]
    for i in range(1, k):
        diff = data[:, None, :] - centroids[None, :i, :]
        dist2 = np.sum(diff**2, axis=2)
        min_dist2 = np.min(dist2, axis=1)
        probs = min_dist2 / min_dist2.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = data[idx]
    for _ in range(max_iter):
        diff = data[:, None, :] - centroids[None, :, :]
        dist2 = np.sum(diff**2, axis=2)
        labels = np.argmin(dist2, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int32)
        for j in range(n):
            c = labels[j]
            new_centroids[c] += data[j]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                new_centroids[c] /= counts[c]
            else:
                new_centroids[c] = centroids[c]
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X_bytes)
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(n, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)

    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Z = (X - mean) / std

    n_tune = n // 5
    tune_idx = slice(n - n_tune, n)
    train_idx = slice(0, n - n_tune)
    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_tune, y_tune = Z[tune_idx], y[tune_idx]

    best_P = 1
    best_acc = -1.0

    for P in CANDIDATE_P:
        protos = []
        for c in range(C):
            class_data = Z_train[y_train == c]
            if len(class_data) == 0:
                continue
            rng = np.random.default_rng(SEED * 1000 + c * 10 + P)
            cents = kmeans(class_data.astype(np.float32), P, rng, KMEANS_ITER)
            for cent in cents:
                protos.append((c, cent))
        if not protos:
            continue
        proto_classes = np.array([p[0] for p in protos], dtype=np.int64)
        proto_vecs = np.stack([p[1] for p in protos])
        diff = Z_tune[:, None, :] - proto_vecs[None, :, :]
        dist2 = np.sum(diff**2, axis=2)
        pred = proto_classes[np.argmin(dist2, axis=1)]
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_P = P

    protos = []
    for c in range(C):
        class_data = Z[y == c]
        if len(class_data) == 0:
            continue
        rng = np.random.default_rng(SEED * 1000 + c * 10 + best_P)
        cents = kmeans(class_data.astype(np.float32), best_P, rng, KMEANS_ITER)
        for cent in cents:
            protos.append((c, cent))
    proto_classes = np.array([p[0] for p in protos], dtype=np.int64)
    proto_vecs = np.stack([p[1] for p in protos]).astype(np.float32)

    magic = 0x544F5250
    header = struct.pack("<4i", magic, d, C, best_P)
    out = header + mean.astype("<f4").tobytes() + std.astype("<f4").tobytes() + proto_vecs.astype("<f4").tobytes()
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
    int d = hd[1], C = hd[2], P = hd[3];
    if (C < 1 || C > 256 || P < 1 || d < 1) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * d * 2 + 4L * C * P * d;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *protos = std + d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mean[j]) / std[j];
        }
        float best_dist = 1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < P; p++) {
                const float *proto = protos + ((long)c * P + p) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = z[j] - proto[j];
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