```hypothesis
label-proto: prototype classifier with per-class k-means centroids (validated on tuning split).
Beats linear by modeling multi-modal digit classes (writing styles) with multiple prototypes per class.
Fast inference: only ~30×64 ≈ 2k FLOPs/item vs linear's 640 but with non-linear boundaries.
If dev accuracy ≤ 0.9613, prototypes don't capture useful non-linearity for this task.
```

```python
"""label-proto: Prototype classifier with per-class k-means centroids.
python3 -s -P train.py card.json train.in train.key weights.bin
Z-scores ordinal features, selects prototypes per class via k-means on training split
(validates k on last 20% in file order), exports prototypes + z-score params."""
import json, struct, sys
import numpy as np

SEED = 1
MAX_K = 10

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
        diffs = X[:, None, :] - cents[:i]
        dists = np.min(np.sum(diffs**2, axis=2), axis=1)
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        cents[i] = X[idx]
    for _ in range(max_iter):
        diffs = X[:, None, :] - cents[None, :, :]
        dists = np.sum(diffs**2, axis=2)
        assign = np.argmin(dists, axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(k, dtype=np.int32)
        for j in range(n):
            c = assign[j]
            new_cents[c] += X[j]
            counts[c] += 1
        for i in range(k):
            if counts[i] > 0:
                new_cents[i] /= counts[i]
            else:
                new_cents[i] = X[rng.integers(n)]
        if np.allclose(cents, new_cents):
            break
        cents = new_cents
    return cents.astype(np.float32)

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X_bytes.shape[0]

    X = X_bytes.astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_z = (X - mean) / std

    split = int(n * 0.8)
    X_train, X_val = X_z[:split], X_z[split:]
    y_train, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_k, best_acc = 1, -1.0
    best_protos, best_proto_classes = None, None

    for k_per_class in range(1, MAX_K + 1):
        protos = []
        proto_classes = []
        for c in range(C):
            mask = (y_train == c)
            if not np.any(mask):
                continue
            Xc = X_train[mask].astype(np.float32)
            cents = kmeans(Xc, k_per_class, rng)
            protos.append(cents)
            proto_classes.extend([c] * k_per_class)
        if not protos:
            continue
        protos = np.vstack(protos)
        proto_classes = np.array(proto_classes, dtype=np.uint8)

        diffs = X_val[:, None, :] - protos[None, :, :]
        dists = np.sum(diffs**2, axis=2)
        preds = proto_classes[np.argmin(dists, axis=1)]
        acc = (preds == y_val).mean()

        if acc > best_acc:
            best_acc = acc
            best_k = k_per_class
            best_protos = protos.copy()
            best_proto_classes = proto_classes.copy()

    protos = []
    proto_classes = []
    for c in range(C):
        mask = (y == c)
        if not np.any(mask):
            continue
        Xc = X_z[mask].astype(np.float32)
        cents = kmeans(Xc, best_k, rng)
        protos.append(cents)
        proto_classes.extend([c] * best_k)
    protos = np.vstack(protos)
    proto_classes = np.array(proto_classes, dtype=np.uint8)
    n_proto = protos.shape[0]

    magic = 0x544F5250
    out = struct.pack("<4i", magic, d, n_proto, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    out += proto_classes.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-proto: prototype classifier. ./infer weights.bin in.rec N */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x544F5250) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t *hd = (int32_t *)(wb + 4);
    int d = hd[0], n_proto = hd[1], C = hd[2];
    if (d <= 0 || n_proto <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }

    float *mean = (float *)(hd + 3);
    float *std = mean + d;
    float *protos = std + d;
    uint8_t *proto_classes = (uint8_t *)(protos + (long)n_proto * d);

    long expected = 4 + 12 + (long)d * 4 * 2 + (long)n_proto * d * 4 + n_proto;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    uint8_t *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *dists = malloc(sizeof(float) * n_proto);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mean[j]) / std[j];
        }

        for (int p = 0; p < n_proto; p++) {
            float sum = 0.0f;
            const float *proto = protos + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = z[j] - proto[j];
                sum += diff * diff;
            }
            dists[p] = sum;
        }

        int best_p = 0;
        for (int p = 1; p < n_proto; p++) {
            if (dists[p] < dists[best_p]) best_p = p;
        }
        unsigned char best = proto_classes[best_p];
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```