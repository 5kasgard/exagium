```hypothesis
Propose "label-proto": a prototype-based classifier using per-class k-means centroids.  
Optdigits classes form compact clusters in pixel space; a small set of prototypes per class captures non-linear boundaries while being far cheaper than full k-NN.  
We choose the number of prototypes per class (K ∈ {1..5}) on the tuning split (last 20% of train).  
If dev accuracy falls below label-linear (0.9613) or inference cost exceeds ~5× linear, the family does not meet its goal.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
K_CHOICES = [1, 2, 3, 4, 5]   # prototypes per class
MAX_KMEANS_ITER = 100
TOL = 1e-4

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans(X, K, rng):
    """Deterministic k-means++ initialization + Lloyd's algorithm."""
    n, d = X.shape
    if n <= K:
        return X[:K]
    centroids = np.empty((K, d), dtype=np.float32)
    # k-means++
    idx = rng.integers(n)
    centroids[0] = X[idx]
    dist2 = np.full(n, np.inf, dtype=np.float32)
    for c in range(1, K):
        diff = X - centroids[c-1]
        d2 = np.sum(diff * diff, axis=1)
        np.minimum(dist2, d2, out=dist2)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centroids[c] = X[idx]
    # Lloyd
    for _ in range(MAX_KMEANS_ITER):
        # assign
        diff = X[:, None, :] - centroids[None, :, :]   # (n, K, d)
        d2 = np.sum(diff * diff, axis=2)               # (n, K)
        assign = np.argmin(d2, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.int32)
        for i in range(n):
            a = assign[i]
            new_centroids[a] += X[i]
            counts[a] += 1
        for k in range(K):
            if counts[k] > 0:
                new_centroids[k] /= counts[k]
            else:
                new_centroids[k] = centroids[k]
        shift = np.sqrt(np.sum((new_centroids - centroids)**2))
        centroids = new_centroids
        if shift < TOL:
            break
    return centroids

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_all.shape[0]
    # split: last 20% for tuning
    split = int(n * 0.8)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    rng = np.random.default_rng(SEED)

    # standardize on training portion
    mu = X_train.mean(axis=0).astype(np.float32)
    sd = X_train.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    def standardize(X):
        return ((X.astype(np.float32) - mu) / sd).astype(np.float32)

    X_train_std = standardize(X_train)
    X_val_std = standardize(X_val)

    best_K = 1
    best_acc = -1.0
    for K in K_CHOICES:
        # build prototypes per class on training portion
        protos = []
        proto_classes = []
        for c in range(C):
            Xc = X_train_std[y_train == c]
            if len(Xc) == 0:
                continue
            cents = kmeans(Xc, min(K, len(Xc)), rng)
            protos.append(cents)
            proto_classes.extend([c] * len(cents))
        if not protos:
            continue
        protos = np.vstack(protos).astype(np.float32)
        proto_classes = np.array(proto_classes, dtype=np.uint8)
        # evaluate on validation
        diff = X_val_std[:, None, :] - protos[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        preds = proto_classes[np.argmin(d2, axis=1)]
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_K = K

    # final training on all data with best_K
    mu_full = X_all.mean(axis=0).astype(np.float32)
    sd_full = X_all.std(axis=0).astype(np.float32)
    sd_full[sd_full == 0] = 1.0
    X_all_std = ((X_all.astype(np.float32) - mu_full) / sd_full).astype(np.float32)

    protos = []
    proto_classes = []
    for c in range(C):
        Xc = X_all_std[y_all == c]
        if len(Xc) == 0:
            continue
        cents = kmeans(Xc, min(best_K, len(Xc)), rng)
        protos.append(cents)
        proto_classes.extend([c] * len(cents))
    protos = np.vstack(protos).astype(np.float32)   # (P, d)
    proto_classes = np.array(proto_classes, dtype=np.uint8)
    P = protos.shape[0]

    # write weights.bin
    # magic "PROT" = 0x544F5250? Let's use 0x50524F54 (little-endian 'PROT')
    magic = 0x50524F54
    out = struct.pack("<4i", magic, d, C, best_K)
    out += mu_full.astype("<f4").tobytes()
    out += sd_full.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    out += proto_classes.tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x50524F54) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1];
    int C = (int)hd[2];
    int K = (int)hd[3];
    if (d != 64 || C != 10 || K < 1 || K > 5) { fprintf(stderr, "bad weights header\n"); return 3; }
    int P = K * C;
    size_t expected = 16 + 4L * d * 2 + 4L * P * d + P;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *mu = (const float *)(wb + 16);
    const float *sd = mu + d;
    const float *protos = sd + d;
    const unsigned char *proto_class = (const unsigned char *)(protos + (size_t)P * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_std = malloc(sizeof(float) * d);
    float *d2 = malloc(sizeof(float) * P);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            x_std[j] = (xi[j] - mu[j]) / sd[j];
        }
        for (int p = 0; p < P; p++) {
            float sum = 0.0f;
            const float *pr = protos + (size_t)p * d;
            for (int j = 0; j < d; j++) {
                float diff = x_std[j] - pr[j];
                sum += diff * diff;
            }
            d2[p] = sum;
        }
        int best_p = 0;
        for (int p = 1; p < P; p++) if (d2[p] < d2[best_p]) best_p = p;
        unsigned char pred = proto_class[best_p];
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```