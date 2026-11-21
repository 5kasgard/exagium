```hypothesis
Family: label-lvq (Learning Vector Quantization). 
Why: LVQ learns a small set of prototypes per class, yielding non-linear boundaries well-suited for the clustered structure of handwritten digits. With 3 prototypes/class (30 total), inference costs ~30*64=1920 multiply-adds (3× label-linear), still extremely fast. On optdigits, LVQ historically reaches ~97% accuracy. 
Failure signal: If dev accuracy ≤ 0.961 (label-linear) or instructions/item > 15000, the family does not justify its complexity.
```

```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded. Tunes M (prototypes/class) on last 20% of train, then retrains on all.
Exports: magic 0x4C565131, d, C, M; float32 prototypes[C*M][d]; uint8 proto_class[C*M]."""
import json, math, struct, sys

import numpy as np

SEED = 1
EPOCHS = 30
LR_START, LR_END = 0.1, 0.001
M_CANDIDATES = [1, 2, 3, 4]  # prototypes per class

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def lvq_train(X, y, C, M, epochs, lr_start, lr_end, rng):
    n, d = X.shape
    # Initialize prototypes: k-means++ per class (deterministic)
    protos = []
    proto_cls = []
    for c in range(C):
        Xc = X[y == c]
        if len(Xc) == 0:
            # fallback: random samples from whole set
            idx = rng.choice(n, M, replace=False)
            protos.append(X[idx].astype(np.float32))
        else:
            # k-means++ init
            centroids = [Xc[rng.integers(len(Xc))].astype(np.float32)]
            for _ in range(1, M):
                dists = np.min(np.sum((Xc[:, None] - np.array(centroids))**2, axis=2), axis=1)
                probs = dists / (dists.sum() + 1e-12)
                idx = rng.choice(len(Xc), p=probs)
                centroids.append(Xc[idx].astype(np.float32))
            protos.append(np.array(centroids))
        proto_cls.extend([c] * M)
    protos = np.vstack(protos).astype(np.float32)  # (C*M, d)
    proto_cls = np.array(proto_cls, dtype=np.uint8)
    # LVQ1 training
    for epoch in range(epochs):
        lr = lr_start * (lr_end / lr_start) ** (epoch / max(1, epochs - 1))
        order = rng.permutation(n)
        for i in order:
            x = X[i].astype(np.float32)
            yi = y[i]
            # squared Euclidean distances
            diff = protos - x
            dists = np.sum(diff * diff, axis=1)
            # closest correct and wrong
            same = (proto_cls == yi)
            if not np.any(same) or np.all(same):
                continue
            idx_c = np.argmin(dists + 1e9 * (~same))
            idx_w = np.argmin(dists + 1e9 * same)
            protos[idx_c] += lr * (x - protos[idx_c])
            protos[idx_w] -= lr * (x - protos[idx_w])
    return protos, proto_cls

def accuracy(X, y, protos, proto_cls):
    diff = protos[:, None] - X.T[None]  # (P, n, d) - memory heavy, do batch
    # batch version
    n = X.shape[0]
    correct = 0
    for i in range(0, n, 1024):
        xb = X[i:i+1024].astype(np.float32)
        yb = y[i:i+1024]
        dists = np.sum((protos[:, None] - xb[None])**2, axis=2)  # (P, B)
        pred = proto_cls[np.argmin(dists, axis=0)]
        correct += np.sum(pred == yb)
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    # Tuning split: last 20%
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_M, best_acc = 1, -1.0
    for M in M_CANDIDATES:
        rng_m = np.random.default_rng(SEED + M * 1000)
        protos, pcls = lvq_train(X_tr, y_tr, C, M, EPOCHS, LR_START, LR_END, rng_m)
        acc = accuracy(X_val, y_val, protos, pcls)
        if acc > best_acc:
            best_acc, best_M = acc, M
    # Retrain on full data with best M
    rng_final = np.random.default_rng(SEED + best_M * 1000 + 999)
    protos, pcls = lvq_train(X, y, C, best_M, EPOCHS, LR_START, LR_END, rng_final)
    # Write weights
    magic = 0x4C565131  # "LVQ1"
    P = C * best_M
    out = struct.pack("<4i", magic, d, C, best_M)
    out += protos.astype("<f4").tobytes()
    out += pcls.astype("<u1").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: 1 byte class index.
 * Weights format: magic(0x4C565131), d, C, M; float32 protos[C*M][d]; uint8 proto_class[C*M]. */
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
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], M = hd[3];
    int P = C * M;
    if (d <= 0 || C <= 0 || M <= 0 || P > 10000) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + (long)P * d * 4 + P;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *protos = (const float *)(hd + 4);
    const unsigned char *proto_cls = (const unsigned char *)protos + (long)P * d * 4;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xbuf = malloc(d * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xbuf[j] = (float)xi[j];
        float best_dist = 1e30f; unsigned char best_cls = 0;
        for (int p = 0; p < P; p++) {
            const float *pr = protos + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = xbuf[j] - pr[j];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_cls = proto_cls[p]; }
        }
        put_rec(stdout, &best_cls, 1);
    }
    return 0;
}
```