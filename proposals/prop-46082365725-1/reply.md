```hypothesis
Family: label-lvq (Learning Vector Quantization). LVQ learns a small set of prototypical examples per class; classification is nearest-prototype (Euclidean). On 8x8 digits, prototypes capture canonical stroke shapes, giving non-linear boundaries with only ~50-100 prototypes. Expect dev accuracy >0.97 (beating label-linear's 0.9613) at ~5000 Ir/item (faster than label-mlp16's 8685). Failure: if accuracy ≤0.965 or Ir/item ≥10000, the approach is not competitive.
```
```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread. Uses per-pixel z-scoring (global mean/std).
Tunes prototypes-per-class K on last 20% of train, then retrains on all train.
Exports: magic, d, C, K, mean[d], std[d], prototypes[K*C][d] (float32), proto_class[K*C] (uint8)."""
import json, math, struct, sys

import numpy as np

SEED = 1
K_CHOICES = [4, 6, 8, 10]          # prototypes per class to try
EPOCHS = 30                        # LVQ1 epochs per tuning run
LR0 = 0.3                          # initial learning rate
LR_DECAY = 0.995                   # per-step decay


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def normalize(X, mean, std):
    return (X.astype(np.float32) - mean) / std


def lvq_train(X, y, C, K, rng, epochs=EPOCHS, lr0=LR0, lr_decay=LR_DECAY):
    """LVQ1: returns prototypes [K*C, d], proto_class [K*C]."""
    d = X.shape[1]
    # Initialize: K random samples per class
    protos = []
    pcls = []
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(K, len(idx)), replace=False)
        protos.append(X[chosen])
        pcls.append(np.full(len(chosen), c, dtype=np.uint8))
    protos = np.vstack(protos).astype(np.float32)
    pcls = np.concatenate(pcls)
    n_protos = protos.shape[0]

    n = X.shape[0]
    step = 0
    total_steps = epochs * n
    for _ in range(epochs):
        order = rng.permutation(n)
        for idx in order:
            x = X[idx]
            y_true = y[idx]
            # distances to all prototypes
            diff = protos - x
            dist2 = np.sum(diff * diff, axis=1)
            best = np.argmin(dist2)
            # LVQ1 update
            lr = lr0 * (lr_decay ** step)
            step += 1
            if pcls[best] == y_true:
                protos[best] += lr * (x - protos[best])
            else:
                protos[best] -= lr * (x - protos[best])
    return protos, pcls


def evaluate(X, y, protos, pcls):
    diff = protos[None, :, :] - X[:, None, :]          # [n, n_protos, d]
    dist2 = np.sum(diff * diff, axis=2)                # [n, n_protos]
    best = np.argmin(dist2, axis=1)
    pred = pcls[best]
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    n = X.shape[0]
    split = int(0.8 * n)  # last 20% for tuning
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    # Global z-scoring on training split
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0

    X_train_n = normalize(X_train, mean, std)
    X_tune_n = normalize(X_tune, mean, std)

    rng = np.random.default_rng(SEED)
    best_K = K_CHOICES[0]
    best_acc = -1.0

    for K in K_CHOICES:
        r = np.random.default_rng(SEED + K * 1000)  # independent seed per K
        protos, pcls = lvq_train(X_train_n, y_train, C, K, r)
        acc = evaluate(X_tune_n, y_tune, protos, pcls)
        if acc > best_acc:
            best_acc = acc
            best_K = K

    # Retrain on full data with best_K
    X_all_n = normalize(X, mean, std)
    r_final = np.random.default_rng(SEED + 9999)
    protos, pcls = lvq_train(X_all_n, y_all_n, C, best_K, r_final)

    # Pack weights
    n_protos = protos.shape[0]
    magic = 0x4C565131  # "LVQ1"
    out = struct.pack("<4i", magic, d, C, best_K)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += protos.astype("<f4").tobytes()
    out += pcls.astype(np.uint8).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Weights: magic, d, C, K, mean[d], std[d], protos[K*C][d] (float32), cls[K*C] (uint8).
 * Each input record: d bytes. Output: 1 byte class (argmin Euclidean distance to prototypes). */
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
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4C565131) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (d != 64 || C != 10 || K < 1 || K > 20) { fprintf(stderr, "bad header dims\n"); return 3; }
    int n_protos = K * C;
    long expected = 16L + 4L * d * 2 + 4L * n_protos * d + n_protos;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    float *mean = (float *)(wb + 16);
    float *std  = mean + d;
    float *protos = std + d;
    unsigned char *pcls = (unsigned char *)(protos + n_protos * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xnorm[j] = (xi[j] - mean[j]) / std[j];
        float best_dist = 1e30f; unsigned char best_cls = 0;
        for (int p = 0; p < n_protos; p++) {
            float dist = 0.0f;
            const float *pr = protos + (long)p * d;
            for (int j = 0; j < d; j++) {
                float diff = xnorm[j] - pr[j];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_cls = pcls[p]; }
        }
        put_rec(stdout, &best_cls, 1);
    }
    return 0;
}
```