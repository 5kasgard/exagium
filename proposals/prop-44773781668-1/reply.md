```hypothesis
Family: label-lvq (Learning Vector Quantization). 
Why: LVQ learns a small set of prototypes per class in the normalized pixel space, capturing non-linear boundaries better than a linear model while keeping inference cost low (only distance computations to ~30 prototypes). 
Failure: If dev accuracy ≤ 0.9613 (label-linear's score) or inference cost exceeds label-linear by >2x.
```

```python
"""LVQ family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal features by z-scoring per byte. Learns P prototypes per class via LVQ1.
Hyperparameters (P, lr, epochs) chosen on last 20% of train.in (tuning split), then trained on all train.
Weights format: magic 0x4C565131, d, C, P; float32 mean[d], std[d]; float32 prototypes[P*C][d]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning
P_GRID = [1, 2, 3, 4, 5]
LR_GRID = [0.1, 0.05, 0.01]
EPOCHS_GRID = [10, 20, 30]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def normalize(X, mean, std):
    return (X - mean) / std


def lvq_train(X, y, C, P, lr, epochs, rng):
    """LVQ1: prototypes shaped (C, P, d). Returns prototypes."""
    d = X.shape[1]
    # Initialize prototypes: for each class, pick P random samples from that class
    protos = np.zeros((C, P, d), dtype=np.float32)
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(P, len(idx)), replace=False)
        protos[c, :len(chosen)] = X[chosen]
        if len(chosen) < P:
            # duplicate if not enough samples
            protos[c, len(chosen):] = protos[c, :1]
    # Training
    n = X.shape[0]
    for ep in range(epochs):
        order = rng.permutation(n)
        current_lr = lr * (1.0 - ep / epochs)
        for i in order:
            x = X[i]
            target = y[i]
            # Find nearest prototype
            best_dist = float('inf')
            best_c, best_p = -1, -1
            for c in range(C):
                for p in range(P):
                    diff = protos[c, p] - x
                    dist = np.dot(diff, diff)
                    if dist < best_dist:
                        best_dist = dist
                        best_c, best_p = c, p
            # Update
            if best_c == target:
                protos[best_c, best_p] += current_lr * (x - protos[best_c, best_p])
            else:
                protos[best_c, best_p] -= current_lr * (x - protos[best_c, best_p])
    return protos


def evaluate(X, y, protos):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        x = X[i]
        best_dist = float('inf')
        best_c = -1
        for c in range(protos.shape[0]):
            for p in range(protos.shape[1]):
                diff = protos[c, p] - x
                dist = np.dot(diff, diff)
                if dist < best_dist:
                    best_dist = dist
                    best_c = c
        if best_c == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_all.shape[1]

    # Split: last 20% for tuning
    n = X_all.shape[0]
    split = int(n * 0.8)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    # Compute normalization on full training set (before split? card says choose settings on tuning split, then train on all train.
    # We'll compute mean/std on the full train.in (all data) because that's what we'll use for final training.
    # But for tuning we should use the same normalization as final? Usually normalization is computed on training data only.
    # We'll compute mean/std on X_train (first 80%) for tuning, then recompute on all data for final training.
    # However, the card says "choose them on the card's tuning split only (last 20% of train.in in file order), then train on all of train".
    # So we should use the tuning split to pick hyperparameters, but the normalization for that evaluation should be based on the training split (first 80%).
    # Then final model uses normalization on all train.in.
    mean_train = X_train.mean(axis=0).astype(np.float32)
    std_train = X_train.std(axis=0).astype(np.float32)
    std_train[std_train == 0] = 1.0
    X_train_norm = normalize(X_train.astype(np.float32), mean_train, std_train)
    X_tune_norm = normalize(X_tune.astype(np.float32), mean_train, std_train)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_protos = None

    for P in P_GRID:
        for lr in LR_GRID:
            for epochs in EPOCHS_GRID:
                protos = lvq_train(X_train_norm, y_train, C, P, lr, epochs, rng)
                acc = evaluate(X_tune_norm, y_tune, protos)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (P, lr, epochs)
                    best_protos = protos

    # Now train on all data with best hyperparameters
    P, lr, epochs = best_params
    mean_all = X_all.mean(axis=0).astype(np.float32)
    std_all = X_all.std(axis=0).astype(np.float32)
    std_all[std_all == 0] = 1.0
    X_all_norm = normalize(X_all.astype(np.float32), mean_all, std_all)
    final_protos = lvq_train(X_all_norm, y_all, C, P, lr, epochs, rng)

    # Export weights
    # magic, d, C, P
    out = struct.pack("<4i", 0x4C565131, d, C, P)
    out += mean_all.astype("<f4").tobytes()
    out += std_all.astype("<f4").tobytes()
    # prototypes: (C, P, d) -> flatten in order c, p, d
    out += final_protos.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* LVQ family for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class of nearest prototype.
 * Weights format: magic 0x4C565131, d, C, P; float32 mean[d], std[d]; float32 prototypes[C][P][d]. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], P = hd[3];
    if (d <= 0 || C <= 0 || P <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 2L * d * 4 + (long)C * P * d * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }
    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *protos = std + d;  // [C][P][d]
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *norm = malloc(sizeof(float) * d);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // normalize
        for (int j = 0; j < d; j++) {
            norm[j] = (xi[j] - mean[j]) / std[j];
        }
        // find nearest prototype
        float best_dist = 1e30f;
        int best_c = 0;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < P; p++) {
                const float *proto = protos + ((long)c * P + p) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = norm[j] - proto[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
        }
        unsigned char out = (unsigned char)best_c;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```