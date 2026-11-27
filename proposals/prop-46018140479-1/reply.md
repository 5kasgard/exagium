```hypothesis
Family: label-lvq (Learning Vector Quantization). Learns a small set of prototypes per class (e.g., 5 per class = 50 total) using LVQ1. Provides non-linear boundaries at lower inference cost than KNN (which uses all 2867 training points). On optdigits (64 ordinal features, 10 classes), 50 prototypes should exceed linear accuracy (0.9613) and approach MLP/GBDT, while being faster than KNN (2.2M Ir/item). Failure if dev accuracy < 0.96 or inference cost > label-linear (5k Ir/item).
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED = 1
EPOCHS = 20
ALPHA_START = 0.1
ALPHA_END = 0.01
MAX_PROTOS_PER_CLASS = 10  # will tune 1..10 on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def lvq_train(X, y, C, protos_per_class, rng):
    """Return prototypes (P, d) and labels (P,) after LVQ1 on given data."""
    d = X.shape[1]
    P = C * protos_per_class
    # Initialize prototypes: pick protos_per_class random samples per class
    protos = np.zeros((P, d), dtype=np.float32)
    proto_labels = np.zeros(P, dtype=np.uint8)
    idx = 0
    for c in range(C):
        class_mask = (y == c)
        class_samples = X[class_mask]
        if len(class_samples) == 0:
            continue
        chosen = rng.choice(len(class_samples), size=protos_per_class, replace=False)
        protos[idx:idx+protos_per_class] = class_samples[chosen].astype(np.float32)
        proto_labels[idx:idx+protos_per_class] = c
        idx += protos_per_class
    # LVQ1 epochs
    n = X.shape[0]
    for epoch in range(EPOCHS):
        alpha = ALPHA_START + (ALPHA_END - ALPHA_START) * epoch / (EPOCHS - 1) if EPOCHS > 1 else ALPHA_START
        order = rng.permutation(n)
        for i in order:
            x = X[i].astype(np.float32)
            yi = y[i]
            # squared Euclidean distances
            diff = protos - x
            dists = np.sum(diff * diff, axis=1)
            nearest = np.argmin(dists)
            if proto_labels[nearest] == yi:
                protos[nearest] += alpha * (x - protos[nearest])
            else:
                protos[nearest] -= alpha * (x - protos[nearest])
    return protos, proto_labels

def evaluate(X, y, protos, proto_labels):
    """Return accuracy on given data."""
    n = X.shape[0]
    correct = 0
    for i in range(n):
        x = X[i].astype(np.float32)
        diff = protos - x
        dists = np.sum(diff * diff, axis=1)
        nearest = np.argmin(dists)
        if proto_labels[nearest] == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning, first 80% for training
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_acc = -1.0
    best_protos_per_class = 1
    best_protos = None
    best_labels = None

    for ppc in range(1, MAX_PROTOS_PER_CLASS + 1):
        protos, labels = lvq_train(X_train, y_train, C, ppc, rng)
        acc = evaluate(X_tune, y_tune, protos, labels)
        if acc > best_acc:
            best_acc = acc
            best_protos_per_class = ppc
            best_protos = protos
            best_labels = labels

    # Retrain on full data with best hyperparameter
    protos, labels = lvq_train(X, y, C, best_protos_per_class, rng)
    P = protos.shape[0]

    # Write weights.bin: magic, d, C, P, prototypes (P*d float32), labels (P uint8)
    magic = 0x50524F54  # "PROT"
    out = struct.pack("<4I", magic, d, C, P)
    out += protos.astype("<f4").tobytes()
    out += labels.astype(np.uint8).tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* LVQ family for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class of nearest prototype. */
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
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], P = hd[3];
    if (C < 1 || C > 256 || d < 1 || P < 1) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16 + (long)P * d * 4 + P;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *protos = (const float *)(hd + 4);
    const unsigned char *labels = (const unsigned char *)(protos + (long)P * d);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_dist = 1e30f;
        unsigned char best_label = 0;
        for (int p = 0; p < P; p++) {
            const float *w = protos + (long)p * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - w[j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_label = labels[p];
            }
        }
        put_rec(stdout, &best_label, 1);
    }
    return 0;
}
```