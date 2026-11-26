```hypothesis
We propose a Learning Vector Quantization (LVQ) family. LVQ learns a small set of prototypes per class and classifies by nearest prototype (Euclidean distance). It is non‑linear, has very few parameters (≈50 prototypes × 64 features), and inference is fast (only a few thousand FLOPs/item). On optdigits the pixel patterns have non‑linear boundaries, so LVQ should beat the linear baseline (0.9613). If dev accuracy ≤ 0.9613, the family does not improve over label‑linear.
```

```python
"""LVQ family for label tasks with fixed‑size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyper‑parameter grid for tuning split (last 20% of train.in)
GRID = {
    "prototypes_per_class": [3, 5, 7],
    "epochs": [10, 20, 30],
    "alpha0": [0.1, 0.05, 0.01],
}

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def lvq_train(X, y, C, P, epochs, alpha0, rng):
    """X: (n, d) float32, y: (n,) int64. Returns prototypes (C*P, d) float32 and their labels."""
    n, d = X.shape
    # Initialize prototypes by randomly picking P samples from each class
    protos = []
    proto_labels = []
    for c in range(C):
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=P, replace=False)
        protos.append(X[chosen])
        proto_labels.append(np.full(P, c, dtype=np.int64))
    protos = np.vstack(protos).astype(np.float32)          # (C*P, d)
    proto_labels = np.concatenate(proto_labels)            # (C*P,)
    n_protos = protos.shape[0]

    for ep in range(epochs):
        alpha = alpha0 * (1.0 - ep / epochs)               # linear decay
        order = rng.permutation(n)
        for i in order:
            x = X[i]
            # squared Euclidean distances
            diff = protos - x
            dist2 = np.sum(diff * diff, axis=1)
            best = np.argmin(dist2)
            if proto_labels[best] == y[i]:
                protos[best] += alpha * (x - protos[best])
            else:
                protos[best] -= alpha * (x - protos[best])
    return protos, proto_labels

def evaluate(X, y, protos, proto_labels):
    diff = protos - X[:, None, :]          # (n, n_protos, d)  -- memory heavy for large n, but n≤2867, n_protos≤70
    dist2 = np.sum(diff * diff, axis=2)    # (n, n_protos)
    best = np.argmin(dist2, axis=1)
    pred = proto_labels[best]
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Split: last 20% for tuning (file order)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # Convert to float32 for training
    X_train_f = X_train.astype(np.float32)
    X_tune_f = X_tune.astype(np.float32)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_protos = None
    best_labels = None

    for P in GRID["prototypes_per_class"]:
        for epochs in GRID["epochs"]:
            for alpha0 in GRID["alpha0"]:
                # Use a fresh RNG for each configuration to keep determinism across grid
                cfg_rng = np.random.default_rng(SEED + P*1000 + epochs*100 + int(alpha0*1000))
                protos, labels = lvq_train(X_train_f, y_train, C, P, epochs, alpha0, cfg_rng)
                acc = evaluate(X_tune_f, y_tune, protos, labels)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (P, epochs, alpha0)
                    best_protos = protos
                    best_labels = labels

    # Retrain on full training set with best hyperparameters
    P, epochs, alpha0 = best_params
    X_full_f = X.astype(np.float32)
    final_rng = np.random.default_rng(SEED + 9999)   # separate seed for final training
    protos, labels = lvq_train(X_full_f, y, C, P, epochs, alpha0, final_rng)
    n_protos = protos.shape[0]

    # Write weights.bin
    # Format: magic (0x4C565131), n_protos (int32), d (int32), C (int32),
    # then for each prototype: class (uint8), vector (d * float32 little‑endian)
    magic = 0x4C565131  # "LVQ1"
    out = struct.pack("<4i", magic, n_protos, d, C)
    for i in range(n_protos):
        out += struct.pack("<B", int(best_labels[i]))
        out += protos[i].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* LVQ inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (pixel counts). Output is one byte: class of nearest prototype. */
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
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x4C565131) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_protos = (int32_t)(wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24);
    int32_t d = (int32_t)(wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24);
    int32_t C = (int32_t)(wb[12] | wb[13] << 8 | wb[14] << 16 | wb[15] << 24);
    if (n_protos <= 0 || d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + (long)n_protos * (1 + (long)d * 4);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *proto_class = malloc(n_protos);
    float *proto_vec = malloc((long)n_protos * d * sizeof(float));
    unsigned char *p = wb + 16;
    for (int i = 0; i < n_protos; i++) {
        proto_class[i] = *p++;
        memcpy(&proto_vec[(long)i * d], p, d * sizeof(float));
        p += d * 4;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *diff = malloc(d * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f;
        unsigned char best_class = 0;
        for (int j = 0; j < n_protos; j++) {
            const float *pv = &proto_vec[(long)j * d];
            float dist2 = 0.0f;
            for (int f = 0; f < d; f++) {
                float df = (float)xi[f] - pv[f];
                dist2 += df * df;
            }
            if (dist2 < best_dist) {
                best_dist = dist2;
                best_class = proto_class[j];
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```