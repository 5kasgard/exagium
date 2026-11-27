```hypothesis
Family: label-lvq (Learning Vector Quantization).  
Why: Optdigits forms tight per-digit clusters; LVQ learns discriminative prototypes with minimal compute.  
Failure: Dev accuracy ≤ 0.9613 (label-linear) or inference cost > 2× label-linear.
```

```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
train.py card.json train.in train.key weights.bin
Deterministic (seed 12345), single thread, stdlib + numpy.
Hyperparameters chosen on last 20% of train.in (file order), then retrained on all train.
Weights format: magic 0x4C565131, d, P, C; float32 prototypes[P][d]; uint8 proto_class[P]."""
import json, math, struct, sys
import numpy as np

SEED = 12345
# Hyperparameter grid for tuning
PROTO_PER_CLASS_OPTS = [1, 2, 3, 4]
LR_OPTS = [0.05, 0.1, 0.2]
EPOCHS_OPTS = [50, 100]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def lvq_train(X, y, d, C, proto_per_class, lr, epochs, rng):
    """Train LVQ, return prototypes (P, d) float32 and proto_class (P,) uint8."""
    P = proto_per_class * C
    # Initialize prototypes: for each class, pick first proto_per_class samples after shuffling
    # We'll shuffle indices per class with rng.
    prototypes = np.zeros((P, d), dtype=np.float32)
    proto_class = np.zeros(P, dtype=np.uint8)
    idx = 0
    for c in range(C):
        class_mask = (y == c)
        class_indices = np.where(class_mask)[0]
        if len(class_indices) == 0:
            # Should not happen, but fallback: random init
            for _ in range(proto_per_class):
                prototypes[idx] = rng.uniform(0, 16, d).astype(np.float32)
                proto_class[idx] = c
                idx += 1
            continue
        rng.shuffle(class_indices)
        chosen = class_indices[:proto_per_class]
        for ci in chosen:
            prototypes[idx] = X[ci].astype(np.float32)
            proto_class[idx] = c
            idx += 1
    # Precompute squared norms of prototypes for distance calculation
    # Not needed during training, but we'll update prototypes directly.
    n = X.shape[0]
    Xf = X.astype(np.float32)
    for ep in range(epochs):
        lr_ep = lr * (1.0 - ep / epochs)
        order = rng.permutation(n)
        for i in order:
            x = Xf[i]
            yi = y[i]
            # Find closest prototype (Euclidean distance squared)
            diff = prototypes - x  # (P, d)
            dist2 = np.sum(diff * diff, axis=1)
            best = np.argmin(dist2)
            if proto_class[best] == yi:
                prototypes[best] += lr_ep * (x - prototypes[best])
            else:
                prototypes[best] -= lr_ep * (x - prototypes[best])
    return prototypes, proto_class

def evaluate(X, y, prototypes, proto_class):
    """Return accuracy on given set."""
    Xf = X.astype(np.float32)
    diff = prototypes[None, :, :] - Xf[:, None, :]  # (n, P, d)
    dist2 = np.sum(diff * diff, axis=2)  # (n, P)
    best = np.argmin(dist2, axis=1)
    pred = proto_class[best]
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n_total = X.shape[0]
    n_tune = int(0.2 * n_total)
    train_idx = np.arange(n_total - n_tune)
    tune_idx = np.arange(n_total - n_tune, n_total)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_prototypes = None
    best_proto_class = None

    for ppc in PROTO_PER_CLASS_OPTS:
        for lr in LR_OPTS:
            for ep in EPOCHS_OPTS:
                # Train on train_idx
                proto, pclass = lvq_train(X_train, y_train, d, C, ppc, lr, ep, np.random.default_rng(SEED))
                acc = evaluate(X_tune, y_tune, proto, pclass)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (ppc, lr, ep)
                    best_prototypes = proto
                    best_proto_class = pclass

    # Retrain on full dataset with best hyperparameters
    ppc, lr, ep = best_params
    rng_full = np.random.default_rng(SEED)
    prototypes, proto_class = lvq_train(X, y, d, C, ppc, lr, ep, rng_full)

    # Write weights.bin
    P = prototypes.shape[0]
    magic = 0x4C565131  # 'LVQ1' little-endian
    header = struct.pack("<4i", magic, d, P, C)
    proto_bytes = prototypes.astype("<f4").tobytes()
    class_bytes = proto_class.astype(np.uint8).tobytes()
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(proto_bytes)
        f.write(class_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class per record (4-byte LE length + byte).
 * Weights: magic 0x4C565131, d, P, C; float32 prototypes[P][d]; uint8 proto_class[P]. */
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
    if (nw < 16 || hd[0] != 0x4C565131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    if (d <= 0 || P <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * d * P + P;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }
    const float *prototypes = (const float *)(hd + 4);
    const unsigned char *proto_class = (const unsigned char *)(prototypes + (long)d * P);

    // Precompute squared norms of prototypes
    float *proto_norm2 = malloc(P * sizeof(float));
    for (int i = 0; i < P; i++) {
        float sum = 0.0f;
        const float *p = prototypes + (long)i * d;
        for (int j = 0; j < d; j++) sum += p[j] * p[j];
        proto_norm2[i] = sum;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_f = malloc(d * sizeof(float));
    float *dots = malloc(P * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Convert to float
        for (int j = 0; j < d; j++) x_f[j] = (float)xi[j];
        // Compute dot products with prototypes
        for (int p = 0; p < P; p++) {
            float dot = 0.0f;
            const float *pr = prototypes + (long)p * d;
            for (int j = 0; j < d; j++) dot += x_f[j] * pr[j];
            dots[p] = dot;
        }
        // Find prototype minimizing squared distance = |x|^2 + |p|^2 - 2 dot
        // |x|^2 is constant, so minimize |p|^2 - 2 dot
        float best_val = 1e30f;
        int best_p = 0;
        for (int p = 0; p < P; p++) {
            float val = proto_norm2[p] - 2.0f * dots[p];
            if (val < best_val) {
                best_val = val;
                best_p = p;
            }
        }
        unsigned char out = proto_class[best_p];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```