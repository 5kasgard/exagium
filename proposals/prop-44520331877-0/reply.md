```hypothesis
Family: label-lvq (Learning Vector Quantization). 
Why: Optdigits pixels are ordinal 0–16; LVQ learns a small set of prototypes per class (3–5) capturing non-linear decision boundaries that a linear model misses, while inference remains a simple nearest-prototype search (≈30×64 ops). 
Failure signal: If dev accuracy ≤ 0.961 (label-linear) or Ir/item > 15000 (2× linear), the family does not justify its complexity.
```
```python
"""label-lvq: Learning Vector Quantization for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Tunes K (prototypes/class) on last 20% of train, then retrains on all data.
Weights: magic 0x4C565131, int32 d, C, K; float32 prototypes[C*K, d] row-major."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR_INIT, LR_FINAL, EPOCHS_TUNE, EPOCHS_FULL = 0.1, 0.01, 10, 25
K_CHOICES = [2, 3, 4, 5]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def lvq_train(X, y, C, d, K, epochs, lr_init, lr_final, rng, verbose=False):
    """X: (n, d) float32, y: (n,) int64. Returns prototypes (C*K, d) float32."""
    n = X.shape[0]
    # Initialize prototypes by sampling from each class
    protos = np.zeros((C * K, d), dtype=np.float32)
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(K, len(idx)), replace=False)
        protos[c * K:(c + 1) * K] = X[chosen]
    # LVQ1
    for ep in range(epochs):
        lr = lr_init * (lr_final / lr_init) ** (ep / max(1, epochs - 1))
        order = rng.permutation(n)
        for i in order:
            x = X[i]
            yi = y[i]
            # squared distances to all prototypes
            diff = protos - x  # (C*K, d)
            dist2 = np.sum(diff * diff, axis=1)
            best = np.argmin(dist2)
            best_class = best // K
            if best_class == yi:
                protos[best] += lr * (x - protos[best])
            else:
                protos[best] -= lr * (x - protos[best])
    return protos

def evaluate(X, y, protos, C, K):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        diff = protos - X[i]
        dist2 = np.sum(diff * diff, axis=1)
        best = np.argmin(dist2)
        if best // K == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]
    # Convert to float32 for training
    X = X_bytes.astype(np.float32)
    # Split: last 20% for tuning
    n_val = n // 5
    tr_idx = np.arange(n - n_val)
    val_idx = np.arange(n - n_val, n)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    rng = np.random.default_rng(SEED)
    best_K, best_acc = K_CHOICES[0], -1.0
    for K in K_CHOICES:
        rng_k = np.random.default_rng(SEED + K * 1000)  # deterministic per K
        protos = lvq_train(X_tr, y_tr, C, d, K, EPOCHS_TUNE, LR_INIT, LR_FINAL, rng_k)
        acc = evaluate(X_val, y_val, protos, C, K)
        if acc > best_acc:
            best_acc, best_K = acc, K
    # Retrain on full data with best_K
    rng_full = np.random.default_rng(SEED + best_K * 1000 + 12345)
    protos = lvq_train(X, y, C, d, best_K, EPOCHS_FULL, LR_INIT, LR_FINAL, rng_full)
    # Write weights
    magic = 0x4C565131  # "LVQ1"
    out = struct.pack("<4i", magic, d, C, best_K)
    out += protos.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-lvq inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output record: 1 byte class index.
 * Weights: magic 0x4C565131, int32 d, C, K; float32 prototypes[C*K][d]. */
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
    int d = hd[1], C = hd[2], K = hd[3];
    int PK = C * K;
    if (C < 1 || C > 256 || K < 1 || d < 1 || nw != 16 + (long)PK * d * 4) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *protos = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *diff = (float *)malloc(d * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_dist = 1e30f; int best_proto = 0;
        for (int p = 0; p < PK; p++) {
            const float *pr = protos + (long)p * d;
            float dist2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float delta = (float)xi[j] - pr[j];
                dist2 += delta * delta;
            }
            if (dist2 < best_dist) { best_dist = dist2; best_proto = p; }
        }
        unsigned char cls = (unsigned char)(best_proto / K);
        put_rec(stdout, &cls, 1);
    }
    free(diff);
    return 0;
}
```