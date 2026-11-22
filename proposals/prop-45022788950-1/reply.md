```hypothesis
Family: label-mlp2h (two-hidden-layer MLP with per-byte lookup first layer). 
Why: Optdigits has 64 ordinal pixel features; a single hidden layer (label-linear H>0, label-mlp16) already reaches ~97.7%. Adding a second hidden layer allows richer feature combinations while the lookup-table first layer keeps inference fast. We expect dev accuracy >97% at ~7000 Ir/item (between label-linear and label-mlp16). 
Failure: If dev accuracy ≤96.5% or inference cost exceeds label-mlp16 by >30%, the hypothesis is invalid.
```

```python
"""label-mlp2h: two-hidden-layer MLP with per-byte lookup first layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys
import numpy as np

# Fixed hyperparameter grid for tuning (last 20% of train.in)
GRID = [
    {"H1": 64, "H2": 32, "lr": 1e-3, "epochs": 80},
    {"H1": 64, "H2": 16, "lr": 1e-3, "epochs": 80},
    {"H1": 32, "H2": 16, "lr": 3e-3, "epochs": 60},
    {"H1": 32, "H2": 32, "lr": 1e-3, "epochs": 80},
]
SEED = 1
L2 = 1e-4
BATCH = 64

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def relu(x):
    return np.maximum(x, 0, dtype=np.float32)

def softmax_cross_entropy_grad(logits, y):
    # logits: (N, C), y: (N,)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs.astype(np.float32) / len(y)

def train_model(X, y, X_val, y_val, H1, H2, lr, epochs, seed, d, C):
    rng = np.random.default_rng(seed)
    # He initialization
    W1 = rng.normal(0, math.sqrt(2.0 / d), size=(d, H1)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H1), size=(H1, H2)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = rng.normal(0, math.sqrt(2.0 / H2), size=(H2, C)).astype(np.float32)
    b3 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z = X[idx]          # (B, d)
            yy = y[idx]         # (B,)

            # Forward
            a1 = z @ W1 + b1
            h1 = relu(a1)
            a2 = h1 @ W2 + b2
            h2 = relu(a2)
            logits = h2 @ W3 + b3

            # Loss gradient
            g3 = softmax_cross_entropy_grad(logits, yy)   # (B, C)

            # Backward
            gW3 = h2.T @ g3 + L2 * W3
            gb3 = g3.sum(axis=0)
            gh2 = g3 @ W3.T
            gh2[a2 <= 0] = 0
            gW2 = h1.T @ gh2 + L2 * W2
            gb2 = gh2.sum(axis=0)
            gh1 = gh2 @ W2.T
            gh1[a1 <= 0] = 0
            gW1 = z.T @ gh1 + L2 * W1
            gb1 = gh1.sum(axis=0)

            grads = [gW1, gb1, gW2, gb2, gW3, gb3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * gr * gr
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Validation accuracy
    a1 = X_val @ W1 + b1
    h1 = relu(a1)
    a2 = h1 @ W2 + b2
    h2 = relu(a2)
    logits = h2 @ W3 + b3
    pred = logits.argmax(axis=1)
    acc = (pred == y_val).mean()
    return acc, (W1, b1, W2, b2, W3, b3)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    # No shuffling for tuning split (file order)
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Normalization on training split
    mu = X_train.mean(axis=0).astype(np.float32)
    sd = X_train.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X_train_n = (X_train - mu) / sd
    X_val_n = (X_val - mu) / sd

    # Grid search
    best_acc = -1
    best_params = None
    best_cfg = None
    for cfg in GRID:
        acc, params = train_model(X_train_n, y_train, X_val_n, y_val,
                                  cfg["H1"], cfg["H2"], cfg["lr"], cfg["epochs"],
                                  SEED, d, C)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_cfg = cfg
    print(f"Tuning best: {best_cfg}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    mu_full = X.mean(axis=0).astype(np.float32)
    sd_full = X.std(axis=0).astype(np.float32)
    sd_full[sd_full == 0] = 1.0
    X_full_n = (X - mu_full) / sd_full
    _, final_params = train_model(X_full_n, y, X_full_n, y,  # dummy val
                                  best_cfg["H1"], best_cfg["H2"], best_cfg["lr"], best_cfg["epochs"],
                                  SEED, d, C)
    W1, b1, W2, b2, W3, b3 = final_params
    H1, H2 = best_cfg["H1"], best_cfg["H2"]

    # Build per-byte lookup table for first layer: T1[j, v] = ((v - mu_j)/sd_j) * W1[j]
    # v in 0..255, j in 0..d-1, output H1
    T1 = np.empty((d, 256, H1), dtype=np.float32)
    for j in range(d):
        scale = (np.arange(256, dtype=np.float32) - mu_full[j]) / sd_full[j]
        T1[j] = scale[:, None] * W1[j]   # (256, H1)

    # Pack weights.bin
    # Magic: 0x4D4C5032 ('M','L','P','2' in little-endian)
    magic = 0x4D4C5032
    header = struct.pack("<5i", magic, d, H1, H2, C)
    # Order: T1, b1, W2, b2, W3, b3
    out = header
    out += T1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-mlp2h inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class (0-9).
 * Weights format: magic(0x4D4C5032), d, H1, H2, C (int32 each),
 * then float32 arrays: T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]. */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || H1 < 1 || H2 < 1) { fprintf(stderr, "bad dims\n"); return 3; }

    // Expected file size
    long expected = 20 +
        (long)d * 256 * H1 * 4 +
        H1 * 4 +
        (long)H1 * H2 * 4 +
        H2 * 4 +
        (long)H2 * C * 4 +
        C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *h1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *h2 = malloc(sizeof(float) * H2);
    float *acc3 = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // First layer: acc1 = b1 + sum_j T1[j][xi[j]]
        memcpy(acc1, b1, H1 * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) acc1[u] += t[u];
        }
        // ReLU
        for (int u = 0; u < H1; u++) h1[u] = acc1[u] > 0.0f ? acc1[u] : 0.0f;

        // Second layer: acc2 = b2 + h1 @ W2
        memcpy(acc2, b2, H2 * sizeof(float));
        for (int u = 0; u < H1; u++) {
            float hu = h1[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) acc2[v] += hu * w[v];
        }
        // ReLU
        for (int v = 0; v < H2; v++) h2[v] = acc2[v] > 0.0f ? acc2[v] : 0.0f;

        // Third layer: acc3 = b3 + h2 @ W3
        memcpy(acc3, b3, C * sizeof(float));
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
            if (hv == 0.0f) continue;
            const float *w = W3 + (long)v * C;
            for (int c = 0; c < C; c++) acc3[c] += hv * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc3[c] > acc3[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```