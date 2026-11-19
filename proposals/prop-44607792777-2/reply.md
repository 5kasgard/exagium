```hypothesis
We propose "label-embed-mlp": a shallow neural network with per-position byte embeddings summed into a context vector, followed by a ReLU hidden layer and a linear output. This captures positional interactions and nonlinearities missing in label-linear, which is crucial for character-level sequence prediction. If dev accuracy does not exceed 0.3375 (label-linear) or inference cost per item is significantly higher, the hypothesis fails.
```

```python
"""label-embed-mlp: per-position embeddings + one ReLU hidden layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, numpy + stdlib only.
Tunes on last 20% of train.in, then retrains on all data.
Weights format: magic 0x4D4F444C, int32 d,D,H,C; float32 embeddings[d][256][D], W1[D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid (ordered for deterministic tie-breaking)
GRID = [
    {"D": 16, "H": 32, "lr": 1e-3, "epochs": 20},
    {"D": 16, "H": 32, "lr": 1e-3, "epochs": 30},
    {"D": 16, "H": 32, "lr": 3e-3, "epochs": 20},
    {"D": 16, "H": 32, "lr": 3e-3, "epochs": 30},
    {"D": 16, "H": 64, "lr": 1e-3, "epochs": 20},
    {"D": 16, "H": 64, "lr": 1e-3, "epochs": 30},
    {"D": 16, "H": 64, "lr": 3e-3, "epochs": 20},
    {"D": 16, "H": 64, "lr": 3e-3, "epochs": 30},
    {"D": 32, "H": 32, "lr": 1e-3, "epochs": 20},
    {"D": 32, "H": 32, "lr": 1e-3, "epochs": 30},
    {"D": 32, "H": 32, "lr": 3e-3, "epochs": 20},
    {"D": 32, "H": 32, "lr": 3e-3, "epochs": 30},
    {"D": 32, "H": 64, "lr": 1e-3, "epochs": 20},
    {"D": 32, "H": 64, "lr": 1e-3, "epochs": 30},
    {"D": 32, "H": 64, "lr": 3e-3, "epochs": 20},
    {"D": 32, "H": 64, "lr": 3e-3, "epochs": 30},
]
BATCH = 64
L2 = 1e-4
VAL_FRAC = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)


def forward(X, params):
    """X: (B, d) uint8. Returns logits (B, C), and cache for backward."""
    E, W1, b1, W2, b2 = params
    d, D, H, C = E.shape[0], E.shape[2], W1.shape[1], W2.shape[1]
    B = X.shape[0]
    # Sum embeddings
    h = np.zeros((B, D), np.float32)
    for p in range(d):
        h += E[p, X[:, p]]
    # Hidden layer
    z = h @ W1 + b1
    a = np.maximum(z, 0)
    # Output
    logits = a @ W2 + b2
    cache = (h, z, a)
    return logits, cache


def backward(X, y, params, cache, logits):
    E, W1, b1, W2, b2 = params
    h, z, a = cache
    B = X.shape[0]
    d, D, H, C = E.shape[0], E.shape[2], W1.shape[1], W2.shape[1]
    # dlogits
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    dlogits = probs / B
    # Output layer grads
    dW2 = a.T @ dlogits + L2 * W2
    db2 = dlogits.sum(axis=0)
    # Hidden layer grads
    da = dlogits @ W2.T
    dz = da * (z > 0)
    dW1 = h.T @ dz + L2 * W1
    db1 = dz.sum(axis=0)
    # Embedding grads
    dh = dz @ W1.T  # (B, D)
    dE = np.zeros_like(E)
    for p in range(d):
        np.add.at(dE[p], X[:, p], dh)
    return [dE, dW1, db1, dW2, db2]


def train_loop(X, y, params, cfg, rng):
    n = X.shape[0]
    steps_per_epoch = max(1, n // BATCH)
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, cache = forward(xb, params)
            grads = backward(xb, yb, params, cache, logits)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= cfg["lr"] * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def init_params(d, D, H, C, rng):
    # Embeddings: uniform [-1, 1] * sqrt(3/D) approx
    E = rng.uniform(-1, 1, (d, 256, D)).astype(np.float32) * math.sqrt(3.0 / D)
    W1 = rng.uniform(-1, 1, (D, H)).astype(np.float32) * math.sqrt(6.0 / D)
    b1 = np.zeros(H, np.float32)
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
    b2 = np.zeros(C, np.float32)
    return [E, W1, b1, W2, b2]


def evaluate(X, y, params):
    logits, _ = forward(X, params)
    return accuracy(logits, y)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(n * (1 - VAL_FRAC))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    base_rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    for idx, cfg in enumerate(GRID):
        # Derive a seed for this config
        cfg_seed = int(base_rng.integers(2**31))
        rng = np.random.default_rng(cfg_seed)
        params = init_params(d, cfg["D"], cfg["H"], C, rng)
        params = train_loop(X_train, y_train, params, cfg, rng)
        acc = evaluate(X_val, y_val, params)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = params

    # Retrain on full data with best config
    print(f"Best config: {best_cfg}, val acc: {best_acc:.4f}", file=sys.stderr)
    rng = np.random.default_rng(int(base_rng.integers(2**31)))
    params = init_params(d, best_cfg["D"], best_cfg["H"], C, rng)
    params = train_loop(X, y, params, best_cfg, rng)

    E, W1, b1, W2, b2 = params
    D, H = best_cfg["D"], best_cfg["H"]
    # Pack weights
    magic = 0x4D4F444C  # 'MODL'
    header = struct.pack("<5i", magic, d, D, H, C)
    # Embeddings: d * 256 * D floats
    emb_bytes = E.astype("<f4").tobytes()
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    out = header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x4D4F444C, int32 d,D,H,C; float32 embeddings[d][256][D], W1[D][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x4D4F444C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + (long)d * 256 * D;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * D);
    float *z = (float *)malloc(sizeof(float) * H);
    float *a = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Sum embeddings
        for (int j = 0; j < D; j++) h[j] = 0.0f;
        for (int p = 0; p < d; p++) {
            const float *ep = E + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) h[j] += ep[j];
        }
        // Hidden layer: z = h @ W1 + b1
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int j = 0; j < D; j++) sum += h[j] * W1[(long)j * H + u];
            z[u] = sum;
            a[u] = sum > 0.0f ? sum : 0.0f;  // ReLU
        }
        // Output layer: logits = a @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += a[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```