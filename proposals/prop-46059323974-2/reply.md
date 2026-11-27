```hypothesis
Family: label-embed-mlp (per-byte embeddings + one ReLU layer, hyperparameters tuned on validation split).
Why: The linear model (label-linear) uses one-hot features directly; adding a learned embedding layer before the hidden layer allows the model to share statistical strength across byte values and positions, and a larger hidden layer (64–128) can capture non-linear interactions that a single linear layer misses. The label-mlp16 (H=16) underperformed, likely due to insufficient capacity and untuned hyperparameters. Tuning H, learning rate, and L2 on the held-out 20% should find a better trade-off.
Failure signal: If dev accuracy does not exceed 0.3375 (label-linear) or if inference cost per item exceeds label-linear by >20%, the added complexity is not justified.
```

```python
"""label-embed-mlp: per-byte embeddings + one ReLU layer, hyperparameters chosen on validation split.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys, hashlib

import numpy as np

# Fixed training budget (same as label-linear)
SEED = 1
SAMPLES = 600000
MAX_EPOCHS = 100
BATCH = 64

# Hyperparameter grid to search on the validation split (last 20% of train.in)
GRID = [
    {"H": 32, "lr": 1e-3, "l2": 1e-4},
    {"H": 32, "lr": 3e-3, "l2": 1e-4},
    {"H": 32, "lr": 1e-3, "l2": 1e-5},
    {"H": 64, "lr": 1e-3, "l2": 1e-4},
    {"H": 64, "lr": 3e-3, "l2": 1e-4},
    {"H": 64, "lr": 1e-3, "l2": 1e-5},
    {"H": 128, "lr": 1e-3, "l2": 1e-4},
    {"H": 128, "lr": 3e-3, "l2": 1e-4},
    {"H": 128, "lr": 1e-3, "l2": 1e-5},
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def prepare_data(card, X, y):
    """Build one-hot encoding matrices and concatenated design matrix Z."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    return enc, Z

def train_model(enc, Z, y, H, lr, l2, batch, max_epochs, samples, seed):
    """Train on given data, return (T, b1, W2, b2) and validation accuracy if y_val provided."""
    n, k = Z.shape
    C = 65  # card["classes"]
    rng = np.random.default_rng(seed)

    # Initialize parameters
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    epochs = min(max_epochs, math.ceil(samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + l2 * W1,
                    gh.sum(0),
                    h.T @ g + l2 * W2,
                    g.sum(0)
                ]
            else:
                g = softmax_grad(z @ W1 + b1, yy)
                grads = [z.T @ g + l2 * W1, g.sum(0)]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Build per-byte lookup tables T
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T)  # (d, 256, W) where W = H if H else C
    b1 = params[1]
    if H:
        W2, b2 = params[2], params[3]
    else:
        W2, b2 = None, None
    return T, b1, W2, b2

def evaluate(T, b1, W2, b2, X, y):
    """Compute accuracy on given data using the lookup tables."""
    n, d = X.shape
    H = W2.shape[0] if W2 is not None else 0
    C = 65
    W = H if H else C
    correct = 0
    for i in range(n):
        acc = b1.copy()
        for j in range(d):
            acc += T[j, X[i, j]]
        if H:
            h = np.maximum(acc, 0)
            scores = h @ W2 + b2
        else:
            scores = acc
        pred = np.argmax(scores)
        if pred == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    # Split: last 20% for validation (tuning split)
    n = len(X)
    val_start = int(0.8 * n)
    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:], y[val_start:]

    enc_train, Z_train = prepare_data(card, X_train, y_train)
    enc_val, Z_val = prepare_data(card, X_val, y_val)

    best_acc = -1.0
    best_cfg = None
    best_params = None

    # Grid search on validation split with reduced epochs for speed
    VAL_EPOCHS = 10
    VAL_SAMPLES = min(SAMPLES, VAL_EPOCHS * len(X_train))
    for cfg in GRID:
        # Deterministic seed per configuration
        cfg_seed = int(hashlib.md5(str(cfg).encode()).hexdigest()[:8], 16) % (2**32)
        T, b1, W2, b2 = train_model(
            enc_train, Z_train, y_train,
            cfg["H"], cfg["lr"], cfg["l2"], BATCH, VAL_EPOCHS, VAL_SAMPLES, cfg_seed
        )
        acc = evaluate(T, b1, W2, b2, X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = (T, b1, W2, b2)

    print(f"Best validation accuracy: {best_acc:.4f} with {best_cfg}", file=sys.stderr)

    # Retrain on full training data with best hyperparameters
    enc_full, Z_full = prepare_data(card, X, y)
    T, b1, W2, b2 = train_model(
        enc_full, Z_full, y,
        best_cfg["H"], best_cfg["lr"], best_cfg["l2"], BATCH, MAX_EPOCHS, SAMPLES, SEED
    )

    # Pack weights exactly like label-linear format
    H = best_cfg["H"]
    W = H if H else C
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```