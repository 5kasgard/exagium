```hypothesis
Family: label-mlp-tuned – a one-hidden-layer MLP with hyperparameters (hidden size, learning rate, L2) chosen on the tuning split (last 20% of train.in) and then trained on all data. The linear model (H=0) is included in the search space, so this family can only match or improve label-linear. The ReLU hidden layer adds non‑linear capacity for character‑level patterns; tuning on the held‑out 20% prevents overfitting. If the best configuration on the tuning split is H=0, or if the final model does not exceed label-linear’s dev accuracy (0.3375), the hypothesis fails.
```
```python
"""label-mlp-tuned: MLP with one ReLU layer, hyperparameters tuned on the last 20% of train.in.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seeds), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed training hyperparameters (shared across configs)
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid to search on the tuning split
GRID = [
    {"H": 0,   "LR": 3e-3, "L2": 1e-4},
    {"H": 64,  "LR": 3e-3, "L2": 1e-4},
    {"H": 64,  "LR": 1e-3, "L2": 1e-4},
    {"H": 64,  "LR": 3e-3, "L2": 1e-5},
    {"H": 128, "LR": 3e-3, "L2": 1e-4},
    {"H": 128, "LR": 1e-3, "L2": 1e-4},
    {"H": 128, "LR": 3e-3, "L2": 1e-5},
    {"H": 256, "LR": 3e-3, "L2": 1e-4},
    {"H": 256, "LR": 1e-3, "L2": 1e-4},
    {"H": 256, "LR": 3e-3, "L2": 1e-5},
]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoder(X, types):
    """Return list of encoding matrices E_j (256 x n_vals) for each position."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """Apply encoders to X (n x d) -> Z (n x k)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits (n x C)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def train_model(X, y, enc, C, H, LR, L2, seed):
    """Train on given data, return parameters (W1, b1, W2, b2)."""
    n, k = X.shape
    rng = np.random.default_rng(seed)

    W = H if H else C
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
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = X[idx], y[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
                g = softmax_grad(logits, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + L2 * W1,
                    gh.sum(0),
                    h.T @ g + L2 * W2,
                    g.sum(0)
                ]
            else:
                logits = z @ W1 + b1
                g = softmax_grad(logits, yy)
                grads = [
                    z.T @ g + L2 * W1,
                    g.sum(0)
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    if H:
        return params[0], params[1], params[2], params[3]
    else:
        return params[0], params[1], None, None


def predict(X, enc, W1, b1, W2, b2, H, C):
    """Return predicted class indices for X (n x d)."""
    Z = encode(X, enc)
    if H:
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        logits = Z @ W1 + b1
    return logits.argmax(axis=1).astype(np.int64)


def write_weights(card, enc, W1, b1, W2, b2, H, C, path):
    """Write weights in the same format as label-linear (magic 0x54424C31)."""
    d = card["input"]["bytes"]
    W = H if H else C
    # Build per-position lookup tables T[j][v] = E_j[v] @ W1[offset:offset+E_j.shape[1]]
    T_list = []
    off = 0
    for E in enc:
        T_list.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # (d, 256, W)
    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    with open(path, "wb") as f:
        f.write(out)


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]

    n = len(X_all)
    val_size = n // 5  # last 20%
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    # Build encoder on training split only (to avoid data leakage)
    enc_train = build_encoder(X_train, types)
    Z_train = encode(X_train, enc_train)
    Z_val = encode(X_val, enc_train)

    best_acc = -1.0
    best_cfg = None
    best_params = None

    for cfg_idx, cfg in enumerate(GRID):
        H, LR, L2 = cfg["H"], cfg["LR"], cfg["L2"]
        seed = SEED + cfg_idx + 1
        W1, b1, W2, b2 = train_model(Z_train, y_train, enc_train, C, H, LR, L2, seed)
        preds = predict(X_val, enc_train, W1, b1, W2, b2, H, C)
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = (W1, b1, W2, b2)
        print(f"Config {cfg}: val_acc={acc:.4f}", file=sys.stderr)

    print(f"Best config: {best_cfg} with val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best config
    enc_all = build_encoder(X_all, types)
    Z_all = encode(X_all, enc_all)
    H, LR, L2 = best_cfg["H"], best_cfg["LR"], best_cfg["L2"]
    W1, b1, W2, b2 = train_model(Z_all, y_all, enc_all, C, H, LR, L2, SEED)

    write_weights(card, enc_all, W1, b1, W2, b2, H, C, sys.argv[4])


if __name__ == "__main__":
    main()
```
```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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