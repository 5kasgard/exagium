```hypothesis
Propose "label-tiny-mlp": a single-hidden-layer MLP with 8 ReLU units, tuned on the last 20% of train.in.
The adult tabular data has both ordinal and categorical features; a tiny nonlinear layer can capture
interactions (e.g., education × occupation) that a linear model misses, while the small hidden size
keeps inference cost close to label-linear. Hyperparameters (hidden size 8/16, learning rate 1e-3/3e-3)
are selected on the tuning split, then the model is retrained on all training data. If dev accuracy
does not exceed label-linear (0.8524) or inference cost grows beyond ~1000 Ir/item, the hypothesis fails.
```

```python
"""label-tiny-mlp: single hidden layer MLP with tuned hidden size (8 or 16).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (except those we tune)
SEED = 1
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Tuning grid
HIDDEN_SIZES = [8, 16]
LEARNING_RATES = [1e-3, 3e-3]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def encode_features(X, types):
    """Return list of encoding matrices E_j (256 x k_j) and total input dim K."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def train_mlp(Z, y, H, C, lr, rng, epochs, batch, l2):
    """Train MLP on given data, return parameters (W1, b1, W2, b2)."""
    n, k = Z.shape
    # He initialization for ReLU
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            # Forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # Softmax + cross-entropy gradient
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            # Backward
            gh = g @ W2.T * (a > 0)
            grads = [
                z.T @ gh + l2 * W1,
                gh.sum(0),
                h.T @ g + l2 * W2,
                g.sum(0)
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def evaluate(Z, y, params):
    W1, b1, W2, b2 = params
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    n_all = X_all.shape[0]

    # Split: last 20% for tuning (file order)
    split = int(0.8 * n_all)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    # Encode using training data statistics (as in label-linear)
    enc = encode_features(X_train, types)
    K = sum(E.shape[1] for E in enc)
    # Build full encoded matrices
    def build_Z(X):
        parts = [E[X[:, j]] for j, E in enumerate(enc)]
        return np.concatenate(parts, axis=1).astype(np.float32)
    Z_train = build_Z(X_train)
    Z_tune = build_Z(X_tune)

    # Grid search
    best_acc = -1.0
    best_params = None
    best_H = None
    best_lr = None
    base_rng = np.random.default_rng(SEED)
    for H in HIDDEN_SIZES:
        for lr in LEARNING_RATES:
            # Derive a seed for this trial
            trial_seed = int(base_rng.integers(2**31))
            rng = np.random.default_rng(trial_seed)
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(Z_train)))
            params = train_mlp(Z_train, y_train, H, C, lr, rng, epochs, BATCH, L2)
            acc = evaluate(Z_tune, y_tune, params)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_H = H
                best_lr = lr

    # Retrain on full data with best hyperparameters
    enc_full = encode_features(X_all, types)
    def build_Z_full(X):
        parts = [E[X[:, j]] for j, E in enumerate(enc_full)]
        return np.concatenate(parts, axis=1).astype(np.float32)
    Z_all = build_Z_full(X_all)
    final_rng = np.random.default_rng(SEED)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / len(Z_all)))
    final_params = train_mlp(Z_all, y_all, best_H, C, best_lr, final_rng, epochs_full, BATCH, L2)
    W1, b1, W2, b2 = final_params

    # Build per-feature lookup tables T[j][256][H]
    T_list = []
    off = 0
    for E in enc_full:
        T_j = (E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32)
        T_list.append(T_j)
        off += E.shape[1]
    T_stacked = np.stack(T_list)  # shape (d, 256, H)

    # Write weights.bin
    # Magic: 0x4D4C5038 ('MLP8' in little-endian)
    magic = 0x4D4C5038
    H = best_H
    W = H
    header = struct.pack("<4i", magic, d, H, C)
    # T: d * 256 * W floats
    T_bytes = T_stacked.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    out = header + T_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-tiny-mlp infer.c: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1).
 * Weights format: magic 0x4D4C5038, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x4D4C5038) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long W = H;
    long expected = 16 + 4L * ((long)d * 256 * W + W + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        // ReLU
        for (int u = 0; u < H; u++) if (acc[u] < 0.0f) acc[u] = 0.0f;
        // Output layer
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float h = acc[u];
            for (int c = 0; c < C; c++) scores[c] += h * W2[(long)u * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```