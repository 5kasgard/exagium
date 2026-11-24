```hypothesis
We propose a two-hidden-layer MLP (label-mlp2h) with 32 and 16 ReLU units. The optdigits dataset has 64 ordinal pixel features; a deeper network can capture hierarchical features (e.g., strokes then digits) better than a single layer. We expect dev accuracy >0.97, surpassing label-linear (0.961) and label-mlp16 (0.977). If dev accuracy is ≤0.965, the model does not justify its extra cost.
```

```python
"""Two-hidden-layer MLP for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Architecture: input (d) -> ReLU(H1) -> ReLU(H2) -> linear(C).
First layer folded into per-byte lookup tables for fast inference.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all data."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (selected via tuning split experimentation)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CONFIGS = [(16, 8), (32, 16), (64, 32)]  # (H1, H2) candidates

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def encode_features(X, types):
    """Return list of encoding matrices (256 x dim_j) and total encoded dim."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def build_design(X, enc):
    """Apply encodings to X, return float32 matrix (n, total_dim)."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)

def train_model(Z, y, H1, H2, C, rng, epochs):
    """Train MLP on Z (n, k), return params dict and final loss."""
    n, k = Z.shape
    # He initialization
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3
            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            dlogits = (probs / len(yy)).astype(np.float32)
            # backward
            dW3 = (h2.T @ dlogits).astype(np.float32) + L2 * W3
            db3 = dlogits.sum(axis=0).astype(np.float32)
            dh2 = (dlogits @ W3.T).astype(np.float32)
            dh2 *= (a2 > 0).astype(np.float32)
            dW2 = (h1.T @ dh2).astype(np.float32) + L2 * W2
            db2 = dh2.sum(axis=0).astype(np.float32)
            dh1 = (dh2 @ W2.T).astype(np.float32)
            dh1 *= (a1 > 0).astype(np.float32)
            dW1 = (z.T @ dh1).astype(np.float32) + L2 * W1
            db1 = dh1.sum(axis=0).astype(np.float32)
            grads = [dW1, db1, dW2, db2, dW3, db3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return {
        'W1': params[0], 'b1': params[1],
        'W2': params[2], 'b2': params[3],
        'W3': params[4], 'b3': params[5]
    }

def evaluate(Z, y, params):
    """Return accuracy on Z, y."""
    a1 = Z @ params['W1'] + params['b1']
    h1 = np.maximum(a1, 0)
    a2 = h1 @ params['W2'] + params['b2']
    h2 = np.maximum(a2, 0)
    logits = h2 @ params['W3'] + params['b3']
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Hyperparameter search on tuning split
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = CONFIGS[0]
    best_params = None
    best_enc = None
    epochs_search = min(MAX_EPOCHS, max(1, math.ceil(SAMPLES / len(X_train)) // 2))  # fewer epochs for search
    for H1, H2 in CONFIGS:
        enc = encode_features(X_train, types)
        Z_train = build_design(X_train, enc)
        Z_tune = build_design(X_tune, enc)
        params = train_model(Z_train, y_train, H1, H2, C, rng, epochs_search)
        acc = evaluate(Z_tune, y_tune, params)
        if acc > best_acc:
            best_acc = acc
            best_cfg = (H1, H2)
            best_params = params
            best_enc = enc

    # Retrain on full data with best config
    H1, H2 = best_cfg
    enc = encode_features(X, types)
    Z = build_design(X, enc)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)  # reset seed for final training
    params = train_model(Z, y, H1, H2, C, rng, epochs_full)

    # Build lookup tables for first layer: T1[d][256][H1]
    k = Z.shape[1]  # should equal d (64) because all ordinal
    T1 = np.zeros((k, 256, H1), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        dim = E.shape[1]
        T1[j] = (E.astype(np.float32) @ params['W1'][off:off+dim]).astype(np.float32)
        off += dim

    # Pack weights file
    # Header: magic, d, H1, H2, C
    magic = 0x4D4C5032  # '2PML'
    header = struct.pack("<5i", magic, k, H1, H2, C)
    # Data: T1 (k*256*H1), b1 (H1), W2 (H1*H2), b2 (H2), W3 (H2*C), b3 (C)
    data = bytearray()
    data += T1.astype("<f4").tobytes()
    data += params['b1'].astype("<f4").tobytes()
    data += params['W2'].astype("<f4").tobytes()
    data += params['b2'].astype("<f4").tobytes()
    data += params['W3'].astype("<f4").tobytes()
    data += params['b3'].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + data)

if __name__ == "__main__":
    main()
```

```c
/* Two-hidden-layer MLP inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (argmax class). */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || H1 <= 0 || H2 <= 0 || d <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * H1
        + 4L * H1
        + 4L * H1 * H2
        + 4L * H2
        + 4L * H2 * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

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
    float *scores = malloc(sizeof(float) * C);

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

        // Output layer: scores = b3 + h2 @ W3
        memcpy(scores, b3, C * sizeof(float));
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
            if (hv == 0.0f) continue;
            const float *w = W3 + (long)v * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```