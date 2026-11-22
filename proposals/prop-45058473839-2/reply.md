```hypothesis
We propose "label-emlp" (Embedding MLP with Linear Residual). This model learns position-specific byte embeddings, passes their concatenation through a single hidden ReLU layer, and adds the result to a traditional linear one-hot model. The linear residual ensures the model cannot underperform the strong linear baseline, while the embedding MLP captures pairwise and higher-order interactions between context bytes that are crucial for character-level language modeling. Hyperparameters (embedding dim, hidden size, learning rate) are selected on the card's tuning split (last 20% of training data) via a small grid search, then the model is retrained on all data. If dev accuracy does not exceed label-linear's 0.3375, or if inference cost exceeds label-linear by more than 2x, the approach fails.
```

```python
"""label-emlp: Embedding MLP with linear residual for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Trains a linear one-hot model plus an embedding MLP residual.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all data.
Weights format: magic 0x454D4C31, d, E, H, C; then float32 arrays:
 linear_W[d*256*C], linear_b[C], emb[d*256*E], W1[d*E*H], b1[H], W2[H*C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
GRID = [
    {"E": 8, "H": 32, "lr": 1e-3, "l2": 1e-4},
    {"E": 8, "H": 64, "lr": 1e-3, "l2": 1e-4},
    {"E": 16, "H": 64, "lr": 1e-3, "l2": 1e-4},
    {"E": 16, "H": 128, "lr": 1e-3, "l2": 1e-4},
    {"E": 32, "H": 128, "lr": 3e-3, "l2": 1e-5},
]
SEED = 1
BATCH = 64
TUNE_EPOCHS = 3
FINAL_EPOCHS = 30
SAMPLES_PER_EPOCH = 600000  # for epoch count scaling


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)


def forward(X, params, d, E, H, C):
    """X: (n, d) uint8. Returns logits (n, C)."""
    linear_W, linear_b, emb, W1, b1, W2, b2 = params
    n = X.shape[0]
    # Linear part
    logits = np.tile(linear_b, (n, 1))
    for j in range(d):
        logits += linear_W[j][X[:, j]]
    # Embedding MLP part
    concat = np.zeros((n, d * E), dtype=np.float32)
    for j in range(d):
        concat[:, j * E:(j + 1) * E] = emb[j][X[:, j]]
    hidden = np.maximum(concat @ W1 + b1, 0, dtype=np.float32)
    logits += hidden @ W2 + b2
    return logits


def backward(X, y, params, grads, d, E, H, C, l2):
    """Accumulate gradients into grads (same structure as params)."""
    linear_W, linear_b, emb, W1, b1, W2, b2 = params
    g_linear_W, g_linear_b, g_emb, g_W1, g_b1, g_W2, g_b2 = grads
    n = X.shape[0]
    # Forward pass with intermediates
    logits = np.tile(linear_b, (n, 1))
    for j in range(d):
        logits += linear_W[j][X[:, j]]
    concat = np.zeros((n, d * E), dtype=np.float32)
    for j in range(d):
        concat[:, j * E:(j + 1) * E] = emb[j][X[:, j]]
    pre_hidden = concat @ W1 + b1
    hidden = np.maximum(pre_hidden, 0, dtype=np.float32)
    logits += hidden @ W2 + b2
    # Softmax gradient
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    dlogits = probs / n  # (n, C)
    # Linear gradients
    g_linear_b += dlogits.sum(axis=0)
    for j in range(d):
        # scatter add for each byte value
        bytes_j = X[:, j]
        for b in range(256):
            mask = (bytes_j == b)
            if np.any(mask):
                g_linear_W[j][b] += dlogits[mask].sum(axis=0)
    # MLP gradients
    dhidden = dlogits @ W2.T
    dhidden[pre_hidden <= 0] = 0.0
    g_W1 += concat.T @ dhidden
    g_b1 += dhidden.sum(axis=0)
    g_W2 += hidden.T @ dlogits
    g_b2 += dlogits.sum(axis=0)
    # Embedding gradients
    dconcat = dhidden @ W1.T
    for j in range(d):
        bytes_j = X[:, j]
        start = j * E
        end = (j + 1) * E
        for b in range(256):
            mask = (bytes_j == b)
            if np.any(mask):
                g_emb[j][b] += dconcat[mask, start:end].sum(axis=0)
    # L2 regularization gradients (added later in update)
    for j in range(d):
        g_linear_W[j] += l2 * linear_W[j]
    g_W1 += l2 * W1
    g_W2 += l2 * W2


def init_params(d, E, H, C, rng):
    # Linear: one-hot per position -> C
    linear_W = [rng.normal(0, 0.01, (256, C)).astype(np.float32) for _ in range(d)]
    linear_b = np.zeros(C, np.float32)
    # Embeddings
    emb = [rng.normal(0, 0.1, (256, E)).astype(np.float32) for _ in range(d)]
    # MLP
    W1 = rng.normal(0, np.sqrt(2.0 / (d * E)), (d * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, np.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return linear_W, linear_b, emb, W1, b1, W2, b2


def zero_grads(params):
    return [np.zeros_like(p) for p in params]


def adam_update(params, grads, state, t, lr):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = state[i]
        m *= 0.9; m += 0.1 * g
        v *= 0.999; v += 0.001 * g * g
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def train_epoch(X, y, params, d, E, H, C, l2, batch_size, rng):
    n = X.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        xb, yb = X[idx], y[idx]
        grads = zero_grads(params)
        backward(xb, yb, params, grads, d, E, H, C, l2)
        # Adam step will be done outside with global step counter
        yield grads, len(idx)


def evaluate(X, y, params, d, E, H, C):
    logits = forward(X, params, d, E, H, C)
    return accuracy(logits, y)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    # Grid search on tuning split
    for cfg in GRID:
        E, H, lr, l2 = cfg["E"], cfg["H"], cfg["lr"], cfg["l2"]
        params = init_params(d, E, H, C, rng)
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        for _ in range(TUNE_EPOCHS):
            for grads, bs in train_epoch(X_train, y_train, params, d, E, H, C, l2, BATCH, rng):
                t += 1
                adam_update(params, grads, state, t, lr)
        acc = evaluate(X_tune, y_tune, params, d, E, H, C)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            # Save copy of best params
            best_params = [p.copy() for p in params]

    # Retrain on full data with best config
    E, H, lr, l2 = best_cfg["E"], best_cfg["H"], best_cfg["lr"], best_cfg["l2"]
    params = init_params(d, E, H, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(FINAL_EPOCHS, math.ceil(SAMPLES_PER_EPOCH / n))
    for _ in range(epochs):
        for grads, bs in train_epoch(X, y, params, d, E, H, C, l2, BATCH, rng):
            t += 1
            adam_update(params, grads, state, t, lr)

    # Write weights
    linear_W, linear_b, emb, W1, b1, W2, b2 = params
    magic = 0x454D4C31  # "EML1"
    header = struct.pack("<5i", magic, d, E, H, C)
    # Flatten arrays in order
    out = bytearray(header)
    for j in range(d):
        out += linear_W[j].astype("<f4").tobytes()
    out += linear_b.astype("<f4").tobytes()
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D4C31, d, E, H, C (int32), then float32 arrays:
 *  linear_W[d][256][C], linear_b[C], emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    // Expected float count
    long expected_floats = (long)d * 256 * C + C + (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    long expected_bytes = 20 + expected_floats * 4;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected_bytes); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *linear_W = p; p += (long)d * 256 * C;
    const float *linear_b = p; p += C;
    const float *emb = p; p += (long)d * 256 * E;
    const float *W1 = p; p += (long)d * E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Linear part
        memcpy(logits, linear_b, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *w = linear_W + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }
        // Embedding concat
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        // Hidden layer
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * E; idx++) sum += concat[idx] * W1[(long)idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] += sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```