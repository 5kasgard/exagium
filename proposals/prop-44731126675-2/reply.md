```hypothesis
Family: label-embmlp (shared byte embeddings + one ReLU hidden layer). Tinyshakespeare characters share semantics across positions; low-dimensional embeddings capture this and reduce parameters vs. one-hot linear, improving generalization. A small hidden layer models interactions between context positions. If dev accuracy ≤ label-linear (0.3375) or inference cost exceeds 2× label-linear, the hypothesis fails.
```

```python
"""label-embmlp: shared byte embeddings + one ReLU hidden layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Tunes embedding dim E and hidden size H on last 20% of train.in, then trains on all.
Exports weights: magic 0x454D424C, d, E, H, C; float32 emb[256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Fixed training hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Grid to search on validation split
GRID = [(8, 64), (16, 64), (8, 128)]  # (E, H)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_val_split(X, y, val_frac=0.2):
    n = X.shape[0]
    val_size = int(n * val_frac)
    return X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:]

def accuracy(logits, y):
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def forward(X, emb, W1, b1, W2, b2):
    """X: (n, d) uint8 indices. Returns logits (n, C)."""
    n, d = X.shape
    E = emb.shape[1]
    # Embedding lookup: (n, d, E)
    x_emb = emb[X]  # (n, d, E)
    x_flat = x_emb.reshape(n, d * E)  # (n, d*E)
    h = np.maximum(x_flat @ W1 + b1, 0)  # (n, H)
    logits = h @ W2 + b2  # (n, C)
    return logits, h, x_flat

def backward(X, y, logits, h, x_flat, emb, W1, W2, batch_size):
    """Compute gradients for all parameters."""
    n, d = X.shape
    E = emb.shape[1]
    H = W1.shape[1]
    C = W2.shape[1]

    # dlogits: (n, C)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    dlogits = probs / n

    # W2, b2
    dW2 = h.T @ dlogits + L2 * W2
    db2 = dlogits.sum(axis=0)

    # dh
    dh = dlogits @ W2.T
    dh *= (h > 0).astype(np.float32)

    # W1, b1
    dW1 = x_flat.T @ dh + L2 * W1
    db1 = dh.sum(axis=0)

    # dx_flat
    dx_flat = dh @ W1.T  # (n, d*E)

    # Embedding gradients: scatter-add
    dE = np.zeros_like(emb, dtype=np.float32)
    dx_emb = dx_flat.reshape(n, d, E)
    for j in range(d):
        idx = X[:, j]  # (n,)
        np.add.at(dE, idx, dx_emb[:, j, :])

    return dE, dW1, db1, dW2, db2

def adam_update(params, grads, states, t, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = states[i]
        m *= beta1; m += (1 - beta1) * g
        v *= beta2; v += (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)

def train_model(X, y, E, H, epochs, rng):
    n, d = X.shape
    C = 65
    # Initialize parameters
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, h, x_flat = forward(xb, emb, W1, b1, W2, b2)
            grads = backward(xb, yb, logits, h, x_flat, emb, W1, W2, len(idx))
            t += 1
            adam_update(params, grads, states, t)
    return params

def evaluate(X, y, params):
    emb, W1, b1, W2, b2 = params
    logits, _, _ = forward(X, emb, W1, b1, W2, b2)
    return accuracy(logits, y)

def save_weights(params, d, E, H, C, path):
    emb, W1, b1, W2, b2 = params
    magic = 0x454D424C  # 'EMBL'
    header = struct.pack("<5i", magic, d, E, H, C)
    with open(path, "wb") as f:
        f.write(header)
        f.write(emb.astype("<f4").tobytes())
        f.write(W1.astype("<f4").tobytes())
        f.write(b1.astype("<f4").tobytes())
        f.write(W2.astype("<f4").tobytes())
        f.write(b2.astype("<f4").tobytes())

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]

    # Split for tuning
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, 0.2)
    n_tr = X_tr.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))

    best_acc = -1.0
    best_params = None
    best_EH = None

    for E, H in GRID:
        rng = np.random.default_rng(SEED)
        params = train_model(X_tr, y_tr, E, H, epochs, rng)
        acc = evaluate(X_val, y_val, params)
        print(f"E={E}, H={H}, val_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_EH = (E, H)

    print(f"Best: E={best_EH[0]}, H={best_EH[1]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data
    n_all = X.shape[0]
    epochs_all = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    rng = np.random.default_rng(SEED)
    final_params = train_model(X, y, best_EH[0], best_EH[1], epochs_all, rng)

    save_weights(final_params, d, best_EH[0], best_EH[1], C, sys.argv[4])

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x454D424C, int32 d, E, H, C; float32 emb[256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    unsigned char *wdata = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hd = (int32_t *)wdata;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights: invalid dims\n"); return 3; }
    long expected = 20L + 4L * (256L * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256L * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and flatten
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(x_emb + j * E, e, sizeof(float) * E);
        }

        // Hidden layer: ReLU(x_emb @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += x_emb[f] * W1[(long)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax
        best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(wdata); free(x); free(x_emb); free(hidden); free(logits);
    return 0;
}
```