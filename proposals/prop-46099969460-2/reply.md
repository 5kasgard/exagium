```hypothesis
Family: label-embmlp — per-position byte embeddings + one ReLU hidden layer.
Why: The linear model uses one-hot (256-d) per position → 16*256*65 ≈ 266k params, which overfits and is slow. Learned embeddings (E=16–32) compress each position, a hidden layer (H=64–128) captures interactions, and the total params drop to ~50–150k. This should improve accuracy by better generalization and reduce CPU cost by shrinking the largest matrix multiply (from 16*256*65 to 16*E*H + H*65). If dev accuracy does not exceed label-linear (0.3375), it means the task needs the full one-hot capacity or the embedding dimension is too small.
```

```python
"""label-embmlp: per-position byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter search space (embedding_dim, hidden_dim)
CONFIGS = [(16, 64), (32, 64), (16, 128), (32, 128)]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
SEARCH_EPOCHS = 10  # epochs for tuning split evaluation

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, E, H, C, rng):
    # embeddings: [d, 256, E]
    W_emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    # hidden layer: input size d*E
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # output layer
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return [W_emb, W1, b1, W2, b2]

def forward(X, params, d, E, H, C):
    """X: (N, d) uint8. Returns logits (N, C), and cache for backward."""
    W_emb, W1, b1, W2, b2 = params
    N = X.shape[0]
    # Embedding lookup and concatenation
    # We'll do a loop over positions for memory efficiency
    x_emb = np.zeros((N, d * E), dtype=np.float32)
    for j in range(d):
        x_emb[:, j*E:(j+1)*E] = W_emb[j, X[:, j]]
    # Hidden layer
    h_pre = x_emb @ W1 + b1
    h = np.maximum(h_pre, 0, dtype=np.float32)
    # Output
    logits = h @ W2 + b2
    cache = (x_emb, h_pre, h)
    return logits, cache

def loss_and_grad(X, y, params, cache, d, E, H, C, L2):
    W_emb, W1, b1, W2, b2 = params
    x_emb, h_pre, h = cache
    N = X.shape[0]
    # Softmax + cross-entropy gradient
    logits = h @ W2 + b2
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    probs /= N
    # Output layer gradients
    dW2 = h.T @ probs + L2 * W2
    db2 = probs.sum(axis=0)
    # Hidden layer gradients
    dh = probs @ W2.T
    dh[h_pre <= 0] = 0.0  # ReLU derivative
    dW1 = x_emb.T @ dh + L2 * W1
    db1 = dh.sum(axis=0)
    # Embedding gradients
    d_x_emb = dh @ W1.T  # (N, d*E)
    dW_emb = np.zeros_like(W_emb, dtype=np.float32)
    for j in range(d):
        # scatter add: for each example, add gradient to the embedding of the byte at position j
        np.add.at(dW_emb[j], X[:, j], d_x_emb[:, j*E:(j+1)*E])
    dW_emb += L2 * W_emb
    return [dW_emb, dW1, db1, dW2, db2], float(-np.log(probs[np.arange(N), y] + 1e-12).mean())

def accuracy(X, y, params, d, E, H, C):
    logits, _ = forward(X, params, d, E, H, C)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_model(X, y, params, d, E, H, C, epochs, rng, batch_size=BATCH):
    n = X.shape[0]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            Xb, yb = X[idx], y[idx]
            logits, cache = forward(Xb, params, d, E, H, C)
            grads, _ = loss_and_grad(Xb, yb, params, cache, d, E, H, C, L2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20%
    tune_size = int(0.2 * n)
    train_size = n - tune_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_tune, y_tune = X[train_size:], y[train_size:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_config = None
    # Search over configs
    for E, H in CONFIGS:
        params = init_params(d, E, H, C, rng)
        # Train briefly on train split
        params = train_model(X_train, y_train, params, d, E, H, C, SEARCH_EPOCHS, rng)
        acc = accuracy(X_tune, y_tune, params, d, E, H, C)
        print(f"Config E={E} H={H}: tune accuracy={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]
            best_config = (E, H)
    print(f"Best config: E={best_config[0]} H={best_config[1]} (acc={best_acc:.4f})", file=sys.stderr)
    # Retrain on full data with best config
    E, H = best_config
    params = init_params(d, E, H, C, rng)
    full_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params = train_model(X, y, params, d, E, H, C, full_epochs, rng)
    # Save weights
    W_emb, W1, b1, W2, b2 = params
    # Header: magic, d, E, H, C
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, E, H, C)
    # Embeddings: d * 256 * E floats
    emb_bytes = W_emb.astype("<f4").tobytes()
    # Hidden layer: W1 (d*E x H), b1 (H)
    w1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    # Output layer: W2 (H x C), b2 (C)
    w2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + w1_bytes + b1_bytes + w2_bytes + b2_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index 0..64).
 * Weights format: magic(0x454D424D), d, E, H, C; then float32 arrays in order:
 *   emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(x_emb + j * E, e, E * sizeof(float));
        }
        // Hidden layer: h = ReLU(x_emb @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += x_emb[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer: logits = h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```