```hypothesis
Family: label-embed (embedding-based MLP). Each of the 14 byte columns gets a learned embedding vector (size E). Concatenated embeddings feed a single ReLU hidden layer (size H) then a 2-class output. Embeddings capture category similarities, reducing parameters vs one-hot and adding non-linearity. Should beat label-linear (0.8524) and label-mlp16 (0.8591) with lower inference cost than mlp16. Failure: if dev accuracy ≤ 0.8524 or Ir/item ≥ 2325.
```

```python
"""label-embed: embedding + one hidden ReLU layer for tabular byte features.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then retrain on all."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning split search
GRID = {
    'E': [4, 8],
    'H': [8, 16],
    'lr': [1e-3, 3e-3],
    'L2': [1e-4],
    'batch': [64],
    'epochs': [100],
    'samples': 600000,
}
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t logits."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def forward(X_idx, params, d, E, H, C):
    """X_idx: (n, d) uint8. Returns logits (n, C), and cache for backward."""
    emb, W1, b1, W2, b2 = params
    n = X_idx.shape[0]
    # Embedding lookup: (n, d, E)
    emb_lookup = np.stack([emb[j][X_idx[:, j]] for j in range(d)], axis=1)  # (n, d, E)
    # Flatten to (n, d*E)
    flat = emb_lookup.reshape(n, -1)
    # Hidden layer
    hidden_pre = flat @ W1 + b1
    hidden = np.maximum(hidden_pre, 0)
    # Output
    logits = hidden @ W2 + b2
    cache = (emb_lookup, flat, hidden_pre, hidden)
    return logits, cache

def backward(X_idx, y, cache, params, grads_out, d, E, H, C, L2):
    """Compute gradients and accumulate into grads_out list."""
    emb, W1, b1, W2, b2 = params
    emb_lookup, flat, hidden_pre, hidden = cache
    n = X_idx.shape[0]
    # Output layer gradients
    g_logits = grads_out  # (n, C)
    g_W2 = hidden.T @ g_logits + L2 * W2
    g_b2 = g_logits.sum(axis=0)
    g_hidden = g_logits @ W2.T
    # ReLU
    g_hidden_pre = g_hidden * (hidden_pre > 0)
    # Hidden layer gradients
    g_W1 = flat.T @ g_hidden_pre + L2 * W1
    g_b1 = g_hidden_pre.sum(axis=0)
    g_flat = g_hidden_pre @ W1.T  # (n, d*E)
    # Embedding gradients
    g_emb = [np.zeros_like(emb[j]) for j in range(d)]
    g_flat_reshaped = g_flat.reshape(n, d, E)
    for j in range(d):
        idx = X_idx[:, j]
        np.add.at(g_emb[j], idx, g_flat_reshaped[:, j])
    return [g_emb, g_W1, g_b1, g_W2, g_b2]

def evaluate(X_idx, y, params, d, E, H, C):
    logits, _ = forward(X_idx, params, d, E, H, C)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def train_model(X_train, y_train, X_val, y_val, d, C, E, H, lr, L2, batch, epochs, rng):
    n = X_train.shape[0]
    # Initialize parameters
    emb = [rng.uniform(-0.1, 0.1, (256, E)).astype(np.float32) for _ in range(d)]
    W1 = (rng.uniform(-1, 1, (d*E, H)) * math.sqrt(6.0 / (d*E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = []
    for p_list in params:
        if isinstance(p_list, list):
            state.append([(np.zeros_like(p), np.zeros_like(p)) for p in p_list])
        else:
            state.append((np.zeros_like(p_list), np.zeros_like(p_list)))
    t = 0
    best_val = -1
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb, yb = X_train[idx], y_train[idx]
            logits, cache = forward(xb, params, d, E, H, C)
            g_logits = softmax_cross_entropy_grad(logits, yb)
            grads = backward(xb, yb, cache, params, g_logits, d, E, H, C, L2)
            t += 1
            # Update embeddings
            for j in range(d):
                p = params[0][j]
                g = grads[0][j]
                m, v = state[0][j]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            # Update W1, b1, W2, b2
            for i, (p, g) in enumerate(zip(params[1:], grads[1:]), start=1):
                m, v = state[i]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        val_acc = evaluate(X_val, y_val, params, d, E, H, C)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [ [e.copy() for e in params[0]],
                            params[1].copy(), params[2].copy(),
                            params[3].copy(), params[4].copy() ]
    return best_val, best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_overall = -1
    best_hparams = None
    best_model = None
    # Grid search
    for E in GRID['E']:
        for H in GRID['H']:
            for lr in GRID['lr']:
                for L2 in GRID['L2']:
                    for batch in GRID['batch']:
                        for epochs in GRID['epochs']:
                            epochs = min(epochs, math.ceil(GRID['samples'] / len(X_train)))
                            val_acc, model = train_model(X_train, y_train, X_val, y_val,
                                                         d, C, E, H, lr, L2, batch, epochs, rng)
                            if val_acc > best_overall:
                                best_overall = val_acc
                                best_hparams = (E, H, lr, L2, batch, epochs)
                                best_model = model
    # Retrain on full data with best hyperparameters
    E, H, lr, L2, batch, epochs = best_hparams
    epochs = min(epochs, math.ceil(GRID['samples'] / n))
    _, final_model = train_model(X, y, X[:1], y[:1], d, C, E, H, lr, L2, batch, epochs, rng)
    emb, W1, b1, W2, b2 = final_model
    # Pack weights: magic, d, E, H, C, then embeddings (d*256*E), W1 (d*E*H), b1 (H), W2 (H*C), b2 (C)
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header
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
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20
        + 4L * d * 256 * E          /* embeddings */
        + 4L * d * E * H            /* W1 */
        + 4L * H                    /* b1 */
        + 4L * H * C                /* W2 */
        + 4L * C;                   /* b2 */
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += d * 256 * E;
    const float *W1 = p; p += d * E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += H * C;
    const float *b2 = p;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* hidden = b1 */
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        /* accumulate embeddings * W1 */
        for (int j = 0; j < d; j++) {
            const float *e = emb + (j * 256 + xi[j]) * E;
            const float *w = W1 + j * E * H;
            for (int ee = 0; ee < E; ee++) {
                float ev = e[ee];
                if (ev == 0.0f) continue;
                for (int h = 0; h < H; h++) hidden[h] += ev * w[ee * H + h];
            }
        }
        /* ReLU */
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;
        /* output */
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            for (int c = 0; c < C; c++) scores[c] += hv * W2[h * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```