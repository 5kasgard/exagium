```hypothesis
The label-embmlp family uses position-specific byte embeddings (16×256×D) summed into a dense vector, followed by an optional ReLU hidden layer (size H) and a linear output layer. This reduces the 4096 one-hot dimensions to a learned dense representation, capturing character similarities and non-linear interactions with fewer parameters than a full one-hot MLP. On tinyshakespeare, local character patterns (common n-grams, suffixes) should be well modeled by embeddings. If dev accuracy does not exceed label-linear's 0.3375, it suggests the task requires longer-range context or that the embedding dimension is insufficient for the needed statistics.
```

```python
"""label-embmlp: position-specific byte embeddings + optional 1-hidden-layer MLP.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"D": 16, "H": 0,   "lr": 3e-3, "l2": 1e-4},
    {"D": 16, "H": 32,  "lr": 3e-3, "l2": 1e-4},
    {"D": 32, "H": 0,   "lr": 3e-3, "l2": 1e-4},
    {"D": 32, "H": 32,  "lr": 3e-3, "l2": 1e-4},
]
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, D, H, C, rng):
    """Return (emb, W1, b1, W2, b2) or (emb, W, b) if H==0."""
    emb = []
    for _ in range(d):
        scale = math.sqrt(6.0 / (256 + D))
        e = (rng.uniform(-1, 1, (256, D)) * scale).astype(np.float32)
        emb.append(e)
    if H:
        W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / (D + H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return emb, W1, b1, W2, b2
    else:
        W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / (D + C))).astype(np.float32)
        b = np.zeros(C, np.float32)
        return emb, W, b

def forward(emb, X, params, H):
    """X: (B, d) uint8. Returns logits (B, C) and cache for backward."""
    B, d = X.shape
    # sum embeddings
    sum_emb = np.zeros((B, emb[0].shape[1]), dtype=np.float32)
    for j in range(d):
        sum_emb += emb[j][X[:, j]]
    if H:
        W1, b1, W2, b2 = params
        hidden = sum_emb @ W1 + b1
        np.maximum(hidden, 0, out=hidden)  # ReLU in-place
        logits = hidden @ W2 + b2
        cache = (sum_emb, hidden)
    else:
        W, b = params
        logits = sum_emb @ W + b
        cache = (sum_emb,)
    return logits, cache

def loss_and_grad(logits, y, cache, params, emb, X, H, l2):
    """Compute gradients for all parameters. Returns (grad_emb, grad_params)."""
    B, C = logits.shape
    # softmax cross-entropy gradient
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    dlogits = probs / B  # (B, C)

    if H:
        W1, b1, W2, b2 = params
        sum_emb, hidden = cache
        # output layer
        dW2 = hidden.T @ dlogits + l2 * W2
        db2 = dlogits.sum(axis=0)
        dhidden = dlogits @ W2.T
        dhidden *= (hidden > 0)  # ReLU derivative
        # hidden layer
        dW1 = sum_emb.T @ dhidden + l2 * W1
        db1 = dhidden.sum(axis=0)
        dsum_emb = dhidden @ W1.T
        grad_params = (dW1, db1, dW2, db2)
    else:
        W, b = params
        sum_emb, = cache
        dW = sum_emb.T @ dlogits + l2 * W
        db = dlogits.sum(axis=0)
        dsum_emb = dlogits @ W.T
        grad_params = (dW, db)

    # embedding gradients
    B, d = X.shape
    D = emb[0].shape[1]
    grad_emb = [np.zeros_like(e) for e in emb]
    for j in range(d):
        idx = X[:, j]
        np.add.at(grad_emb[j], idx, dsum_emb)
    return grad_emb, grad_params

def evaluate(emb, params, X, y, H):
    """Return accuracy on full dataset."""
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        Xb = X[s:s+BATCH]
        yb = y[s:s+BATCH]
        logits, _ = forward(emb, Xb, params, H)
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def train_on_split(emb, params, X, y, idx_train, idx_val, H, lr, l2, epochs, rng):
    """Train on idx_train, validate on idx_val. Returns best val accuracy."""
    # Adam state
    all_params = []
    if H:
        all_params.extend([emb, params[0], params[1], params[2], params[3]])  # emb, W1, b1, W2, b2
    else:
        all_params.extend([emb, params[0], params[1]])
    # flatten for Adam
    flat_params = []
    for p in all_params:
        if isinstance(p, list):
            flat_params.extend(p)
        else:
            flat_params.append(p)
    m = [np.zeros_like(p) for p in flat_params]
    v = [np.zeros_like(p) for p in flat_params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    n_train = len(X_train)

    best_val = 0.0
    for epoch in range(epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_train[idx], y_train[idx]
            logits, cache = forward(emb, Xb, params, H)
            grad_emb, grad_params = loss_and_grad(logits, yb, cache, params, emb, Xb, H, l2)
            # pack gradients in same order as flat_params
            flat_grads = []
            for g in grad_emb:
                flat_grads.append(g)
            for g in grad_params:
                flat_grads.append(g)
            t += 1
            for i, (p, g) in enumerate(zip(flat_params, flat_grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # validate
        val_acc = evaluate(emb, params, X_val, y_val, H)
        if val_acc > best_val:
            best_val = val_acc
    return best_val

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    # split: last 20% for tuning
    split = int(0.8 * n)
    idx_train = np.arange(split)
    idx_val = np.arange(split, n)

    # Grid search on tuning split
    best_cfg = None
    best_val = -1.0
    for i, cfg in enumerate(GRID):
        D, H, lr, l2 = cfg["D"], cfg["H"], cfg["lr"], cfg["l2"]
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(idx_train)))
        rng = np.random.default_rng(SEED + i)
        emb, *params = init_params(d, D, H, C, rng)
        val_acc = train_on_split(emb, params, X, y, idx_train, idx_val, H, lr, l2, epochs, rng)
        print(f"Config {cfg}: val_acc={val_acc:.4f}", file=sys.stderr)
        if val_acc > best_val:
            best_val = val_acc
            best_cfg = cfg

    print(f"Best config: {best_cfg} with val_acc={best_val:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    D, H, lr, l2 = best_cfg["D"], best_cfg["H"], best_cfg["lr"], best_cfg["l2"]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)
    emb, *params = init_params(d, D, H, C, rng)
    # Train on all data (no validation)
    X_train, y_train = X, y
    n_train = n
    # Adam state
    all_params = [emb]
    all_params.extend(params)
    flat_params = []
    for p in all_params:
        if isinstance(p, list):
            flat_params.extend(p)
        else:
            flat_params.append(p)
    m = [np.zeros_like(p) for p in flat_params]
    v = [np.zeros_like(p) for p in flat_params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_train[idx], y_train[idx]
            logits, cache = forward(emb, Xb, params, H)
            grad_emb, grad_params = loss_and_grad(logits, yb, cache, params, emb, Xb, H, l2)
            flat_grads = []
            for g in grad_emb:
                flat_grads.append(g)
            for g in grad_params:
                flat_grads.append(g)
            t += 1
            for i, (p, g) in enumerate(zip(flat_params, flat_grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Save weights
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    header = struct.pack("<5i", magic, d, D, H, C)
    out = bytearray(header)
    for e in emb:
        out += e.astype("<f4").tobytes()
    if H:
        W1, b1, W2, b2 = params
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        W, b = params
        out += W.astype("<f4").tobytes()
        out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad header values\n"); return 3; }

    // Expected size
    long expected = 20L + (long)d * 256 * D * 4;
    if (H) {
        expected += ((long)D * H + H + (long)H * C + C) * 4;
    } else {
        expected += ((long)D * C + C) * 4;
    }
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1, *b1, *W2, *b2, *W, *b;
    if (H) {
        W1 = emb + (long)d * 256 * D;
        b1 = W1 + (long)D * H;
        W2 = b1 + H;
        b2 = W2 + (long)H * C;
    } else {
        W = emb + (long)d * 256 * D;
        b = W + (long)D * C;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = (float *)malloc(sizeof(float) * D);
    float *hidden = H ? (float *)malloc(sizeof(float) * H) : NULL;
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // sum embeddings
        memset(sum_emb, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int kk = 0; kk < D; kk++) sum_emb[kk] += e[kk];
        }

        if (H) {
            // hidden layer
            for (int h = 0; h < H; h++) hidden[h] = b1[h];
            for (int kk = 0; kk < D; kk++) {
                float v = sum_emb[kk];
                if (v != 0.0f) {
                    const float *w = W1 + (long)kk * H;
                    for (int h = 0; h < H; h++) hidden[h] += v * w[h];
                }
            }
            for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

            // output layer
            for (int c = 0; c < C; c++) logits[c] = b2[c];
            for (int h = 0; h < H; h++) {
                float v = hidden[h];
                if (v != 0.0f) {
                    const float *w = W2 + (long)h * C;
                    for (int c = 0; c < C; c++) logits[c] += v * w[c];
                }
            }
        } else {
            // linear on embeddings
            for (int c = 0; c < C; c++) logits[c] = b[c];
            for (int kk = 0; kk < D; kk++) {
                float v = sum_emb[kk];
                if (v != 0.0f) {
                    const float *w = W + (long)kk * C;
                    for (int c = 0; c < C; c++) logits[c] += v * w[c];
                }
            }
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(sum_emb); free(hidden); free(logits);
    return 0;
}
```