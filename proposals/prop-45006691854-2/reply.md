```hypothesis
Family: label-embed-mlp. Uses position-specific byte embeddings (16 positions × 256 bytes × E) concatenated into a 16·E vector, followed by one ReLU hidden layer of size H and a linear output layer (65 classes). Embeddings capture local character similarities better than one-hot linear, while a single hidden layer adds non-linearity. Trained with Adam on 80% of train.in, hyperparameters (E, H) chosen on the last 20% tuning split, then retrained on all data. If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds ~15k Ir/item, the approach fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
PATIENCE = 5
VAL_FRAC = 0.2

# Hyperparameter grid to search on the tuning split
GRID = [
    {"E": 8, "H": 32},
    {"E": 8, "H": 64},
    {"E": 16, "H": 32},
    {"E": 16, "H": 64},
    {"E": 16, "H": 128},
    {"E": 32, "H": 64},
    {"E": 32, "H": 128},
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_val_split(X, y, val_frac, seed):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(n * (1 - val_frac))
    return idx[:split], idx[split:]

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def forward(X, params, E, H, C, d):
    # X: (n, d) uint8
    embed, W1, b1, W2, b2 = params
    n = X.shape[0]
    # gather embeddings: (n, d, E)
    # embed shape: (d, 256, E)
    emb = embed[X, np.arange(d)]  # (n, d, E)
    emb = emb.reshape(n, d * E)   # (n, d*E)
    h = emb @ W1 + b1
    h = np.maximum(h, 0)
    logits = h @ W2 + b2
    return logits, h, emb

def backward(X, y, logits, h, emb, params, E, H, C, d, L2):
    embed, W1, b1, W2, b2 = params
    n = X.shape[0]
    # softmax gradient
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    # output layer grads
    grad_W2 = h.T @ probs + L2 * W2
    grad_b2 = probs.sum(axis=0)
    # hidden layer grads
    grad_h = probs @ W2.T
    grad_h[h <= 0] = 0.0
    grad_W1 = emb.T @ grad_h + L2 * W1
    grad_b1 = grad_h.sum(axis=0)
    # embedding grads: (d, 256, E)
    grad_emb = np.zeros_like(embed)
    grad_emb_flat = grad_h @ W1.T  # (n, d*E)
    grad_emb_flat = grad_emb_flat.reshape(n, d, E)
    for i in range(n):
        for j in range(d):
            grad_emb[j, X[i, j]] += grad_emb_flat[i, j]
    return [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

def adam_update(params, grads, state, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    new_params = []
    new_state = []
    for p, g, (m, v) in zip(params, grads, state):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        new_params.append(p)
        new_state.append((m, v))
    return new_params, new_state

def init_params(d, E, H, C, rng):
    # embeddings: (d, 256, E)
    embed = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    # W1: (d*E, H)
    W1 = rng.normal(0, np.sqrt(2.0 / (d * E)), size=(d * E, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # W2: (H, C)
    W2 = rng.normal(0, np.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)
    return [embed, W1, b1, W2, b2]

def pack_weights(d, E, H, C, params):
    embed, W1, b1, W2, b2 = params
    # magic: 0x454D4244 ("EMBD")
    header = struct.pack("<5i", 0x454D4244, d, E, H, C)
    body = embed.astype("<f4").tobytes()
    body += W1.astype("<f4").tobytes()
    body += b1.astype("<f4").tobytes()
    body += W2.astype("<f4").tobytes()
    body += b2.astype("<f4").tobytes()
    return header + body

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Split indices for tuning (last 20% in file order)
    val_size = int(n * VAL_FRAC)
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    best_val_acc = -1.0
    best_params = None
    best_config = None

    for config in GRID:
        E, H = config["E"], config["H"]
        rng = np.random.default_rng(SEED)
        params = init_params(d, E, H, C, rng)
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        patience = 0
        best_local_acc = -1.0
        best_local_params = None

        for epoch in range(MAX_EPOCHS):
            # shuffle training data
            perm = rng.permutation(len(X_train))
            for s in range(0, len(X_train), BATCH):
                idx = perm[s:s + BATCH]
                xb, yb = X_train[idx], y_train[idx]
                logits, h, emb = forward(xb, params, E, H, C, d)
                grads = backward(xb, yb, logits, h, emb, params, E, H, C, d, L2)
                t += 1
                params, state = adam_update(params, grads, state, t, LR)
            # validation
            val_logits, _, _ = forward(X_val, params, E, H, C, d)
            val_pred = val_logits.argmax(axis=1)
            val_acc = accuracy(y_val, val_pred)
            if val_acc > best_local_acc:
                best_local_acc = val_acc
                best_local_params = [p.copy() for p in params]
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
        if best_local_acc > best_val_acc:
            best_val_acc = best_local_acc
            best_params = best_local_params
            best_config = config
            print(f"New best: E={E}, H={H}, val_acc={best_val_acc:.4f}")

    # Retrain on full data with best config
    E, H = best_config["E"], best_config["H"]
    rng = np.random.default_rng(SEED)
    params = init_params(d, E, H, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    # Train for a fixed number of epochs (or until convergence)
    # Use same early stopping on a small held-out part? We'll just train for MAX_EPOCHS*2 on full data.
    for epoch in range(MAX_EPOCHS * 2):
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, h, emb = forward(xb, params, E, H, C, d)
            grads = backward(xb, yb, logits, h, emb, params, E, H, C, d, L2)
            t += 1
            params, state = adam_update(params, grads, state, t, LR)

    # Save weights
    out_bytes = pack_weights(d, E, H, C, params)
    open(sys.argv[4], "wb").write(out_bytes)

if __name__ == "__main__":
    main()
```

```c
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
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        float *emb_concat = malloc(sizeof(float) * d * E);
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_concat + j * E, e, sizeof(float) * E);
        }

        // hidden layer: ReLU(emb_concat * W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += emb_concat[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }
        free(emb_concat);

        // output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```