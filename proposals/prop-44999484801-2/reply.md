```hypothesis
Family: label-embmlp (per-position embeddings + one hidden ReLU layer).
Why: Character prediction benefits from dense embeddings capturing semantic similarity between characters; a small MLP on top models interactions. With E=16, H=64, it has ~25k params vs linear's 266k, so lower compute and memory.
Failure: If dev accuracy <= 0.3375 (linear) or Ir/item >= 7359.9 (linear cost), the family does not improve.
```

```python
"""label-embmlp: per-position embeddings + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then trained on all data.
Weights format: magic 0x454D424D, d, E, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Grid search space (kept small for speed)
E_CHOICES = [8, 16, 32]
H_CHOICES = [32, 64, 128]
SEED = 1
LR, L2, BATCH = 3e-3, 1e-4, 64
GRID_EPOCHS = 5
FINAL_EPOCHS = 20
SAMPLES_PER_EPOCH = 600000  # max samples per epoch for final training


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,)
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B


def train_model(X, y, E, H, epochs, rng):
    n, d = X.shape
    C = 65
    # Initialize embeddings per position: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    # First layer weights: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    # Flatten for Adam state
    flat_params = np.concatenate([p.ravel() for p in params])
    m = np.zeros_like(flat_params)
    v = np.zeros_like(flat_params)
    t = 0

    def unpack(flat):
        idx = 0
        out = []
        for shp in shapes:
            sz = int(np.prod(shp))
            out.append(flat[idx:idx + sz].reshape(shp))
            idx += sz
        return out

    def pack(emb, W1, b1, W2, b2):
        return np.concatenate([emb.ravel(), W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])

    def forward(X_batch, emb, W1, b1, W2, b2):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Embedding lookup: (B, d, E)
        emb_seq = np.stack([emb[j, X_batch[:, j]] for j in range(d)], axis=1)
        # Flatten to (B, d*E)
        flat = emb_seq.reshape(B, -1)
        # Hidden layer
        hidden = flat @ W1 + b1
        hidden = np.maximum(hidden, 0)  # ReLU
        # Output
        logits = hidden @ W2 + b2
        return logits, hidden, flat

    def backward(X_batch, y_batch, hidden, flat, emb, W1, b1, W2, b2):
        B = X_batch.shape[0]
        d, _, E = emb.shape
        # Output gradient
        logits_grad = softmax_cross_entropy_grad(hidden @ W2 + b2, y_batch)  # (B, C)
        # Gradients for W2, b2
        grad_W2 = hidden.T @ logits_grad  # (H, C)
        grad_b2 = logits_grad.sum(axis=0)
        # Gradient back to hidden
        grad_hidden = logits_grad @ W2.T  # (B, H)
        grad_hidden[hidden <= 0] = 0  # ReLU derivative
        # Gradient for W1, b1
        grad_W1 = flat.T @ grad_hidden  # (d*E, H)
        grad_b1 = grad_hidden.sum(axis=0)
        # Gradient for embeddings
        grad_flat = grad_hidden @ W1.T  # (B, d*E)
        grad_emb_seq = grad_flat.reshape(B, d, E)  # (B, d, E)
        grad_emb = np.zeros_like(emb)
        for j in range(d):
            # scatter add grad_emb_seq[:, j] to grad_emb[j, X_batch[:, j]]
            np.add.at(grad_emb[j], X_batch[:, j], grad_emb_seq[:, j])
        return grad_emb, grad_W1, grad_b1, grad_W2, grad_b2

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            logits, hidden, flat = forward(Xb, emb, W1, b1, W2, b2)
            g_emb, g_W1, g_b1, g_W2, g_b2 = backward(Xb, yb, hidden, flat, emb, W1, b1, W2, b2)
            # Pack gradients
            grads_flat = pack(g_emb, g_W1, g_b1, g_W2, g_b2)
            # Adam update
            t += 1
            m = 0.9 * m + 0.1 * grads_flat
            v = 0.999 * v + 0.001 * grads_flat * grads_flat
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            flat_params -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            # Unpack updated params
            emb, W1, b1, W2, b2 = unpack(flat_params)
            # L2 regularization (weight decay)
            for p in [emb, W1, W2]:
                p -= LR * L2 * p

    return emb, W1, b1, W2, b2


def evaluate(X, y, emb, W1, b1, W2, b2):
    logits, _, _ = forward(X, emb, W1, b1, W2, b2)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_acc = -1.0
    best_cfg = None
    best_weights = None
    base_rng = np.random.default_rng(SEED)

    for E in E_CHOICES:
        for H in H_CHOICES:
            # Deterministic seed per config
            cfg_seed = hash((E, H)) & 0xFFFFFFFF
            rng = np.random.default_rng(cfg_seed)
            emb, W1, b1, W2, b2 = train_model(X_train, y_train, E, H, GRID_EPOCHS, rng)
            acc = evaluate(X_tune, y_tune, emb, W1, b1, W2, b2)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (E, H)
                best_weights = (emb, W1, b1, W2, b2)

    # Retrain on full data with best config
    E, H = best_cfg
    rng = np.random.default_rng(SEED)
    emb, W1, b1, W2, b2 = train_model(X, y, E, H, FINAL_EPOCHS, rng)

    # Precompute T = embeddings @ W1 per position
    d = X.shape[1]
    W1_reshaped = W1.reshape(d, E, H)  # (d, E, H)
    T = np.matmul(emb, W1_reshaped)    # (d, 256, H)
    T = T.astype(np.float32)

    # Write weights.bin
    magic = 0x454D424D  # 'EMBM'
    C = 65
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D424D, int32 d, E, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;  // ReLU
        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            if (ah != 0.0f) {
                const float *w = W2 + (long)h * C;
                for (int c = 0; c < C; c++) sc[c] += ah * w[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```