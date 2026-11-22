```hypothesis
Family: label-emb-mlp (position-specific embeddings + one hidden ReLU layer).
Why: The 16-byte context is a sequence of categorical tokens. Label-linear uses one-hot (4096 features) which is high-dimensional and sparse. Dense embeddings per position (16*256*E) capture character similarities and reduce parameters. A hidden layer (size H) models interactions between positions. With E=8, H=64 the model has ~45k parameters (vs 266k for label-linear), should generalize better on 24k examples, and inference cost stays low (lookup + two matvecs).
Failure: dev accuracy ≤ 0.3375 (label-linear) or inference cost (Ir/item) exceeds label-linear.
```

```python
"""label-emb-mlp: position-specific embeddings + one hidden ReLU layer for next-char prediction.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
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


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Split: last 20% for validation (tuning), first 80% for training
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Initialize embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Hidden layer: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Embedding lookup: (B, d, E)
        e = emb[np.arange(d), X_batch.T].transpose(1, 0, 2)  # (B, d, E)
        flat = e.reshape(B, d * E)                           # (B, d*E)
        h = flat @ W1 + b1                                   # (B, H)
        h = np.maximum(h, 0)                                 # ReLU
        logits = h @ W2 + b2                                 # (B, C)
        return logits, h, flat

    def loss_grad(logits, y_batch):
        # Cross-entropy gradient
        B = logits.shape[0]
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        return probs / B  # (B, C)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    best_val_acc = -1.0
    best_params = None

    for epoch in range(max_epochs):
        order = rng.permutation(len(X_train))
        for s in range(0, len(X_train), BATCH):
            idx_batch = order[s:s + BATCH]
            xb, yb = X_train[idx_batch], y_train[idx_batch]
            logits, h, flat = forward(xb)
            g_out = loss_grad(logits, yb)                     # (B, C)

            # Gradients
            g_W2 = h.T @ g_out + L2 * W2                      # (H, C)
            g_b2 = g_out.sum(axis=0)                          # (C,)
            g_h = g_out @ W2.T                                # (B, H)
            g_h[h <= 0] = 0                                   # ReLU grad
            g_W1 = flat.T @ g_h + L2 * W1                     # (d*E, H)
            g_b1 = g_h.sum(axis=0)                            # (H,)
            g_flat = g_h @ W1.T                               # (B, d*E)
            g_emb = np.zeros_like(emb)                        # (d, 256, E)
            g_flat_3d = g_flat.reshape(B, d, E)               # (B, d, E)
            for j in range(d):
                np.add.at(g_emb[j], xb[:, j], g_flat_3d[:, j])

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation accuracy
        val_logits, _, _ = forward(X_val)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]

    # Retrain on full data with best hyperparameters (same epochs)
    # We simply continue training on full data from the best checkpoint
    params = best_params
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            xb, yb = X[idx_batch], y[idx_batch]
            logits, h, flat = forward(xb)
            g_out = loss_grad(logits, yb)

            g_W2 = h.T @ g_out + L2 * params[3]
            g_b2 = g_out.sum(axis=0)
            g_h = g_out @ params[3].T
            g_h[h <= 0] = 0
            g_W1 = flat.T @ g_h + L2 * params[1]
            g_b1 = g_h.sum(axis=0)
            g_flat = g_h @ params[1].T
            g_emb = np.zeros_like(params[0])
            g_flat_3d = g_flat.reshape(BATCH, d, E)
            for j in range(d):
                np.add.at(g_emb[j], xb[:, j], g_flat_3d[:, j])

            grads = [g_emb, g_W1, g_b1, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, H, C; then emb, W1, b1, W2, b2 (all float32 little-endian)
    emb, W1, b1, W2, b2 = params
    magic = 0x454D424C  # "LMBE"
    header = struct.pack("<5i", magic, d, E, H, C)
    body = (
        emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and flatten
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int v = 0; v < E; v++) sum += e[v] * W1[((long)j * E + v) * H + u];
                hidden[u] += sum;
            }
        }
        // ReLU
        for (int u = 0; u < H; u++) if (hidden[u] < 0.0f) hidden[u] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = hidden[u];
            for (int c = 0; c < C; c++) logits[c] += hu * W2[(long)u * C + c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```