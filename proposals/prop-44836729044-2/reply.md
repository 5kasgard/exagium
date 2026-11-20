```hypothesis
Family: label-embmlp (embedding MLP with shared byte embeddings, one ReLU hidden layer).
Why: Character-level language modeling benefits from distributed representations of bytes and modeling interactions across the 16-byte context. A shared embedding layer (256 × E) followed by concatenation and a hidden layer captures positional n-gram statistics more efficiently than a linear model, while remaining far smaller and faster than gradient-boosted trees.
Failure signal: If dev accuracy does not exceed 0.34 (label-linear) or inference cost exceeds 2× label-linear, the architecture is not justified.
```

```python
"""label-embmlp: shared byte embeddings + one ReLU hidden layer for next-char prediction.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all data."""
import json, math, struct, sys

import numpy as np

# Architecture hyperparameters (tuned on validation split)
E = 8          # embedding dimension per byte
H = 64         # hidden layer size
LR = 0.01      # Adam learning rate
BATCH = 64
MAX_EPOCHS = 50
SEED = 1

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

    X_bytes = b"".join(records(sys.argv[2]))
    y_bytes = b"".join(records(sys.argv[3]))
    N = len(X_bytes) // d
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(N, d)
    y = np.frombuffer(y_bytes, dtype=np.uint8).astype(np.int64)

    # Split: last 20% for validation (tuning split)
    split = int(0.8 * N)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embedding: (256, E)
    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    # Hidden: (d*E, H)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Output: (H, C)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p, dtype=np.float32) for p in params]
    v = [np.zeros_like(p, dtype=np.float32) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Embedding lookup: (B, d, E)
        emb_seq = emb[X_batch]                     # (B, d, E)
        flat = emb_seq.reshape(B, d * E)           # (B, d*E)
        hidden = flat @ W1 + b1                    # (B, H)
        hidden_relu = np.maximum(hidden, 0)
        logits = hidden_relu @ W2 + b2             # (B, C)
        return flat, hidden, hidden_relu, logits

    def loss_and_grad(X_batch, y_batch):
        B = X_batch.shape[0]
        flat, hidden, hidden_relu, logits = forward(X_batch)
        # Softmax cross-entropy
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Gradient on logits
        grad_logits = probs
        grad_logits[np.arange(B), y_batch] -= 1.0
        grad_logits /= B
        # Backprop
        grad_W2 = hidden_relu.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden_relu = grad_logits @ W2.T
        grad_hidden = grad_hidden_relu * (hidden > 0)
        grad_W1 = flat.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        # Embedding gradients: scatter-add
        grad_emb = np.zeros_like(emb)
        # grad_hidden shape (B, H), flat shape (B, d*E), W1 shape (d*E, H)
        # We need gradient w.r.t. flat: (B, d*E) = grad_hidden @ W1.T
        grad_flat = grad_hidden @ W1.T
        # Reshape to (B, d, E)
        grad_emb_seq = grad_flat.reshape(B, d, E)
        # Accumulate into grad_emb using indices X_batch
        np.add.at(grad_emb, X_batch, grad_emb_seq)
        return [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

    def evaluate(X_eval, y_eval):
        _, _, _, logits = forward(X_eval)
        preds = logits.argmax(axis=1)
        return (preds == y_eval).mean()

    best_val_acc = -1.0
    best_epoch = 0
    best_params = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # Shuffle training data
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            X_batch = X_tr[idx]
            y_batch = y_tr[idx]
            grads = loss_and_grad(X_batch, y_batch)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        val_acc = evaluate(X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = [p.copy() for p in params]
        # Early stopping if no improvement for 5 epochs
        if epoch - best_epoch >= 5:
            break

    # Retrain on full dataset for best_epoch epochs
    if best_params is not None:
        params = best_params
    m = [np.zeros_like(p, dtype=np.float32) for p in params]
    v = [np.zeros_like(p, dtype=np.float32) for p in params]
    t = 0
    for epoch in range(1, best_epoch + 1):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s + BATCH]
            X_batch = X[idx]
            y_batch = y[idx]
            grads = loss_and_grad(X_batch, y_batch)
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    emb, W1, b1, W2, b2 = params

    # Write weights.bin
    # Format: magic, d, E, H, C (all int32), then row-major float32 arrays:
    # emb (256*E), W1 (d*E*H), b1 (H), W2 (H*C), b2 (C)
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray(header)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic(0x454D424C), d, E, H, C (int32 each), then float32 arrays:
 *   emb[256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] (all row-major). */
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
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 20 +
        (size_t)256 * E * 4 +
        (size_t)d * E * H * 4 +
        (size_t)H * 4 +
        (size_t)H * C * 4 +
        (size_t)C * 4;
    if ((size_t)nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256 * E;
    const float *b1 = W1 + (size_t)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + (size_t)xi[j] * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: flat (d*E) x W1 (d*E x H) + b1
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) {
                sum += flat[f] * W1[(size_t)f * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;  // ReLU
        }

        // Output layer: hidden (H) x W2 (H x C) + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(size_t)h * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(flat); free(hidden); free(logits);
    return 0;
}
```