```hypothesis
We propose an "embedding-linear" family: learn a small embedding vector (size E) for each byte at each of the 16 positions, concatenate them, and apply a linear classifier. This factorizes the full one-hot weight matrix (16*256*65 ≈ 266K parameters) into embeddings (16*256*E) plus output weights (16*E*65). For E=8 this is ~41K parameters, reducing overfitting on 24K training items and lowering inference cost. The model is trained with Adam on cross-entropy, hyperparameters (E, lr) chosen on the last 20% tuning split. If dev accuracy does not exceed label-linear (0.3375) or inference cost (Ir/item) is not lower, the hypothesis fails.
```

```python
"""Embedding-linear family for fixed-size byte sequences.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Learns per-position byte embeddings (size E) and a linear output layer.
Hyperparameters (E, lr) selected on last 20% of train.in (tuning split),
then retrained on all data. Exported as flat float32 arrays."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
MAX_EPOCHS_TUNE = 30
MAX_EPOCHS_FULL = 100
SAMPLES_PER_EPOCH = 600000  # for epoch count heuristic
# Hyperparameter grid for tuning
E_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2 = 1e-4

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t logits (already averaged over batch)."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def train_step(emb, W_out, b_out, x_batch, y_batch, lr, l2, opt_state, step):
    """One Adam step. emb: list of 16 arrays (256, E). W_out: (d*E, C). b_out: (C,).
       Returns updated parameters and new opt_state."""
    d = len(emb)
    E = emb[0].shape[1]
    C = W_out.shape[1]
    n = x_batch.shape[0]
    # Gather embeddings for batch: (n, d*E)
    x_emb = np.zeros((n, d * E), dtype=np.float32)
    for j in range(d):
        x_emb[:, j*E:(j+1)*E] = emb[j][x_batch[:, j]]
    # Forward
    logits = x_emb @ W_out + b_out  # (n, C)
    # Loss gradient
    grad_logits = softmax_cross_entropy_grad(logits, y_batch)  # (n, C)
    # Gradients
    grad_W_out = x_emb.T @ grad_logits + l2 * W_out  # (d*E, C)
    grad_b_out = grad_logits.sum(axis=0)             # (C,)
    grad_x_emb = grad_logits @ W_out.T                # (n, d*E)
    # Gradient w.r.t embeddings
    grad_emb = [np.zeros_like(e) for e in emb]
    for j in range(d):
        np.add.at(grad_emb[j], x_batch[:, j], grad_x_emb[:, j*E:(j+1)*E])
    # Adam update
    params = [W_out, b_out] + grad_emb  # note: grad_emb are gradients, not params
    # We'll update W_out, b_out, and each emb[j] separately
    # opt_state is list of (m, v) for each param in order: W_out, b_out, emb[0]...emb[15]
    t = step
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    # Update W_out
    m, v = opt_state[0]
    m = beta1 * m + (1 - beta1) * grad_W_out
    v = beta2 * v + (1 - beta2) * (grad_W_out * grad_W_out)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    W_out -= lr * m_hat / (np.sqrt(v_hat) + eps)
    opt_state[0] = (m, v)
    # Update b_out
    m, v = opt_state[1]
    m = beta1 * m + (1 - beta1) * grad_b_out
    v = beta2 * v + (1 - beta2) * (grad_b_out * grad_b_out)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    b_out -= lr * m_hat / (np.sqrt(v_hat) + eps)
    opt_state[1] = (m, v)
    # Update embeddings
    for j in range(d):
        m, v = opt_state[2 + j]
        g = grad_emb[j]
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        emb[j] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        opt_state[2 + j] = (m, v)
    return emb, W_out, b_out, opt_state

def evaluate(emb, W_out, b_out, X, y):
    d = len(emb)
    E = emb[0].shape[1]
    n = X.shape[0]
    x_emb = np.zeros((n, d * E), dtype=np.float32)
    for j in range(d):
        x_emb[:, j*E:(j+1)*E] = emb[j][X[:, j]]
    logits = x_emb @ W_out + b_out
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]
    # Split: last 20% for tuning validation
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_config = None
    best_params = None
    # Hyperparameter search
    for E in E_CHOICES:
        for lr in LR_CHOICES:
            # Initialize
            emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
            W_out = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
            b_out = np.zeros(C, dtype=np.float32)
            # Adam state: (m, v) for each param
            opt_state = [(np.zeros_like(W_out), np.zeros_like(W_out)),
                         (np.zeros_like(b_out), np.zeros_like(b_out))]
            for _ in range(d):
                opt_state.append((np.zeros_like(emb[0]), np.zeros_like(emb[0])))
            # Training loop
            steps_per_epoch = max(1, math.ceil(len(X_train) / BATCH))
            max_epochs = min(MAX_EPOCHS_TUNE, math.ceil(SAMPLES_PER_EPOCH / len(X_train)))
            step = 0
            for epoch in range(max_epochs):
                order = rng.permutation(len(X_train))
                for s in range(0, len(X_train), BATCH):
                    idx = order[s:s+BATCH]
                    step += 1
                    emb, W_out, b_out, opt_state = train_step(
                        emb, W_out, b_out, X_train[idx], y_train[idx], lr, L2, opt_state, step)
            # Evaluate
            acc = evaluate(emb, W_out, b_out, X_val, y_val)
            if acc > best_acc:
                best_acc = acc
                best_config = (E, lr)
                # Save parameters (deep copy)
                best_params = ([e.copy() for e in emb], W_out.copy(), b_out.copy())
    print(f"Tuning best: E={best_config[0]}, lr={best_config[1]}, val_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data with best config
    E, lr = best_config
    emb, W_out, b_out = best_params
    # Re-initialize Adam state for full training
    opt_state = [(np.zeros_like(W_out), np.zeros_like(W_out)),
                 (np.zeros_like(b_out), np.zeros_like(b_out))]
    for _ in range(d):
        opt_state.append((np.zeros_like(emb[0]), np.zeros_like(emb[0])))
    steps_per_epoch = max(1, math.ceil(n / BATCH))
    max_epochs = min(MAX_EPOCHS_FULL, math.ceil(SAMPLES_PER_EPOCH / n))
    step = 0
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            step += 1
            emb, W_out, b_out, opt_state = train_step(
                emb, W_out, b_out, X[idx], y[idx], lr, L2, opt_state, step)
    # Final evaluation on full train (just for info)
    train_acc = evaluate(emb, W_out, b_out, X, y)
    print(f"Final train acc: {train_acc:.4f}", file=sys.stderr)
    # Save weights
    # Format: magic, d, E, C, H(=0), then embeddings[d][256][E], W_out[d*E][C], b_out[C]
    magic = 0x454D4244  # "EMBD" little-endian
    H = 0
    out = struct.pack("<5i", magic, d, E, C, H)
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Embedding-linear inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=16). Output is one byte: class index (0..C-1).
 * Weights format: int32 magic(0x454D4244), d, E, C, H(=0);
 * float32 emb[d][256][E], W_out[d*E][C], b_out[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3], H = hd[4];
    if (H != 0) { fprintf(stderr, "hidden layer not supported\n"); return 3; }
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)d * E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Gather embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(x_emb + j * E, e, E * sizeof(float));
        }
        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int idx = 0; idx < d * E; idx++) {
            float v = x_emb[idx];
            if (v != 0.0f) {
                const float *w = W_out + (long)idx * C;
                for (int c = 0; c < C; c++) scores[c] += v * w[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```