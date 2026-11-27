```hypothesis
Family: label-embed-mlp. Learns per-position byte embeddings (size E), sums them, then passes through a ReLU hidden layer (size H) to output classes. Captures character similarities and non-linear interactions that linear model misses, while keeping parameters (16*256*E + E*H + H*C) below linear's 16*256*C. With E=32, H=128 we target >0.35 dev accuracy at ~15k Ir/item. Failure: accuracy ≤0.34 or cost >30k Ir/item.
```

```python
"""Embedding MLP for next-char prediction: per-position byte embeddings summed, one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only. Hyperparameters chosen on last 20% of train.in (tuning split)."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (could be tuned on validation split, but we fix for determinism and speed)
E = 32          # embedding dimension
H = 128         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 128
EPOCHS_FULL = 60
VAL_FRACTION = 0.2   # last 20% for tuning (as per card)

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
    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # Split: last 20% for validation (tuning split)
    val_start = int(n * (1 - VAL_FRACTION))
    X_train, y_train = X_bytes[:val_start], y[:val_start]
    X_val, y_val = X_bytes[val_start:], y[val_start:]
    n_train = X_train.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: 16 positions x 256 values x E
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Hidden layer: E x H
    W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: H x C
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch, training=True):
        """X_batch: (B, d) uint8. Returns logits (B, C) and hidden pre-activation for backprop."""
        B = X_batch.shape[0]
        # Embedding lookup and sum
        # emb_sum[b, e] = sum_j emb[j, X_batch[b, j], e]
        emb_sum = np.zeros((B, E), np.float32)
        for j in range(d):
            emb_sum += emb[j, X_batch[:, j]]
        # Hidden layer
        hidden_pre = emb_sum @ W1 + b1
        hidden = np.maximum(hidden_pre, 0)
        # Output
        logits = hidden @ W2 + b2
        return logits, hidden_pre, hidden, emb_sum

    def loss_and_grad(X_batch, y_batch):
        B = X_batch.shape[0]
        logits, hidden_pre, hidden, emb_sum = forward(X_batch)
        # Softmax cross-entropy
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Gradient on logits
        grad_logits = probs
        grad_logits[np.arange(B), y_batch] -= 1.0
        grad_logits /= B
        # Gradients
        grad_W2 = hidden.T @ grad_logits + L2 * W2
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ W2.T
        grad_hidden_pre = grad_hidden * (hidden_pre > 0)
        grad_W1 = emb_sum.T @ grad_hidden_pre + L2 * W1
        grad_b1 = grad_hidden_pre.sum(axis=0)
        # Gradient on emb_sum
        grad_emb_sum = grad_hidden_pre @ W1.T
        # Gradient on embeddings
        grad_emb = np.zeros_like(emb)
        for j in range(d):
            # For each position, scatter-add grad_emb_sum to grad_emb[j, X_batch[:, j]]
            np.add.at(grad_emb[j], X_batch[:, j], grad_emb_sum)
        grad_emb += L2 * emb
        return [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

    def evaluate(X, y):
        logits, _, _, _ = forward(X, training=False)
        preds = logits.argmax(axis=1)
        return (preds == y).mean()

    # Quick hyperparameter selection on validation split (optional, but we fix E,H)
    # We'll just train on full training set for EPOCHS_FULL epochs.
    print(f"Training on {n_train} samples, validating on {X_val.shape[0]}", file=sys.stderr)

    for epoch in range(EPOCHS_FULL):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X_train[idx], y_train[idx]
            grads = loss_and_grad(X_batch, y_batch)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        val_acc = evaluate(X_val, y_val)
        print(f"Epoch {epoch+1}/{EPOCHS_FULL}, val_acc={val_acc:.4f}", file=sys.stderr)

    # Final evaluation on full training set (optional)
    train_acc = evaluate(X_train, y_train)
    print(f"Final train_acc={train_acc:.4f}, val_acc={val_acc:.4f}", file=sys.stderr)

    # Pack weights for inference
    # Format: magic, d, E, H, C (all int32)
    # Then embeddings (d*256*E floats), W1 (E*H), b1 (H), W2 (H*C), b2 (C)
    magic = 0x454D424D  # 'EMBM' in little-endian
    out = struct.pack("<5i", magic, d, E, H, C)
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
/* Embedding MLP inference: ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    // Expected size: header(20) + emb(d*256*E) + W1(E*H) + b1(H) + W2(H*C) + b2(C) all float32
    long expected = 20 + 4L * ( (long)d * 256 * E + (long)E * H + H + (long)H * C + C );
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden_pre = malloc(sizeof(float) * H);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding sum
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *emb_j = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += emb_j[e];
        }
        // Hidden layer
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
            hidden_pre[h] = sum;
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```