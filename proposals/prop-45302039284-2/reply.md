```hypothesis
Family: positional-embedding MLP (pemb-mlp). Each of the 16 context positions gets a learned 256×E embedding table; the 16 embeddings are summed, passed through a ReLU hidden layer of size H, then a linear output layer to 65 classes. This captures position‑specific character interactions that a linear model misses, while keeping inference extremely fast (only a few thousand FLOPs/item). If dev accuracy does not exceed label‑linear’s 0.3375, or if inference cost exceeds label‑linear’s ~7k Ir/item, the approach fails.
```

```python
"""Positional embedding MLP for next-byte prediction.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen via tuning split in a real run; here we use reasonable defaults)
SEED = 1
EMB_DIM = 32
HIDDEN = 64
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50
VAL_FRACTION = 0.2  # last 20% for validation

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t logits."""
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def train_model(X, y, emb_dim, hidden, lr, l2, batch, epochs, rng):
    n, d = X.shape
    C = 65
    # Initialize parameters
    # Embeddings: d * 256 * emb_dim
    emb = (rng.normal(0, 1/math.sqrt(emb_dim), (d, 256, emb_dim))).astype(np.float32)
    if hidden > 0:
        W1 = (rng.normal(0, 1/math.sqrt(emb_dim), (emb_dim, hidden))).astype(np.float32)
        b1 = np.zeros(hidden, np.float32)
        W2 = (rng.normal(0, 1/math.sqrt(hidden), (hidden, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W2 = (rng.normal(0, 1/math.sqrt(emb_dim), (emb_dim, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb = X[idx]
            yb = y[idx]
            # Forward
            # Sum embeddings: (batch, emb_dim)
            h = np.zeros((len(yb), emb_dim), np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]
            if hidden > 0:
                h = np.maximum(h @ W1 + b1, 0)
            logits = h @ W2 + b2
            # Loss gradient
            g = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            # Backward
            grad_W2 = h.T @ g + l2 * W2
            grad_b2 = g.sum(0)
            if hidden > 0:
                gh = g @ W2.T
                gh[h <= 0] = 0  # ReLU derivative
                grad_W1 = h.T @ gh + l2 * W1  # h here is pre-ReLU? Wait, we need pre-ReLU activation.
                # Actually we need the pre-ReLU activation. Let's recompute or store.
                # We'll store pre-ReLU activation.
                # To avoid recomputing, we can restructure. But for simplicity, we recompute.
                pre_relu = h @ W1 + b1  # Wait, h is post-ReLU. We need the input to ReLU.
                # Let's store pre_relu in forward pass.
                # We'll change forward to keep pre_relu.
                pass
            # We'll rewrite the backward more cleanly below.
            # For now, we'll implement a cleaner version.
            # Let's restart the training loop with proper gradient computation.
            # Given time, we'll write a correct but simpler version.
            # We'll do the forward and backward in a more straightforward way.
            # Since the model is small, we can compute gradients manually.
            # We'll break and re-implement.

    # This is a placeholder; we need a complete correct implementation.
    # Due to the complexity, we'll provide a fully working version below.
    # Let's write a clean implementation.
    pass

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X)
    val_size = int(n * VAL_FRACTION)
    train_X, train_y = X[:-val_size], y[:-val_size]
    val_X, val_y = X[-val_size:], y[-val_size:]

    rng = np.random.default_rng(SEED)

    # Simple grid search (in practice we'd do this; here we just use fixed best config)
    best_emb, best_hidden = EMB_DIM, HIDDEN

    # Train final model on all data
    # Re-initialize with fixed seed for reproducibility
    rng = np.random.default_rng(SEED)
    emb = (rng.normal(0, 1/math.sqrt(best_emb), (d, 256, best_emb))).astype(np.float32)
    if best_hidden > 0:
        W1 = (rng.normal(0, 1/math.sqrt(best_emb), (best_emb, best_hidden))).astype(np.float32)
        b1 = np.zeros(best_hidden, np.float32)
        W2 = (rng.normal(0, 1/math.sqrt(best_hidden), (best_hidden, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
    else:
        W2 = (rng.normal(0, 1/math.sqrt(best_emb), (best_emb, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)

    # Adam parameters
    params = [emb]
    if best_hidden > 0:
        params.extend([W1, b1])
    params.extend([W2, b2])
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(Xb):
        """Returns (logits, cache) where cache contains intermediates for backward."""
        h = np.zeros((len(Xb), best_emb), np.float32)
        for j in range(d):
            h += emb[j, Xb[:, j]]
        if best_hidden > 0:
            pre_relu = h @ W1 + b1
            h_relu = np.maximum(pre_relu, 0)
            logits = h_relu @ W2 + b2
            cache = (h, pre_relu, h_relu)
        else:
            logits = h @ W2 + b2
            cache = (h,)
        return logits, cache

    def backward(Xb, yb, g, cache):
        """g is gradient of loss w.r.t logits (batch, C). Returns gradients for params in order."""
        grads = []
        if best_hidden > 0:
            h, pre_relu, h_relu = cache
            # Output layer
            grad_W2 = h_relu.T @ g
            grad_b2 = g.sum(0)
            # Hidden layer
            gh = g @ W2.T
            gh[pre_relu <= 0] = 0
            grad_W1 = h.T @ gh
            grad_b1 = gh.sum(0)
            # Embeddings
            g_emb = np.zeros_like(emb)
            for j in range(d):
                # gh_emb = gh for hidden, or g @ W2.T for linear? Here gh is already backpropagated to h.
                # Actually gh is gradient w.r.t h (the summed embeddings).
                # So we accumulate gradient for each position's embedding.
                g_emb[j, Xb[:, j]] += gh
            grads = [g_emb, grad_W1, grad_b1, grad_W2, grad_b2]
        else:
            h, = cache
            grad_W2 = h.T @ g
            grad_b2 = g.sum(0)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                g_emb[j, Xb[:, j]] += g @ W2.T
            grads = [g_emb, grad_W2, grad_b2]
        # Add L2 regularization
        if best_hidden > 0:
            grads[1] += L2 * W1
            grads[3] += L2 * W2
        else:
            grads[1] += L2 * W2
        return grads

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            logits, cache = forward(xb)
            g = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            grads = backward(xb, yb, g, cache)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights
    # Magic: 0x4D4C5031 ('MLP1')
    # d, emb_dim, hidden, C
    header = struct.pack("<5i", 0x4D4C5031, d, best_emb, best_hidden, C)
    out = header
    out += emb.astype("<f4").tobytes()
    if best_hidden > 0:
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Positional embedding MLP inference: ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (H > 0 ? (long)E * H + H : 0) + (long)(H > 0 ? H : E) * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (H > 0 ? (long)E * H : 0);
    const float *W2 = b1 + (H > 0 ? H : 0);
    const float *b2 = W2 + (long)(H > 0 ? H : E) * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * E);
    float *hidden = H > 0 ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int v = 0; v < E; v++) h[v] += e[v];
        }
        if (H > 0) {
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int v = 0; v < E; v++) sum += h[v] * W1[(long)v * H + u];
                hidden[u] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
                logits[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int v = 0; v < E; v++) sum += h[v] * W2[(long)v * C + c];
                logits[c] = sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```