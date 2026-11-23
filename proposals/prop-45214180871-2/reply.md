```hypothesis
We propose label-embmlp: per-position byte embeddings (E=16) followed by a ReLU hidden layer (H=64) and a linear output. This captures character similarities and contextual interactions better than one-hot linear models, which should improve accuracy on this character-level language modeling task. If dev accuracy does not exceed label-linear's 0.3375, it suggests the task is nearly linearly separable in one-hot space or the dataset is too small for embeddings to generalize.
```
```python
"""label-embmlp: per-position byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x454D4245, d, E, H, C; float32 Emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen after tuning on last 20% of train.in)
E = 16          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
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

def softmax_cross_entropy_grad(logits, y):
    """logits: (B, C), y: (B,) int64. Returns gradient wrt logits (B, C)."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def forward(X, Emb, W1, b1, W2, b2):
    """X: (B, d) uint8. Returns logits (B, C), and cache for backward."""
    B, d = X.shape
    # Embedding lookup and concatenation
    z = np.zeros((B, d * E), dtype=np.float32)
    for j in range(d):
        z[:, j*E:(j+1)*E] = Emb[j, X[:, j], :]
    if H:
        a = z @ W1 + b1
        h = np.maximum(a, 0, dtype=np.float32)
        logits = h @ W2 + b2
        cache = (z, a, h)
    else:
        logits = z @ W1 + b1  # W1 is (d*E, C), b1 is (C,)
        cache = (z,)
    return logits, cache

def backward(X, y, cache, Emb, W1, b1, W2, b2):
    """Returns gradients for all parameters."""
    B, d = X.shape
    if H:
        z, a, h = cache
        # grad logits
        dlogits = softmax_cross_entropy_grad(h @ W2 + b2, y)  # (B, C)
        # grad W2, b2
        dW2 = h.T @ dlogits + L2 * W2
        db2 = dlogits.sum(axis=0)
        # grad h
        dh = dlogits @ W2.T
        # grad a (ReLU)
        da = dh * (a > 0)
        # grad W1, b1
        dW1 = z.T @ da + L2 * W1
        db1 = da.sum(axis=0)
        # grad z
        dz = da @ W1.T
        # grad Emb
        dEmb = np.zeros_like(Emb)
        for j in range(d):
            idx = X[:, j]
            # accumulate gradient for each example
            for i in range(B):
                dEmb[j, idx[i]] += dz[i, j*E:(j+1)*E]
        return dEmb, dW1, db1, dW2, db2
    else:
        z = cache[0]
        dlogits = softmax_cross_entropy_grad(z @ W1 + b1, y)
        dW1 = z.T @ dlogits + L2 * W1
        db1 = dlogits.sum(axis=0)
        dz = dlogits @ W1.T
        dEmb = np.zeros_like(Emb)
        for j in range(d):
            idx = X[:, j]
            for i in range(B):
                dEmb[j, idx[i]] += dz[i, j*E:(j+1)*E]
        return dEmb, dW1, db1, None, None

def train_model(X, y, X_val, y_val, epochs, rng):
    """Train and return (params, val_acc)."""
    n, d = X.shape
    C = 65
    # Initialize parameters
    Emb = rng.uniform(-1, 1, (d, 256, E)).astype(np.float32) * math.sqrt(6.0 / (256 + E))
    if H:
        W1 = rng.uniform(-1, 1, (d*E, H)).astype(np.float32) * math.sqrt(6.0 / (d*E))
        b1 = np.zeros(H, np.float32)
        W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
        b2 = np.zeros(C, np.float32)
        params = [Emb, W1, b1, W2, b2]
        shapes = [Emb.shape, W1.shape, b1.shape, W2.shape, b2.shape]
    else:
        W1 = rng.uniform(-1, 1, (d*E, C)).astype(np.float32) * math.sqrt(6.0 / (d*E))
        b1 = np.zeros(C, np.float32)
        params = [Emb, W1, b1]
        shapes = [Emb.shape, W1.shape, b1.shape]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_val_acc = -1.0
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            logits, cache = forward(Xb, *params)
            grads = backward(Xb, yb, cache, *params)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                if gr is None:
                    continue
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        val_logits, _ = forward(X_val, *params)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
    return best_params, best_val_acc

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X_all)
    # Split: last 20% for tuning (validation)
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]
    rng = np.random.default_rng(SEED)
    # Fixed hyperparameters (could be tuned here, but we use fixed E, H)
    params, val_acc = train_model(X_train, y_train, X_val, y_val,
                                  epochs=min(MAX_EPOCHS, math.ceil(SAMPLES / split)), rng=rng)
    # Retrain on full data with best config (using same seed for reproducibility)
    rng = np.random.default_rng(SEED)
    final_params, _ = train_model(X_all, y_all, X_val, y_val,
                                  epochs=min(MAX_EPOCHS, math.ceil(SAMPLES / n)), rng=rng)
    Emb, W1, b1, W2, b2 = final_params
    # Write weights.bin
    d = card["input"]["bytes"]
    C = card["classes"]
    magic = 0x454D4245  # 'EMBE'
    out = struct.pack("<5i", magic, d, E, H, C)
    out += Emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index 0..64).
 * Weights format: magic 0x454D4245, d, E, H, C; float32 Emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20;
    expected += (long)d * 256 * E;           // Emb
    expected += (long)d * E * H;             // W1
    expected += H;                           // b1
    if (H) {
        expected += (long)H * C;             // W2
        expected += C;                       // b2
    }
    if (nw != 4 + expected * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Emb = (const float *)(hd + 5);
    const float *W1 = Emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d * E);
    float *h = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = Emb + ((long)j * 256 + xi[j]) * E;
            memcpy(z + j * E, e, E * sizeof(float));
        }
        if (H) {
            // Hidden layer: ReLU(z @ W1 + b1)
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int v = 0; v < d * E; v++) sum += z[v] * W1[(long)v * H + u];
                h[u] = sum > 0.0f ? sum : 0.0f;
            }
            // Output layer: h @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
                logits[c] = sum;
            }
        } else {
            // Linear: z @ W1 + b1 (W1 is d*E x C, b1 is C)
            for (int c = 0; c < C; c++) {
                float sum = b1[c];
                for (int v = 0; v < d * E; v++) sum += z[v] * W1[(long)v * C + c];
                logits[c] = sum;
            }
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```