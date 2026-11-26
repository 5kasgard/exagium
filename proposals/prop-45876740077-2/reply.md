```hypothesis
Family: label-embed-mlp — shared token embeddings + positional embeddings + one ReLU hidden layer.
Why: The 16-byte context benefits from positional information; label-linear uses 16×256×C ≈ 266k parameters (one-hot per position). Shared embeddings (256×E) plus learned positional embeddings (16×E) drastically reduce parameters (e.g., E=32, H=128 → ~82k) while retaining positional sensitivity. A small MLP on top captures non-linear interactions. This should exceed label-linear's 0.3375 accuracy at comparable or lower inference cost (fewer multiply-adds).
Failure: If dev accuracy ≤ 0.3375 or instructions/item ≥ 7359 (label-linear's cost), the hypothesis is not supported.
```

```python
"""label-embed-mlp: shared token embeddings + positional embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters (E, H) chosen on last 20% of train.in (tuning split), then retrained on all train.
Exported weights: magic 0x454D4244, d, E, H, C; then float32 arrays:
  W_emb[256*E], W_pos[d*E], W1[d*E*H], b1[H], W2[H*C], b2[C] (all little-endian)."""
import json, math, struct, sys

import numpy as np

# Fixed training hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Grid for tuning split (last 20% of train.in)
E_GRID = [8, 16, 32]
H_GRID = [32, 64, 128]
TUNE_EPOCHS = 20  # quick evaluation on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, C, d, E, H, epochs, rng, verbose=False):
    n = X.shape[0]
    # Initialize parameters
    W_emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W_pos = (rng.uniform(-1, 1, (d, E)) * math.sqrt(6.0 / d)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W_emb, W_pos, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute positional embeddings added to token embeddings? We'll add during forward.
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Embeddings: (B, d, E)
            emb = W_emb[xb] + W_pos[None, :, :]   # broadcasting
            flat = emb.reshape(B, d * E)          # (B, d*E)
            hidden_pre = flat @ W1 + b1           # (B, H)
            hidden = np.maximum(hidden_pre, 0)    # ReLU
            logits = hidden @ W2 + b2             # (B, C)

            # Softmax + cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # Backprop
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden_pre <= 0] = 0.0
            grad_W1 = flat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            # Gradients for embeddings
            grad_flat = grad_hidden @ W1.T   # (B, d*E)
            grad_emb = grad_flat.reshape(B, d, E)
            # W_emb gradient: scatter-add over batch and positions
            grad_W_emb = np.zeros_like(W_emb)
            np.add.at(grad_W_emb, xb.ravel(), grad_emb.reshape(-1, E))
            grad_W_emb += L2 * W_emb
            # W_pos gradient: sum over batch
            grad_W_pos = grad_emb.sum(axis=0) + L2 * W_pos

            grads = [grad_W_emb, grad_W_pos, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return params

def evaluate(X, y, C, d, E, H, params):
    W_emb, W_pos, W1, b1, W2, b2 = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        xb = X[s:s+1024]
        yb = y[s:s+1024]
        B = xb.shape[0]
        emb = W_emb[xb] + W_pos[None, :, :]
        flat = emb.reshape(B, d * E)
        hidden = np.maximum(flat @ W1 + b1, 0)
        logits = hidden @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_E = best_H = None

    # Grid search on tuning split
    for E in E_GRID:
        for H in H_GRID:
            # Use a fresh RNG for each config to keep deterministic
            config_rng = np.random.default_rng(SEED + E * 1000 + H)
            params = train_model(X_train, y_train, C, d, E, H, TUNE_EPOCHS, config_rng)
            acc = evaluate(X_val, y_val, C, d, E, H, params)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_E, best_H = E, H

    # Retrain best config on full dataset
    full_rng = np.random.default_rng(SEED)
    full_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params = train_model(X, y, C, d, best_E, best_H, full_epochs, full_rng)

    # Export weights
    W_emb, W_pos, W1, b1, W2, b2 = final_params
    magic = 0x454D4244  # "EMBD"
    header = struct.pack("<5i", magic, d, best_E, best_H, C)
    # All arrays as float32 little-endian
    out = header
    out += W_emb.astype("<f4").tobytes()
    out += W_pos.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index 0..C-1).
 * Weights format: magic(0x454D4244), d, E, H, C (int32 each), then float32 arrays:
 *   W_emb[256*E], W_pos[d*E], W1[d*E*H], b1[H], W2[H*C], b2[C] */
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
    if (d != 16 || C < 1 || C > 256 || E < 1 || H < 1) { fprintf(stderr, "bad header values\n"); return 3; }

    // Expected weight size in bytes
    long expected = 20 + 4L * (256L * E + (long)d * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *W_emb = (const float *)(hd + 5);
    const float *W_pos = W_emb + 256L * E;
    const float *W1 = W_pos + (long)d * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Temporary buffers
    float *flat = (float *)malloc(sizeof(float) * d * E);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    if (!flat || !hidden || !logits) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build flat embedding: for each position j, add token emb + positional emb
        for (int j = 0; j < d; j++) {
            const float *tok = W_emb + (long)xi[j] * E;
            const float *pos_emb = W_pos + (long)j * E;
            float *dst = flat + (long)j * E;
            for (int e = 0; e < E; e++) dst[e] = tok[e] + pos_emb[e];
        }

        // Hidden layer: ReLU(flat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + (long)h;  // W1 is (d*E) x H, stored row-major? In train.py: W1 shape (d*E, H). So row-major: W1[flat_idx * H + h]? Wait: In train.py, flat @ W1 where flat is (B, d*E) and W1 is (d*E, H). So W1 is column-major for output? Actually flat @ W1: flat (B, dE) dot W1 (dE, H) -> (B, H). So W1 is stored row-major: W1[flat_idx * H + h] is the weight from flat_idx to h. But in our C code we have W1 as a flat array from train.py's W1.astype("<f4").tobytes(). In train.py, W1 is shape (d*E, H) and we do .tobytes() which is row-major (C order). So element (i, j) is at index i*H + j. So we need to access W1[i*H + h].
            for (int idx = 0; idx < d * E; idx++) {
                sum += flat[idx] * W1[(long)idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```