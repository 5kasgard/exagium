```hypothesis
Family: label-embmlp (embedding MLP with one hidden layer). Replaces one-hot encoding with learned per-feature embeddings (size E) followed by a ReLU hidden layer (size H) and a linear output. On Adult's mix of ordinal and 255-bin categorical features, embeddings should capture interactions more efficiently than one-hot, beating label-linear's 0.8524 accuracy while keeping inference cost near label-mlp16 (≈2300 Ir/item). Failure: dev accuracy ≤ 0.855.
```

```python
"""label-embmlp: embedding MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then trained on all train.
Weights format (little-endian):
  magic(0x454D424D), d, E, H, C (int32 each)
  embeddings[d][256][E] (float32)
  W2[E][H], b2[H], W3[H][C], b3[C] (float32)
"""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning split
GRID = [
    {"E": 8, "H": 16, "LR": 1e-3, "L2": 1e-4},
    {"E": 8, "H": 32, "LR": 1e-3, "L2": 1e-4},
    {"E": 16, "H": 16, "LR": 3e-3, "L2": 1e-4},
    {"E": 16, "H": 32, "LR": 3e-3, "L2": 1e-4},
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, X_val, y_val, cfg, rng, epochs):
    n, d = X.shape
    C = 2
    E, H, LR, L2 = cfg["E"], cfg["H"], cfg["LR"], cfg["L2"]
    # Initialize embeddings (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    W2 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    b2 = np.zeros(H, np.float32)
    W3 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [emb, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # Embedding lookup and sum
            # xb: (B, d) uint8
            # emb: (d, 256, E) -> gather -> (B, d, E) -> sum over d -> (B, E)
            # Efficient gather: use advanced indexing
            B = xb.shape[0]
            emb_sum = np.zeros((B, E), np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]
            # Forward
            h = emb_sum @ W2 + b2
            h = np.maximum(h, 0)
            logits = h @ W3 + b3
            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)
            # Backprop
            grad_W3 = h.T @ grad_logits + L2 * W3
            grad_b3 = grad_logits.sum(0)
            grad_h = grad_logits @ W3.T
            grad_h[h <= 0] = 0.0
            grad_W2 = emb_sum.T @ grad_h + L2 * W2
            grad_b2 = grad_h.sum(0)
            grad_emb_sum = grad_h @ W2.T
            # Embedding gradients: scatter-add
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W2, grad_b2, grad_W3, grad_b3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Validation accuracy
    B = X_val.shape[0]
    emb_sum = np.zeros((B, E), np.float32)
    for j in range(d):
        emb_sum += emb[j, X_val[:, j]]
    h = np.maximum(emb_sum @ W2 + b2, 0)
    logits = h @ W3 + b3
    pred = logits.argmax(1)
    acc = (pred == y_val).mean()
    return acc, params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
    best_acc, best_cfg, best_params = -1, None, None
    for cfg in GRID:
        # Use a deterministic sub-seed for each config
        sub_rng = np.random.default_rng(SEED + hash(tuple(sorted(cfg.items()))) % 2**32)
        acc, params = train_model(X_train, y_train, X_val, y_val, cfg, sub_rng, epochs)
        if acc > best_acc:
            best_acc, best_cfg, best_params = acc, cfg, params
    # Retrain on full data with best config
    rng_full = np.random.default_rng(SEED)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    _, final_params = train_model(X, y, X[:1], y[:1], best_cfg, rng_full, epochs_full)
    emb, W2, b2, W3, b3 = final_params
    # Write weights
    E, H = best_cfg["E"], best_cfg["H"]
    magic = 0x454D424D  # 'EMBM'
    hdr = struct.pack("<5i", magic, d, E, H, C)
    # Embeddings: (d, 256, E) -> flatten in order d, 256, E
    emb_flat = emb.transpose(0, 1, 2).reshape(-1).astype("<f4")
    W2_flat = W2.astype("<f4").ravel()
    b2_flat = b2.astype("<f4").ravel()
    W3_flat = W3.astype("<f4").ravel()
    b3_flat = b3.astype("<f4").ravel()
    out = hdr + emb_flat.tobytes() + W2_flat.tobytes() + b2_flat.tobytes() + W3_flat.tobytes() + b3_flat.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1).
 * Weights format (little-endian):
 *   uint32 magic (0x454D424D), int32 d, E, H, C
 *   float32 embeddings[d][256][E]
 *   float32 W2[E][H], b2[H], W3[H][C], b3[C]
 */
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
    if (C < 1 || C > 256 || d < 1 || E < 1 || H < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W2 = emb + (long)d * 256 * E;
    const float *b2 = W2 + (long)E * H;
    const float *W3 = b2 + H;
    const float *b3 = W3 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = (float *)malloc(sizeof(float) * E);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(emb_sum, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) emb_sum[e_idx] += e[e_idx];
        }
        for (int h = 0; h < H; h++) {
            float sum = b2[h];
            for (int e_idx = 0; e_idx < E; e_idx++) sum += emb_sum[e_idx] * W2[(long)e_idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b3[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W3[(long)h * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```