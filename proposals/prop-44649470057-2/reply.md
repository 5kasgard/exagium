```hypothesis
Family: label-embed. Uses per-position byte embeddings (E=8) concatenated, then a ReLU hidden layer (H=64), then linear to 65 classes. Captures positional interactions better than linear one-hot while keeping parameters small (~45k). If dev accuracy ≤ label-linear (0.3375) or inference cost exceeds 2× label-linear, the approach fails.
```

```python
"""label-embed: per-position embeddings + one hidden ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid to search on tuning split (last 20% of train.in)
GRID = [
    {"E": 8, "H": 64},
    {"E": 8, "H": 128},
    {"E": 16, "H": 64},
    {"E": 16, "H": 128},
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoders(X, types):
    """Return list of (unique_vals, encoder_matrix) per feature.
    encoder_matrix shape (256, num_unique) with one-hot for seen values, zeros otherwise."""
    encoders = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc = ((np.arange(256) - mu) / sd)[:, None]
            encoders.append((None, enc.astype(np.float32)))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            encoders.append((vals, E))
    return encoders

def encode_batch(X, encoders):
    """X: (n, d) uint8 -> concatenated one-hot/ordinal float32 (n, total_dim)."""
    parts = []
    for j, (vals, E) in enumerate(encoders):
        parts.append(E[X[:, j]])
    return np.concatenate(parts, axis=1)

def init_params(rng, input_dim, H, C):
    if H:
        W1 = (rng.uniform(-1, 1, (input_dim, H)) * math.sqrt(6.0 / input_dim)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return [W1, b1, W2, b2]
    else:
        W1 = np.zeros((input_dim, C), np.float32)
        b1 = np.zeros(C, np.float32)
        return [W1, b1]

def softmax_grad(logits, y):
    """logits (n, C), y (n,) -> gradient (n, C) for cross-entropy."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def forward(params, Z, H):
    if H:
        a = Z @ params[0] + params[1]
        h = np.maximum(a, 0)
        logits = h @ params[2] + params[3]
        return logits, (Z, a, h)
    else:
        logits = Z @ params[0] + params[1]
        return logits, (Z,)

def backward(params, cache, grad_out, H, L2):
    if H:
        Z, a, h = cache
        # grad_out: (n, C)
        gW2 = h.T @ grad_out + L2 * params[2]
        gb2 = grad_out.sum(0)
        gh = grad_out @ params[2].T
        gh *= (a > 0)
        gW1 = Z.T @ gh + L2 * params[0]
        gb1 = gh.sum(0)
        return [gW1, gb1, gW2, gb2]
    else:
        Z = cache[0]
        gW1 = Z.T @ grad_out + L2 * params[0]
        gb1 = grad_out.sum(0)
        return [gW1, gb1]

def train_model(X, y, encoders, cfg, rng, n_epochs, batch_size, lr, l2, verbose=False):
    H, C = cfg["H"], 65
    # Build input matrix Z (one-hot concatenated)
    Z = encode_batch(X, encoders)
    input_dim = Z.shape[1]
    params = init_params(rng, input_dim, H, C)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = len(y)
    for epoch in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            z_batch, y_batch = Z[idx], y[idx]
            logits, cache = forward(params, z_batch, H)
            grad_out = softmax_grad(logits, y_batch)
            grads = backward(params, cache, grad_out, H, l2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params, input_dim

def eval_accuracy(X, y, encoders, params, H):
    Z = encode_batch(X, encoders)
    logits, _ = forward(params, Z, H)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def save_weights(encoders, params, input_dim, H, C, path):
    """Format: magic, d, E, H, C; then per-position embedding tables (float32),
    then W1, b1, (W2, b2 if H)."""
    d = len(encoders)
    E = encoders[0][1].shape[1] if encoders[0][0] is not None else 1  # ordinal not used here
    # For each position, we have an encoder matrix of shape (256, num_unique).
    # We need to convert to embedding tables of shape (256, E) where E is the embedding dimension.
    # But our encoders are one-hot over unique values. The embedding dimension is the hidden size H? No.
    # Actually we are not using separate embeddings; we are using one-hot then linear to hidden.
    # The weight format expected by infer.c must match. We'll design infer.c to use the same one-hot encoding.
    # So we save the one-hot encoder matrices (as float32) and then the MLP weights.
    # Magic: 0x454D4244 ('EMBD')
    magic = 0x454D4244
    # We'll store: d, H, C, then for each position: num_unique (int32), then the one-hot matrix (256 * num_unique float32)
    # But that's variable per position. Simpler: store the full concatenated encoder matrix (input_dim x 256?) No.
    # Let's follow the label-linear format but with H>0. The label-linear format stores per-position lookup tables T (d x 256 x W) where W=H (or C).
    # We can do the same: T[j, byte, w] = encoder_matrix[byte, :] @ W1[offset:offset+num_unique, w]
    # That's exactly what label-linear does. So we can reuse that format.
    # Compute T as in label-linear.
    W1 = params[0]
    b1 = params[1]
    off = 0
    T_list = []
    for j, (vals, E_mat) in enumerate(encoders):
        num_unique = E_mat.shape[1]
        # T_j = E_mat (256, num_unique) @ W1[off:off+num_unique, :]  -> (256, W) where W = H if H else C
        W = H if H else C
        T_j = (E_mat.astype(np.float32) @ W1[off:off+num_unique, :]).astype(np.float32)
        T_list.append(T_j)
        off += num_unique
    T_stack = np.stack(T_list)  # (d, 256, W)
    out = struct.pack("<4i", magic, d, H, C)
    out += T_stack.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        W2 = params[2]
        b2 = params[3]
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    with open(path, "wb") as f:
        f.write(out)

def main():
    card = json.load(open(sys.argv[1]))
    train_in = sys.argv[2]
    train_key = sys.argv[3]
    weights_out = sys.argv[4]

    X_bytes = np.frombuffer(b"".join(records(train_in)), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(train_key)], dtype=np.int64)
    types = card["input"]["types"]
    C = card["classes"]

    # Split: last 20% for validation (by file order)
    n = len(y)
    split = int(0.8 * n)
    X_tr, y_tr = X_bytes[:split], y[:split]
    X_val, y_val = X_bytes[split:], y[split:]

    # Build encoders on full training data (as per label-linear: unique values from training)
    encoders = build_encoders(X_bytes, types)

    rng = np.random.default_rng(SEED)
    best_cfg = None
    best_acc = -1.0
    best_params = None
    best_input_dim = 0

    for cfg in GRID:
        # Train on train split
        params, input_dim = train_model(X_tr, y_tr, encoders, cfg, rng,
                                        n_epochs=min(MAX_EPOCHS, math.ceil(SAMPLES / len(y_tr))),
                                        batch_size=BATCH, lr=LR, l2=L2)
        acc = eval_accuracy(X_val, y_val, encoders, params, cfg["H"])
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = params
            best_input_dim = input_dim

    # Retrain on full data with best config
    final_params, final_input_dim = train_model(X_bytes, y, encoders, best_cfg, rng,
                                                n_epochs=min(MAX_EPOCHS, math.ceil(SAMPLES / n)),
                                                batch_size=BATCH, lr=LR, l2=L2)
    save_weights(encoders, final_params, final_input_dim, best_cfg["H"], C, weights_out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Weights format (same as label-linear with H>0):
 *   int32 magic=0x454D4244, d, H, C
 *   float32 T[d][256][W] where W = H if H>0 else C
 *   float32 b1[W]
 *   if H>0: float32 W2[H][C], b2[C]
 * Input record: d bytes (d=16). Output: one byte class index.
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *logits = acc;
        if (H) {
            for (int c = 0; c < C; c++) sc[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) sc[c] += h * W2[(long)u * C + c];
            }
            logits = sc;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```