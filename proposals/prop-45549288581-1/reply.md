```hypothesis
Family: label-mlp64 – a one-hidden-layer MLP with 64 ReLU units, hyperparameters tuned on the last 20% of train.in.
Why: The linear model (0.9613) cannot capture pixel interactions; a wider hidden layer learns non-linear features (strokes, loops) in 8×8 digits. Tuning H and LR on the validation split should beat linear and approach label-mlp16 (0.9770) with moderate compute.
Failure indicator: dev accuracy ≤ 0.965 means the extra capacity does not generalize or overfits.
```

```python
"""label-mlp64: one-hidden-layer MLP (64 ReLU units) with tuned hyperparameters.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal features as z-scored per-byte lookup tables.
Trains with Adam (lr, L2, batch, epochs from tuning), exports table format (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

SEED = 1
SAMPLES = 600000
MAX_EPOCHS = 100
BATCH = 64

# Hyperparameter grid for tuning (last 20% of train.in in file order)
H_CHOICES = [32, 64, 128]
LR_CHOICES = [1e-3, 3e-3]
L2_CHOICES = [1e-4, 1e-5]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoding(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) for each feature."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def encode_data(X, enc):
    """Apply encodings to X (n x d) -> Z (n x k)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def train_model(Z_train, y_train, Z_val, y_val, H, LR, L2, seed, n_samples, max_epochs, batch):
    """Train MLP on (Z_train, y_train), validate on (Z_val, y_val). Return (acc, params)."""
    n, k = Z_train.shape
    C = int(y_train.max() + 1)
    rng = np.random.default_rng(seed)
    # Xavier init
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(max_epochs, math.ceil(n_samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z_train[idx], y_train[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # softmax + cross-entropy gradient
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            # backward
            gh = (g @ W2.T) * (a > 0)
            grads = [
                z.T @ gh + L2 * W1,
                gh.sum(0),
                h.T @ g + L2 * W2,
                g.sum(0)
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # validation accuracy
    a_val = Z_val @ W1 + b1
    h_val = np.maximum(a_val, 0)
    logits_val = h_val @ W2 + b2
    pred = logits_val.argmax(axis=1)
    acc = (pred == y_val).mean()
    return acc, params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = X.shape[1]

    # Split: last 20% for tuning (file order)
    n = len(X)
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Build encoding on training split only
    enc = build_encoding(X_train, types)
    Z_train = encode_data(X_train, enc)
    Z_tune = encode_data(X_tune, enc)

    # Grid search
    best_acc = -1.0
    best_params = None
    best_H = best_LR = best_L2 = None
    for H in H_CHOICES:
        for LR in LR_CHOICES:
            for L2 in L2_CHOICES:
                # Use a deterministic seed per configuration
                cfg_seed = SEED + H * 1000 + int(LR * 1e5) + int(L2 * 1e6)
                acc, params = train_model(Z_train, y_train, Z_tune, y_tune,
                                          H, LR, L2, cfg_seed, SAMPLES, MAX_EPOCHS, BATCH)
                if acc > best_acc:
                    best_acc = acc
                    best_params = params
                    best_H, best_LR, best_L2 = H, LR, L2
    print(f"Tuned: H={best_H}, LR={best_LR}, L2={best_L2}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full dataset with best hyperparameters
    enc_full = build_encoding(X, types)
    Z_full = encode_data(X, enc_full)
    _, final_params = train_model(Z_full, y, Z_full, y,  # validation not used
                                  best_H, best_LR, best_L2, SEED, SAMPLES, MAX_EPOCHS, BATCH)
    W1, b1, W2, b2 = final_params

    # Export weights in table format (same as label-linear example)
    # T_j = E_j @ W1[offset:offset+dim_j] for each feature j
    off = 0
    T_list = []
    for E in enc_full:
        dim = E.shape[1]
        T_list.append((E.astype(np.float32) @ W1[off:off + dim]).astype(np.float32))
        off += dim
    T_stack = np.stack(T_list)  # (d, 256, H)
    W = best_H  # H > 0
    out = struct.pack("<4i", 0x54424C31, d, best_H, C)
    out += T_stack.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```