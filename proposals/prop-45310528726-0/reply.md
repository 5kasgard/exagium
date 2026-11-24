```hypothesis
We propose "label-fm", a Factorization Machine family that models pairwise feature interactions via low-rank embeddings (dimension D=8). Adult's mixed ordinal/categorical features benefit from 2nd-order interactions, which FM captures efficiently (O(d·D) per item). The model uses the same per-byte preprocessing as label-linear (z-scored ordinals, one-hot categoricals) and trains with Adam on a validation split (last 20%) to choose hyperparameters, then re-trains on all data. If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds ~1500 Ir/item, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Encodes: ordinal -> z-scored, categorical -> one-hot over training values.
FM: w0 + w·x + 0.5*sum_f((sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2).
Exports per-byte lookup tables for linear, embedding, and squared-embedding contributions.
Weights format: magic 0x464D4C31, d, C, D; then per-class: w0, then for each column j (0..d-1)
and byte b (0..255): linear[c][j][b], embed[c][j][b][D], embed_sq[c][j][b][D] (all float32)."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid (will be evaluated on last 20% of train.in)
D_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.005]
L2_CHOICES = [1e-4, 1e-5]
EPOCHS_CHOICES = [30, 50]
BATCH = 64
VAL_FRACTION = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def encode_data(X, types, enc_params=None):
    """Return (Z, enc_params) where Z is n x k float32, enc_params holds per-column info."""
    n, d = X.shape
    if enc_params is None:
        enc_params = []
    Z_parts = []
    for j, t in enumerate(types):
        col = X[:, j]
        if t == "ordinal":
            if enc_params and j < len(enc_params):
                mu, sd = enc_params[j]
            else:
                mu = col.mean()
                sd = col.std() or 1.0
                enc_params.append((mu, sd))
            z = ((col.astype(np.float32) - mu) / sd)[:, None]
            Z_parts.append(z)
        else:
            if enc_params and j < len(enc_params):
                vals = enc_params[j]
            else:
                vals = np.unique(col)
                enc_params.append(vals)
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            Z_parts.append(E[col])
    Z = np.concatenate(Z_parts, axis=1)
    return Z, enc_params

def fm_forward(Z, w0, w, V, C, D):
    """Z: n x k, returns scores n x C"""
    n = Z.shape[0]
    scores = np.zeros((n, C), dtype=np.float32)
    for c in range(C):
        lin = w0[c] + Z @ w[c]
        ZV = Z @ V[c]                 # n x D
        Z2V2 = (Z**2) @ (V[c]**2)     # n x D
        inter = 0.5 * np.sum(ZV**2 - Z2V2, axis=1)
        scores[:, c] = lin + inter
    return scores

def softmax_cross_entropy(scores, y):
    n = scores.shape[0]
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return loss, grad

def fm_backward(Z, grad, w, V, C, D, L2):
    """grad: n x C (dL/dscores). Returns gradients for w0, w, V."""
    n, k = Z.shape
    gw0 = np.zeros(C, dtype=np.float32)
    gw = np.zeros((C, k), dtype=np.float32)
    gV = np.zeros((C, k, D), dtype=np.float32)
    for c in range(C):
        g = grad[:, c]                 # n
        gw0[c] = g.sum()
        gw[c] = Z.T @ g + L2 * w[c]
        ZV = Z @ V[c]                  # n x D
        # d inter / d V = Z.T @ (g[:,None] * ZV) - (Z**2).T @ (g[:,None] * V[c])
        gV[c] = Z.T @ (g[:, None] * ZV) - (Z**2).T @ (g[:, None] * V[c]) + L2 * V[c]
    return gw0, gw, gV

def adam_step(params, grads, state, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    new_params = []
    new_state = []
    for p, gr, (m, v) in zip(params, grads, state):
        m = beta1 * m + (1 - beta1) * gr
        v = beta2 * v + (1 - beta2) * (gr * gr)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        new_params.append(p)
        new_state.append((m, v))
    return new_params, new_state

def train_fm(Z, y, C, D, lr, L2, epochs, rng):
    n, k = Z.shape
    # init
    w0 = np.zeros(C, dtype=np.float32)
    w = rng.uniform(-0.01, 0.01, (C, k)).astype(np.float32)
    V = rng.uniform(-0.01, 0.01, (C, k, D)).astype(np.float32)
    params = [w0, w, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z, yy = Z[idx], y[idx]
            scores = fm_forward(z, w0, w, V, C, D)
            loss, grad = softmax_cross_entropy(scores, yy)
            gw0, gw, gV = fm_backward(z, grad, w, V, C, D, L2)
            t += 1
            params, state = adam_step(params, [gw0, gw, gV], state, t, lr)
            w0, w, V = params
    return w0, w, V

def build_lookup_tables(enc_params, types, w0, w, V, C, D):
    """Return flat arrays for inference: linear[C][d][256], embed[C][d][256][D], embed_sq[C][d][256][D]"""
    d = len(types)
    linear = np.zeros((C, d, 256), dtype=np.float32)
    embed = np.zeros((C, d, 256, D), dtype=np.float32)
    embed_sq = np.zeros((C, d, 256, D), dtype=np.float32)
    # We need to map each column to its feature indices in the concatenated Z.
    # Reconstruct the feature index ranges per column.
    feat_starts = []
    offset = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            feat_starts.append((offset, 1))
            offset += 1
        else:
            vals = enc_params[j]
            feat_starts.append((offset, len(vals)))
            offset += len(vals)
    for c in range(C):
        for j, t in enumerate(types):
            start, length = feat_starts[j]
            if t == "ordinal":
                mu, sd = enc_params[j]
                for b in range(256):
                    x = (b - mu) / sd
                    linear[c, j, b] = w[c, start] * x
                    for f in range(D):
                        v = V[c, start, f]
                        embed[c, j, b, f] = v * x
                        embed_sq[c, j, b, f] = (v * x) ** 2
            else:
                vals = enc_params[j]
                # map byte value to feature index within this column
                for idx, val in enumerate(vals):
                    b = int(val)
                    linear[c, j, b] = w[c, start + idx]
                    for f in range(D):
                        v = V[c, start + idx, f]
                        embed[c, j, b, f] = v
                        embed_sq[c, j, b, f] = v * v
                # unseen (255 or others) remain zero
    return linear, embed, embed_sq

def main():
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    C = card["classes"]
    d = card["input"]["bytes"]
    # Load all training data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split: last 20% for validation
    val_start = int(n * (1 - VAL_FRACTION))
    X_train, X_val = X[:val_start], X[val_start:]
    y_train, y_val = y[:val_start], y[val_start:]
    # Encode training portion to get enc_params
    Z_train, enc_params = encode_data(X_train, types)
    Z_val, _ = encode_data(X_val, types, enc_params)
    k = Z_train.shape[1]
    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_hp = None
    best_model = None
    # Grid search
    for D in D_CHOICES:
        for lr in LR_CHOICES:
            for L2 in L2_CHOICES:
                for epochs in EPOCHS_CHOICES:
                    w0, w, V = train_fm(Z_train, y_train, C, D, lr, L2, epochs, rng)
                    scores_val = fm_forward(Z_val, w0, w, V, C, D)
                    preds = scores_val.argmax(axis=1)
                    acc = (preds == y_val).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_hp = (D, lr, L2, epochs)
                        best_model = (w0.copy(), w.copy(), V.copy())
    print(f"Best hyperparams: D={best_hp[0]}, lr={best_hp[1]}, L2={best_hp[2]}, epochs={best_hp[3]}, val_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data with best hyperparameters
    Z_full, enc_params_full = encode_data(X, types)
    D, lr, L2, epochs = best_hp
    w0, w, V = train_fm(Z_full, y, C, D, lr, L2, epochs, rng)
    # Build lookup tables
    linear, embed, embed_sq = build_lookup_tables(enc_params_full, types, w0, w, V, C, D)
    # Write weights.bin
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, C, D)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(w0.astype("<f4").tobytes())
        # Order: class, column, byte, [D]
        f.write(linear.astype("<f4").tobytes())
        f.write(embed.astype("<f4").tobytes())
        f.write(embed_sq.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], D = hd[3];
    if (C < 1 || C > 256 || D < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    /* Expected size: header(16) + w0(C) + linear(C*d*256) + embed(C*d*256*D) + embed_sq(C*d*256*D) */
    long expected = 16L + 4L * (C + 3L * C * d * 256 * D);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *linear = p; p += (long)C * d * 256;
    const float *embed = p; p += (long)C * d * 256 * D;
    const float *embed_sq = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_linear = calloc(C, sizeof(float));
    float *sum_v = calloc((long)C * D, sizeof(float));
    float *sum_v_sq = calloc((long)C * D, sizeof(float));
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(sum_linear, 0, C * sizeof(float));
        memset(sum_v, 0, (long)C * D * sizeof(float));
        memset(sum_v_sq, 0, (long)C * D * sizeof(float));
        for (int j = 0; j < d; j++) {
            unsigned char b = xi[j];
            for (int c = 0; c < C; c++) {
                long base_lin = ((long)c * d + j) * 256;
                sum_linear[c] += linear[base_lin + b];
                long base_emb = ((long)c * d + j) * 256 * D;
                for (int f = 0; f < D; f++) {
                    long idx = base_emb + b * D + f;
                    sum_v[c * D + f] += embed[idx];
                    sum_v_sq[c * D + f] += embed_sq[idx];
                }
            }
        }
        best = 0;
        float best_score = -1e30f;
        for (int c = 0; c < C; c++) {
            float score = w0[c] + sum_linear[c];
            for (int f = 0; f < D; f++) {
                float v = sum_v[c * D + f];
                score += 0.5f * (v * v - sum_v_sq[c * D + f]);
            }
            if (score > best_score) {
                best_score = score;
                best = (unsigned char)c;
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```