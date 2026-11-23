```hypothesis
We propose a Factorization Machine (FM) family with per-class embeddings (k=8). It captures pairwise feature interactions via low-rank embeddings, which should improve accuracy over the linear model on this tabular dataset with mixed categorical/ordinal features. The model uses a precomputed lookup table for fast inference (similar to label-linear). If dev accuracy does not exceed 0.8524 (label-linear) or matches it at higher CPU cost, the hypothesis fails.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports weights as: magic 0x464D5431, d, k, C; float32 T[d][256][C*(1+k)], bias[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
K = 8                 # embedding dimension
SEED = 1
LR = 1e-2
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

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Build encodings and parameter structures
    feature_info = []
    params = []          # list of parameter arrays
    param_shapes = []    # their shapes for gradient accumulation

    # Bias
    bias = np.zeros(C, dtype=np.float32)
    params.append(bias)
    param_shapes.append(bias.shape)

    # For each feature, create parameters and encoding helpers
    for j, t in enumerate(types):
        if t == "ordinal":
            # Normalize ordinal values
            col = X[:, j].astype(np.float64)
            mu, sd = col.mean(), col.std() or 1.0
            norm_vals = ((np.arange(256) - mu) / sd).astype(np.float32)
            # Parameters: w (C,), V (C, K)
            w = (rng.normal(0, 0.01, size=C)).astype(np.float32)
            V = (rng.normal(0, 0.01, size=(C, K))).astype(np.float32)
            params.append(w)
            params.append(V)
            param_shapes.append(w.shape)
            param_shapes.append(V.shape)
            feature_info.append(("ordinal", norm_vals, len(params)-2, len(params)-1))
        else:
            # Categorical: map seen values to indices
            vals = np.unique(X[:, j])
            mapping = np.full(256, -1, dtype=np.int32)
            mapping[vals] = np.arange(len(vals))
            num_cats = len(vals)
            # Parameters: w_cat (num_cats, C), V_cat (num_cats, C, K)
            w_cat = (rng.normal(0, 0.01, size=(num_cats, C))).astype(np.float32)
            V_cat = (rng.normal(0, 0.01, size=(num_cats, C, K))).astype(np.float32)
            params.append(w_cat)
            params.append(V_cat)
            param_shapes.append(w_cat.shape)
            param_shapes.append(V_cat.shape)
            feature_info.append(("categorical", mapping, len(params)-2, len(params)-1))

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute normalized ordinal columns for speed
    X_ord = np.zeros((n, d), dtype=np.float32)
    for idx, info in enumerate(feature_info):
        if info[0] == "ordinal":
            norm_vals = info[1]
            X_ord[:, idx] = norm_vals[X[:, idx]]

    # Training loop
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            if len(idx) == 0:
                continue
            xb = X[idx]          # (B, d) uint8
            xb_ord = X_ord[idx]  # (B, d) float32
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward pass
            linear = np.broadcast_to(bias, (B, C)).copy()  # (B, C)
            S = np.zeros((B, C, K), dtype=np.float32)
            Q = np.zeros((B, C, K), dtype=np.float32)

            for j, info in enumerate(feature_info):
                if info[0] == "ordinal":
                    vals = xb_ord[:, j]                 # (B,)
                    w_idx, V_idx = info[2], info[3]
                    w = params[w_idx]                   # (C,)
                    V = params[V_idx]                   # (C, K)
                    linear += vals[:, None] * w[None, :]
                    S += vals[:, None, None] * V[None, :, :]
                    Q += vals[:, None, None] * (V[None, :, :] ** 2)
                else:
                    mapping, w_idx, V_idx = info[1], info[2], info[3]
                    cat = xb[:, j]
                    cat_idx = mapping[cat]              # (B,) -1 for unseen
                    mask = cat_idx >= 0
                    if not mask.any():
                        continue
                    w_cat = params[w_idx]               # (num_cats, C)
                    V_cat = params[V_idx]               # (num_cats, C, K)
                    sel_w = w_cat[cat_idx[mask]]        # (M, C)
                    sel_V = V_cat[cat_idx[mask]]        # (M, C, K)
                    linear[mask] += sel_w
                    S[mask] += sel_V
                    Q[mask] += sel_V ** 2

            interaction = 0.5 * np.sum(S * S - Q, axis=2)  # (B, C)
            logits = linear + interaction

            # Softmax cross-entropy gradient
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            g = probs / B                               # (B, C)

            # Backward pass: accumulate gradients in grads list
            grads = [np.zeros_like(p) for p in params]
            grads[0] = g.sum(axis=0)                    # bias gradient

            for j, info in enumerate(feature_info):
                if info[0] == "ordinal":
                    vals = xb_ord[:, j]                 # (B,)
                    w_idx, V_idx = info[2], info[3]
                    w = params[w_idx]
                    V = params[V_idx]
                    # grad_w
                    grads[w_idx] += (g * vals[:, None]).sum(axis=0)
                    # grad_V: g_c * x_i * (S_cf - x_i * V_cf)
                    diff = S - vals[:, None, None] * V[None, :, :]   # (B, C, K)
                    grads[V_idx] += (g[:, :, None] * vals[:, None, None] * diff).sum(axis=0)
                else:
                    mapping, w_idx, V_idx = info[1], info[2], info[3]
                    cat = xb[:, j]
                    cat_idx = mapping[cat]
                    mask = cat_idx >= 0
                    if not mask.any():
                        continue
                    g_mask = g[mask]                    # (M, C)
                    S_mask = S[mask]                    # (M, C, K)
                    cats = cat_idx[mask]
                    # Use np.add.at for unsorted indices
                    np.add.at(grads[w_idx], cats, g_mask)
                    # For V: need diff = S_mask - V_cat[cats]
                    V_cat = params[V_idx]
                    diff = S_mask - V_cat[cats]         # (M, C, K)
                    # g_mask[:, :, None] * diff -> (M, C, K)
                    np.add.at(grads[V_idx], cats, g_mask[:, :, None] * diff)

            # L2 regularization
            for i, p in enumerate(params):
                if param_shapes[i] != (C,):  # skip bias
                    grads[i] += L2 * p

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup table T for inference: T[j][v] -> vector of length C*(1+K)
    W = C * (1 + K)
    T = np.zeros((d, 256, W), dtype=np.float32)
    for j, info in enumerate(feature_info):
        if info[0] == "ordinal":
            norm_vals = info[1]
            w_idx, V_idx = info[2], info[3]
            w = params[w_idx]
            V = params[V_idx]
            for v in range(256):
                val = norm_vals[v]
                for c in range(C):
                    base = c * (1 + K)
                    T[j, v, base] = val * w[c]
                    T[j, v, base+1:base+1+K] = val * V[c]
        else:
            mapping, w_idx, V_idx = info[1], info[2], info[3]
            w_cat = params[w_idx]
            V_cat = params[V_idx]
            for v in range(256):
                cat = mapping[v]
                if cat >= 0:
                    for c in range(C):
                        base = c * (1 + K)
                        T[j, v, base] = w_cat[cat, c]
                        T[j, v, base+1:base+1+K] = V_cat[cat, c]

    # Write weights.bin
    magic = 0x464D5431  # "FMT1"
    header = struct.pack("<4i", magic, d, K, C)
    out = header + T.astype("<f4").tobytes() + bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax of FM logits).
 * Weights format: magic 0x464D5431, d, k, C; float32 T[d][256][C*(1+k)], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x464D5431) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    int W = C * (1 + K);
    long expected = 16L + 4L * ((long)d * 256 * W + C);
    if (C < 1 || C > 256 || nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *bias = T + (long)d * 256 * W;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *linear = malloc(sizeof(float) * C);
    float *S = malloc(sizeof(float) * C * K);
    float *Q = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            linear[c] = bias[c];
            for (int f = 0; f < K; f++) {
                S[c*K + f] = 0.0f;
                Q[c*K + f] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int c = 0; c < C; c++) {
                int base = c * (1 + K);
                linear[c] += t[base];
                for (int f = 0; f < K; f++) {
                    float v = t[base + 1 + f];
                    S[c*K + f] += v;
                    Q[c*K + f] += v * v;
                }
            }
        }

        float best_score = -1e30f;
        unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float s = S[c*K + f];
                inter += s * s - Q[c*K + f];
            }
            float score = linear[c] + 0.5f * inter;
            if (score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```