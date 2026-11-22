```hypothesis
Family: label-fm (Factorization Machine for multi-class classification).
Why: Optdigits has 64 ordinal pixel features; pairwise interactions capture stroke patterns that linear models miss.
A degree-2 FM with low-rank latent vectors (k=10) adds only ~7k parameters (vs 640 for linear) and ~6400 FLOPs/item,
far cheaper than MLP16 (1184 params but similar ops) while modeling feature interactions.
If dev accuracy does not exceed label-linear (0.9613) or latency exceeds label-mlp16 (8685 Ir/item),
the family does not justify its complexity.
```

```python
"""Factorization Machine (degree 2) for multi-class label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal features by z-scoring (mean/std from training data).
Model: per-class bias, linear weights, and latent vectors V[d][k].
Score_c = bias_c + sum_j w_cj * x_j + 0.5 * sum_f ( (sum_j V_cjf * x_j)^2 - sum_j (V_cjf * x_j)^2 )
Trains with Adam (lr=3e-3, L2=1e-4, batch=64, epochs up to 100) on softmax cross-entropy.
Hyperparameters (k, lr, L2) chosen on last 20% of train.in (file order), then retrain on all train.
Weights format: magic 0x464D3132, d, C, k; float32 mean[d], std[d]; per class: bias, w[d], V[d*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
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

def softmax_cross_entropy_grad(scores, y):
    """scores: (B, C), y: (B,) int. Returns gradient dL/dscores (B, C)."""
    B = scores.shape[0]
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B

def fm_scores(X, params, C, k):
    """X: (B, d) normalized. params: list of (bias, w, V) per class.
    Returns scores (B, C)."""
    B, d = X.shape
    scores = np.zeros((B, C), dtype=np.float32)
    for c in range(C):
        bias, w, V = params[c]
        lin = X @ w + bias
        XV = X @ V                      # (B, k)
        XV_sq = (X * X) @ (V * V)       # (B, k)
        inter = 0.5 * np.sum(XV * XV - XV_sq, axis=1)
        scores[:, c] = lin + inter
    return scores

def fm_backward(X, grad_scores, params, C, k, L2):
    """grad_scores: (B, C). Returns list of (gbias, gw, gV) per class."""
    B, d = X.shape
    grads = []
    for c in range(C):
        g = grad_scores[:, c]           # (B,)
        gbias = g.sum()
        gw = X.T @ g                    # (d,)
        # interaction gradient
        # d inter / d V = X.T @ (g[:,None] * (XV - X * V))  but we compute per sample
        # We'll do a simple loop over batch (B <= 64) for clarity and determinism.
        gV = np.zeros((d, k), dtype=np.float32)
        for b in range(B):
            xb = X[b]
            xbV = xb @ params[c][2]           # (k,)
            # derivative of 0.5*( (xbV)^2 - sum(xb^2 * V^2) ) w.r.t V
            # = xb[:,None] * (xbV[None,:] - xb[:,None] * V)
            gV += g[b] * np.outer(xb, xbV - xb * params[c][2])
        gw += L2 * params[c][1]
        gV += L2 * params[c][2]
        grads.append((gbias, gw, gV))
    return grads

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_bytes.shape[1]
    n = X_bytes.shape[0]

    # --- Tuning split: last 20% in file order ---
    split = int(0.8 * n)
    X_tr, X_val = X_bytes[:split], X_bytes[split:]
    y_tr, y_val = y[:split], y[split:]

    # Normalization stats on training portion
    mean = X_tr.mean(axis=0).astype(np.float32)
    std = X_tr.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0

    def normalize(X):
        return ((X.astype(np.float32) - mean) / std).astype(np.float32)

    X_tr_n = normalize(X_tr)
    X_val_n = normalize(X_val)

    # Hyperparameter grid
    best_acc = -1.0
    best_params = None
    best_hp = None

    for k in (5, 10, 20):
        for LR in (1e-3, 3e-3):
            for L2 in (1e-4, 1e-3):
                rng = np.random.default_rng(SEED)
                params = []
                for c in range(C):
                    bias = 0.0
                    w = (rng.uniform(-1, 1, d) * math.sqrt(6.0 / d)).astype(np.float32)
                    V = (rng.uniform(-1, 1, (d, k)) * math.sqrt(6.0 / (d * k))).astype(np.float32)
                    params.append([bias, w, V])
                # Adam state
                state = []
                for c in range(C):
                    state.append([
                        (np.zeros_like(params[c][0]), np.zeros_like(params[c][0])),  # bias
                        (np.zeros_like(params[c][1]), np.zeros_like(params[c][1])),  # w
                        (np.zeros_like(params[c][2]), np.zeros_like(params[c][2]))   # V
                    ])
                t = 0
                epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
                for _ in range(epochs):
                    order = rng.permutation(len(X_tr))
                    for s in range(0, len(X_tr), BATCH):
                        idx = order[s:s + BATCH]
                        xb, yb = X_tr_n[idx], y_tr[idx]
                        scores = fm_scores(xb, params, C, k)
                        grad_scores = softmax_cross_entropy_grad(scores, yb)
                        grads = fm_backward(xb, grad_scores, params, C, k, L2)
                        t += 1
                        for c in range(C):
                            for i in range(3):
                                m, v = state[c][i]
                                gr = grads[c][i]
                                m *= 0.9; m += 0.1 * gr
                                v *= 0.999; v += 0.001 * gr * gr
                                m_hat = m / (1 - 0.9 ** t)
                                v_hat = v / (1 - 0.999 ** t)
                                params[c][i] -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                # Evaluate on validation
                val_scores = fm_scores(X_val_n, params, C, k)
                val_pred = val_scores.argmax(axis=1)
                acc = (val_pred == y_val).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = [ (p[0].copy(), p[1].copy(), p[2].copy()) for p in params ]
                    best_hp = (k, LR, L2)

    print(f"Best hyperparams: k={best_hp[0]}, lr={best_hp[1]}, L2={best_hp[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full training set with best hyperparameters
    k, LR, L2 = best_hp
    mean_full = X_bytes.mean(axis=0).astype(np.float32)
    std_full = X_bytes.std(axis=0).astype(np.float32)
    std_full[std_full == 0] = 1.0
    X_full_n = ((X_bytes.astype(np.float32) - mean_full) / std_full).astype(np.float32)

    rng = np.random.default_rng(SEED)
    params = []
    for c in range(C):
        bias = 0.0
        w = (rng.uniform(-1, 1, d) * math.sqrt(6.0 / d)).astype(np.float32)
        V = (rng.uniform(-1, 1, (d, k)) * math.sqrt(6.0 / (d * k))).astype(np.float32)
        params.append([bias, w, V])
    state = []
    for c in range(C):
        state.append([
            (np.zeros_like(params[c][0]), np.zeros_like(params[c][0])),
            (np.zeros_like(params[c][1]), np.zeros_like(params[c][1])),
            (np.zeros_like(params[c][2]), np.zeros_like(params[c][2]))
        ])
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_full_n[idx], y[idx]
            scores = fm_scores(xb, params, C, k)
            grad_scores = softmax_cross_entropy_grad(scores, yb)
            grads = fm_backward(xb, grad_scores, params, C, k, L2)
            t += 1
            for c in range(C):
                for i in range(3):
                    m, v = state[c][i]
                    gr = grads[c][i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    params[c][i] -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, C, k
    out = struct.pack("<4i", 0x464D3132, d, C, k)
    out += mean_full.astype("<f4").tobytes()
    out += std_full.astype("<f4").tobytes()
    for c in range(C):
        bias, w, V = params[c]
        out += struct.pack("<f", bias)
        out += w.astype("<f4").tobytes()
        out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (degree 2) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (pixels 0..16). Output: one byte class index (argmax). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1], C = (int)hd[2], k = (int)hd[3];
    if (C < 1 || C > 256 || d < 1 || k < 1) { fprintf(stderr, "bad dims\n"); return 3; }

    float *mean = (float *)(hd + 4);
    float *std  = mean + d;
    size_t per_class = 1 + d + (size_t)d * k;  // bias, w[d], V[d*k]
    size_t expected = 16 + (size_t)d * 8 + (size_t)C * per_class * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    float *class_data = std + d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *xnorm = malloc(d * sizeof(float));
    float *sum_v = malloc(k * sizeof(float));
    float *sum_v_sq = malloc(k * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        for (int j = 0; j < d; j++) {
            float xv = (float)xi[j];
            float s = std[j];
            xnorm[j] = s != 0.0f ? (xv - mean[j]) / s : 0.0f;
        }
        float best_score = -INFINITY;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            float *p = class_data + c * per_class;
            float bias = p[0];
            float *w = p + 1;
            float *V = p + 1 + d;
            float score = bias;
            for (int j = 0; j < d; j++) score += w[j] * xnorm[j];
            memset(sum_v, 0, k * sizeof(float));
            memset(sum_v_sq, 0, k * sizeof(float));
            for (int j = 0; j < d; j++) {
                float xj = xnorm[j];
                if (xj == 0.0f) continue;
                for (int f = 0; f < k; f++) {
                    float v = V[(size_t)j * k + f] * xj;
                    sum_v[f] += v;
                    sum_v_sq[f] += v * v;
                }
            }
            for (int f = 0; f < k; f++) {
                score += 0.5f * (sum_v[f] * sum_v[f] - sum_v_sq[f]);
            }
            if (score > best_score) {
                best_score = score;
                best_class = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```