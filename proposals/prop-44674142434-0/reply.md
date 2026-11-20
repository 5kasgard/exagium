```hypothesis
We propose "label-fm", a factorization machine family that adds low-rank pairwise feature interactions to a linear model. On the Adult dataset with 14 mixed ordinal/categorical features (each quantized to 256 values), FMs capture complementary interactions (e.g., education × occupation) that linear models miss, while staying parameter-efficient (rank k≈4 adds only ~14k parameters). Training uses Adam on the full training set after tuning k, lr, L2 on the last 20% validation split. If dev accuracy does not exceed label-linear (0.8524) or inference cost per item is not competitive, the family fails its goal.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the d=14 bytes as a categorical value 0..255.
Model: for each class c, linear weights w_c[j][v] and interaction embeddings v_c[j][v][f] (f=0..k-1).
Score_c = bias_c + sum_j w_c[j][x_j] + 0.5 * sum_f ( (sum_j v_c[j][x_j][f])^2 - sum_j v_c[j][x_j][f]^2 ).
Hyperparameters (k, lr, L2, epochs) chosen on last 20% of train.in, then retrained on all data.
Weights file: int32 magic 0x464D5F31, d, k, C; then per-class float32 arrays:
  linear_weights[C][d*256], interaction_embeddings[C][d*256*k], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
K_CHOICES = [2, 4, 8]
LR_CHOICES = [0.01, 0.03]
L2_CHOICES = [1e-4, 1e-3]
MAX_EPOCHS_TUNE = 30
MAX_EPOCHS_FULL = 100
BATCH = 64
PATIENCE = 5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(X, params, d, k, C):
    """X: (n, d) uint8. params: dict with keys 'lin', 'int', 'bias' per class.
    Returns scores (n, C)."""
    n = X.shape[0]
    scores = np.zeros((n, C), dtype=np.float32)
    for c in range(C):
        lin = params['lin'][c]          # (d, 256)
        emb = params['int'][c]          # (d, 256, k)
        bias = params['bias'][c]
        # linear part
        for j in range(d):
            scores[:, c] += lin[j, X[:, j]]
        scores[:, c] += bias
        # interaction part
        sum_v = np.zeros((n, k), dtype=np.float32)
        sum_v_sq = np.zeros((n, k), dtype=np.float32)
        for j in range(d):
            v = emb[j, X[:, j], :]      # (n, k)
            sum_v += v
            sum_v_sq += v * v
        interaction = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
        scores[:, c] += interaction
    return scores

def softmax_cross_entropy_grad(scores, y):
    """scores (n, C), y (n,) int. Returns gradient dL/dscores (n, C)."""
    n, C = scores.shape
    max_s = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n  # (n, C)

def train_fm(X, y, d, k, C, lr, l2, epochs, rng, verbose=False):
    n = X.shape[0]
    # Initialize parameters
    params = {
        'lin': [rng.normal(0, 0.01, (d, 256)).astype(np.float32) for _ in range(C)],
        'int': [rng.normal(0, 0.01, (d, 256, k)).astype(np.float32) for _ in range(C)],
        'bias': [np.zeros((), dtype=np.float32) for _ in range(C)],
    }
    # Adam state
    adam_m = {key: [np.zeros_like(p) for p in params[key]] for key in ['lin', 'int', 'bias']}
    adam_v = {key: [np.zeros_like(p) for p in params[key]] for key in ['lin', 'int', 'bias']}
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            scores = fm_forward(xb, params, d, k, C)
            grad_scores = softmax_cross_entropy_grad(scores, yb)  # (b, C)
            # Gradients
            grad_lin = [np.zeros((d, 256), np.float32) for _ in range(C)]
            grad_int = [np.zeros((d, 256, k), np.float32) for _ in range(C)]
            grad_bias = [np.zeros((), np.float32) for _ in range(C)]
            for c in range(C):
                g = grad_scores[:, c]  # (b,)
                # linear grad
                for j in range(d):
                    np.add.at(grad_lin[c][j], xb[:, j], g)
                grad_lin[c] += l2 * params['lin'][c]
                grad_bias[c] = g.sum() + l2 * params['bias'][c]
                # interaction grad
                # For each sample, we need sum_v and sum_v_sq per factor
                sum_v = np.zeros((len(xb), k), np.float32)
                for j in range(d):
                    sum_v += params['int'][c][j, xb[:, j], :]
                # dL/dv = g_i * (sum_v_f - v_{i,j,f})
                for j in range(d):
                    v = params['int'][c][j, xb[:, j], :]  # (b, k)
                    # g_i * (sum_v_f - v_{i,j,f})
                    g_expanded = g[:, None]  # (b, 1)
                    grad = g_expanded * (sum_v - v)  # (b, k)
                    np.add.at(grad_int[c][j], xb[:, j], grad)
                grad_int[c] += l2 * params['int'][c]
            # Adam update
            t += 1
            for key, grads in [('lin', grad_lin), ('int', grad_int), ('bias', grad_bias)]:
                for c in range(C):
                    p = params[key][c]
                    gr = grads[c]
                    m = adam_m[key][c]
                    v = adam_v[key][c]
                    m *= beta1; m += (1 - beta1) * gr
                    v *= beta2; v += (1 - beta2) * gr * gr
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params

def evaluate(X, y, params, d, k, C):
    scores = fm_forward(X, params, d, k, C)
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # Tuning split: last 20% by file order
    split = n - n // 5
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_cfg = None
    # Grid search
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                params = train_fm(X_train, y_train, d, k, C, lr, l2, MAX_EPOCHS_TUNE, rng)
                acc = evaluate(X_val, y_val, params, d, k, C)
                if acc > best_acc:
                    best_acc = acc
                    best_params = params
                    best_cfg = (k, lr, l2)
    # Retrain on full data with best config
    k, lr, l2 = best_cfg
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    final_params = train_fm(X, y, d, k, C, lr, l2, MAX_EPOCHS_FULL, rng)
    # Write weights.bin
    # Layout: magic, d, k, C (int32), then per class: linear[d*256], int[d*256*k], bias
    magic = 0x464D5F31  # "FM_1"
    out = struct.pack("<4i", magic, d, k, C)
    for c in range(C):
        out += final_params['lin'][c].astype("<f4").tobytes()
        out += final_params['int'][c].astype("<f4").tobytes()
        out += final_params['bias'][c].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0 || k > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    /* Expected size: 16 + C*(d*256 + d*256*k + 1)*4 */
    long expected = 16L + (long)C * ( (long)d * 256 * (1 + k) + 1 ) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *lin[2], *emb[2], *bias[2];
    for (int c = 0; c < C; c++) {
        lin[c] = p; p += (long)d * 256;
        emb[c] = p; p += (long)d * 256 * k;
        bias[c] = p; p += 1;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float score[2];
        for (int c = 0; c < C; c++) {
            score[c] = bias[c][0];
            // linear part
            for (int j = 0; j < d; j++) {
                score[c] += lin[c][(long)j * 256 + xi[j]];
            }
            // interaction part
            float sum_v[32], sum_v_sq[32];
            for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v_sq[f] = 0.0f; }
            for (int j = 0; j < d; j++) {
                const float *v = emb[c] + ((long)j * 256 + xi[j]) * k;
                for (int f = 0; f < k; f++) {
                    float val = v[f];
                    sum_v[f] += val;
                    sum_v_sq[f] += val * val;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < k; f++) inter += 0.5f * (sum_v[f] * sum_v[f] - sum_v_sq[f]);
            score[c] += inter;
        }
        unsigned char best = (score[1] > score[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```