```hypothesis
We propose a Factorization Machine family (label-fm) that adds low-rank pairwise interactions to the linear model.
The task predicts the next character from 16 previous bytes; interactions between positions (e.g., bigrams) are crucial.
FM captures all pairwise feature interactions with O(k·d) parameters (k=8, d=16) and inference cost similar to linear.
If dev accuracy does not exceed label-linear's 0.3375 or inference cost increases significantly, the hypothesis fails.
```
```python
"""Factorization Machine for next-character prediction (label-fm).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x464D3132, int32 d, k, C, num_features; float32 W_lin[num_features*C], V[num_features*k], bias[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
K = 8                # embedding dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 20

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    num_features = d * 256              # 4096

    # Load training data
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    n = len(X_recs)
    # Convert to feature indices: for each position j, byte b -> index = j*256 + b
    X_idx = np.zeros((n, d), dtype=np.int32)
    for i, rec in enumerate(X_recs):
        X_idx[i] = np.frombuffer(rec, dtype=np.uint8) + np.arange(d) * 256
    y = np.array([r[0] for r in y_recs], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    # Initialize parameters
    W_lin = (rng.normal(0, 0.01, (num_features, C))).astype(np.float32)
    V = (rng.normal(0, 0.01, (num_features, K))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Adam state
    params = [W_lin, V, bias]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_idx = X_idx[idx]          # (B, d)
            batch_y = y[idx]                # (B,)
            B = len(idx)

            # Forward
            # Linear scores: gather W_lin for active features and sum over positions
            # W_lin[batch_idx] -> (B, d, C) -> sum over d -> (B, C)
            lin_scores = W_lin[batch_idx].sum(axis=1)   # (B, C)

            # Interaction term (scalar per example)
            V_batch = V[batch_idx]                      # (B, d, K)
            sum_v = V_batch.sum(axis=1)                 # (B, K)
            sum_v_sq = (V_batch ** 2).sum(axis=1)       # (B, K)
            interaction = 0.5 * ((sum_v ** 2 - sum_v_sq).sum(axis=1))  # (B,)

            scores = lin_scores + interaction[:, None] + bias  # (B, C)

            # Softmax cross-entropy
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), batch_y] -= 1.0
            grad_scores = probs / B                       # (B, C)

            # Backward
            # Gradient for W_lin: for each active feature, add grad_scores[example]
            # Use np.add.at for sparse accumulation
            grad_W = np.zeros_like(W_lin)
            np.add.at(grad_W, batch_idx.ravel(), np.repeat(grad_scores, d, axis=0))
            grad_W += L2 * W_lin

            # Gradient for V: interaction gradient
            # g_s = sum_c grad_scores (since interaction added to all classes)
            g_s = grad_scores.sum(axis=1)                 # (B,)
            # For each example and each active feature: g_s * (sum_v - v_feat)
            # sum_v is (B, K), V_batch is (B, d, K)
            grad_V_batch = g_s[:, None, None] * (sum_v[:, None, :] - V_batch)  # (B, d, K)
            grad_V = np.zeros_like(V)
            np.add.at(grad_V, batch_idx.ravel(), grad_V_batch.reshape(-1, K))
            grad_V += L2 * V

            # Gradient for bias
            grad_bias = grad_scores.sum(axis=0)

            grads = [grad_W, grad_V, grad_bias]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D3132  # "FM12"
    header = struct.pack("<5i", magic, d, K, C, num_features)
    out = header
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference (label-fm): ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3], num_features = hd[4];
    if (d != 16 || C < 1 || C > 256 || num_features != d * 256) {
        fprintf(stderr, "bad weights dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)num_features * C + (long)num_features * K + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W_lin = (const float *)(hd + 5);
    const float *V = W_lin + (long)num_features * C;
    const float *bias = V + (long)num_features * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear scores + bias
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            int feat = j * 256 + xi[j];
            const float *w = W_lin + (long)feat * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        // Interaction term
        for (int u = 0; u < K; u++) { sum_v[u] = 0.0f; sum_v_sq[u] = 0.0f; }
        for (int j = 0; j < d; j++) {
            int feat = j * 256 + xi[j];
            const float *v = V + (long)feat * K;
            for (int u = 0; u < K; u++) {
                float val = v[u];
                sum_v[u] += val;
                sum_v_sq[u] += val * val;
            }
        }
        float interaction = 0.0f;
        for (int u = 0; u < K; u++) interaction += 0.5f * (sum_v[u] * sum_v[u] - sum_v_sq[u]);
        for (int c = 0; c < C; c++) scores[c] += interaction;

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```