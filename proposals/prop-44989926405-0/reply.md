```hypothesis
Propose label-fm: a factorization machine with tied embeddings for linear and pairwise interactions.
Adult's mixed categorical/ordinal features benefit from 2nd-order interactions (e.g., education × occupation).
With embedding size K=8, inference cost is O(d·K) ≈ 112 ops/item – cheaper than label-mlp16 (H=16) while capturing non-linearities.
If dev accuracy ≤ 0.855, the interaction term adds no value over label-linear.
```
```python
"""Factorization Machine with tied embeddings for linear and pairwise interactions.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x464D4C31, d, C, K; then float32 arrays:
  linear[d][256][C], embed[d][256][K], bias[C], inter_w[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
K = 8                # embedding size for interactions
LR = 1e-2            # Adam learning rate
L2 = 1e-4            # L2 regularization
BATCH = 256          # minibatch size
EPOCHS = 30          # training epochs
SEED = 1             # fixed seed for reproducibility
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x464D4C31   # 'FML1'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def adam_update(param, grad, m, v, t, lr=LR):
    m *= BETA1; m += (1 - BETA1) * grad
    v *= BETA2; v += (1 - BETA2) * grad * grad
    m_hat = m / (1 - BETA1 ** t)
    v_hat = v / (1 - BETA2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + EPS)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 14 and C == 2

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters
    linear = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    embed  = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    bias   = np.zeros(C, np.float32)
    inter_w = np.zeros(C, np.float32)

    # Adam states
    m_linear, v_linear = np.zeros_like(linear), np.zeros_like(linear)
    m_embed,  v_embed  = np.zeros_like(embed),  np.zeros_like(embed)
    m_bias,   v_bias   = np.zeros_like(bias),   np.zeros_like(bias)
    m_inter,  v_inter  = np.zeros_like(inter_w), np.zeros_like(inter_w)

    t = 0
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward pass
            linear_sum = np.zeros((B, C), np.float32)
            embed_sum  = np.zeros((B, K), np.float32)
            embed_sq_sum = np.zeros(B, np.float32)
            for j in range(d):
                vals = xb[:, j]
                linear_sum += linear[j, vals]
                e = embed[j, vals]
                embed_sum += e
                embed_sq_sum += np.sum(e * e, axis=1)

            inter = 0.5 * (np.sum(embed_sum * embed_sum, axis=1) - embed_sq_sum)  # (B,)
            logits = linear_sum + inter[:, None] * inter_w[None, :] + bias[None, :]

            # Softmax cross-entropy gradient
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B                     # (B, C)

            # Gradients for bias and inter_w
            grad_bias   = np.sum(grad_logits, axis=0)
            grad_inter_w = np.sum(grad_logits * inter[:, None], axis=0)

            # Gradient for inter scalar
            grad_inter = np.sum(grad_logits * inter_w[None, :], axis=1)  # (B,)

            # Gradients for linear and embed (per feature)
            linear_grad = np.zeros_like(linear)
            embed_grad  = np.zeros_like(embed)
            E = embed_sum
            for j in range(d):
                vals = xb[:, j]
                # linear
                np.add.at(linear_grad[j], vals, grad_logits)
                # embed
                e_j = embed[j, vals]
                grad_e = grad_inter[:, None] * (E - e_j)
                np.add.at(embed_grad[j], vals, grad_e)

            # L2 regularization
            linear_grad += L2 * linear
            embed_grad  += L2 * embed
            grad_bias   += L2 * bias
            grad_inter_w += L2 * inter_w

            # Adam updates
            t += 1
            adam_update(linear,  linear_grad,  m_linear,  v_linear,  t)
            adam_update(embed,   embed_grad,   m_embed,   v_embed,   t)
            adam_update(bias,    grad_bias,    m_bias,    v_bias,    t)
            adam_update(inter_w, grad_inter_w, m_inter,   v_inter,   t)

    # Write weights.bin
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<4i", MAGIC, d, C, K))
        f.write(linear.astype("<f4").tobytes())
        f.write(embed.astype("<f4").tobytes())
        f.write(bias.astype("<f4").tobytes())
        f.write(inter_w.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (d=14). Output: 1 byte class index (0/1).
 * Weights format: magic 0x464D4C31, d, C, K; float32 arrays:
   linear[d][256][C], embed[d][256][K], bias[C], inter_w[C] */
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
    int d = hd[1], C = hd[2], K = hd[3];
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }

    long expected = 16 + 4L * ((long)d * 256 * C + (long)d * 256 * K + C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *embed  = linear + (long)d * 256 * C;
    const float *bias   = embed  + (long)d * 256 * K;
    const float *inter_w = bias + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float linear_sum[2], embed_sum[8], logits[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        linear_sum[0] = linear_sum[1] = 0.0f;
        for (int kk = 0; kk < K; kk++) embed_sum[kk] = 0.0f;
        float embed_sq_sum = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_ptr = linear + ((long)j * 256 + v) * C;
            const float *emb_ptr = embed  + ((long)j * 256 + v) * K;
            linear_sum[0] += lin_ptr[0];
            linear_sum[1] += lin_ptr[1];
            for (int kk = 0; kk < K; kk++) {
                float e = emb_ptr[kk];
                embed_sum[kk] += e;
                embed_sq_sum += e * e;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += embed_sum[kk] * embed_sum[kk];
        inter = 0.5f * (inter - embed_sq_sum);

        logits[0] = linear_sum[0] + inter * inter_w[0] + bias[0];
        logits[1] = linear_sum[1] + inter * inter_w[1] + bias[1];

        unsigned char best = (logits[1] > logits[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```