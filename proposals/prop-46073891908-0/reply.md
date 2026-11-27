```hypothesis
Family: label-fm (factorization machine with pairwise interactions).
Why: Adult has 14 mixed ordinal/categorical fields; FM captures field–field interactions (e.g., education×occupation) that linear models miss, while staying fast (O(d²k) with d=14, k=4). The model is trained with Adam on logistic loss, L2 regularized, deterministic.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost (Ir/item) exceeds label-linear by >20%, the model does not meet the goal.
```

```python
"""Factorization Machine for tabular label tasks (label-fm).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format: magic 0x464D4C31, d, k, C; float32 w0, linear[d*256], emb[d*256*k] (all little-endian)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
K = 4            # embedding dimension for pairwise interactions
LR = 0.01        # Adam learning rate
L2 = 1e-4        # L2 regularization
BATCH = 256      # minibatch size
EPOCHS = 30      # training epochs
SEED = 1         # random seed

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2, "FM implementation assumes binary classification"

    # Load data
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    w0 = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Precompute indices for vectorized gathering
    feat_idx = np.arange(d, dtype=np.int64)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            Xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = Xb.shape[0]

            # Gather linear weights and embeddings for this batch
            # linear_part: (B,)
            lin_vals = linear[feat_idx[:, None], Xb.T].sum(axis=0)  # sum over features
            # embeddings: (B, d, K)
            E = emb[feat_idx[:, None], Xb.T].transpose(1, 0, 2)

            # Pairwise interaction: 0.5 * sum_l ( (sum_j E_{j,l})^2 - sum_j E_{j,l}^2 )
            sum_u = E.sum(axis=1)                 # (B, K)
            sum_sq = (E * E).sum(axis=1)          # (B, K)
            pair_part = 0.5 * ((sum_u * sum_u - sum_sq).sum(axis=1))  # (B,)

            score = w0 + lin_vals + pair_part
            p = 1.0 / (1.0 + np.exp(-score))
            g = p - yb.astype(np.float32)         # gradient of loss w.r.t. score (B,)

            # Gradients
            # w0
            gw0 = g.sum()
            # linear weights: for each feature j, add g[b] to linear[j, Xb[b,j]]
            # We'll accumulate per feature using a loop over features (d=14 small)
            glin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(glin[j], Xb[:, j], g)
            # embeddings: grad_E[b, j, l] = g[b] * (sum_u[b, l] - E[b, j, l])
            gemb = g[:, None, None] * (sum_u[:, None, :] - E)   # (B, d, K)
            # Sum over batch
            gemb_sum = gemb.sum(axis=0)                         # (d, K)

            # Add L2 regularization
            gw0 += L2 * w0
            glin += L2 * linear
            gemb_sum += L2 * emb

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (gw0 * gw0)
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * glin
            v_lin = beta2 * v_lin + (1 - beta2) * (glin * glin)
            m_lin_hat = m_lin / (1 - beta1 ** t)
            v_lin_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_lin_hat / (np.sqrt(v_lin_hat) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * gemb_sum
            v_emb = beta2 * v_emb + (1 - beta2) * (gemb_sum * gemb_sum)
            m_emb_hat = m_emb / (1 - beta1 ** t)
            v_emb_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_emb_hat / (np.sqrt(v_emb_hat) + eps)

    # Write weights.bin
    # Header: magic, d, K, C (int32 little-endian)
    header = struct.pack("<4i", 0x464D4C31, d, K, C)
    # Floats: w0, linear (d*256), emb (d*256*K) all as float32 little-endian
    floats = np.concatenate([
        np.array([w0], dtype=np.float32),
        linear.ravel(),
        emb.ravel()
    ]).astype("<f4")
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(floats.tobytes())

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label-fm: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=14). Output is one byte: class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K <= 0 || K > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = 1L + (long)d * 256 * (1 + K);
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *floats = (const float *)(hd + 4);
    const float w0 = floats[0];
    const float *linear = floats + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        // Linear part
        for (int j = 0; j < d; j++) {
            score += linear[(long)j * 256 + xi[j]];
        }
        // Pairwise interactions: 0.5 * sum_l ( (sum_j emb[j][xi[j]][l])^2 - sum_j emb[j][xi[j]][l]^2 )
        float sum_u[16] = {0};
        float sum_sq[16] = {0};
        for (int j = 0; j < d; j++) {
            const float *emb_j = emb + ((long)j * 256 + xi[j]) * K;
            for (int l = 0; l < K; l++) {
                float v = emb_j[l];
                sum_u[l] += v;
                sum_sq[l] += v * v;
            }
        }
        for (int l = 0; l < K; l++) {
            score += 0.5f * (sum_u[l] * sum_u[l] - sum_sq[l]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```