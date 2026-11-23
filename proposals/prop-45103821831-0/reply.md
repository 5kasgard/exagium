```hypothesis
Family: label-fm (factorization machine, order 2, rank 8).  
Why: Adult has strong pairwise interactions (e.g., education×occupation, marital-status×relationship) that a linear model misses. A low-rank FM captures all pairwise feature interactions with only ~30k extra parameters, staying far cheaper than GBDT while being more expressive than an MLP of similar size. Training with Adam is deterministic and fits the 20 min/8 GiB budget.  
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost ≥ 2325 Ir/item (label-mlp16), the family does not justify its complexity.
```

```python
"""Factorization Machine (order 2, rank k) for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: each of the d=14 bytes is treated as a categorical with 256 values (0..255).
Model: logit = w0 + Σ_i w_i[x_i] + ½( (Σ_i v_i[x_i])² - Σ_i v_i[x_i]² ).
Exported as: magic(0x464D4C31), d, k, C; float32 w0, w[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 20
K = 8          # embedding rank
MAGIC = 0x464D4C31  # 'FML1'

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
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx].astype(np.float32)  # (b,)
            b = xb.shape[0]

            # gather embeddings and linear weights
            # V_gather: (b, d, K), w_gather: (b, d)
            V_gather = np.stack([V[i, xb[:, i]] for i in range(d)], axis=1)
            w_gather = np.stack([w[i, xb[:, i]] for i in range(d)], axis=1)

            sum_v = V_gather.sum(axis=1)          # (b, K)
            sum_v_sq = (V_gather ** 2).sum(axis=1) # (b, K)
            pairwise = 0.5 * ((sum_v ** 2).sum(axis=1) - (sum_v_sq).sum(axis=1))  # (b,)

            logit = w0 + w_gather.sum(axis=1) + pairwise  # (b,)
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)  # (b,)

            # gradients
            g_w0 = grad.mean() + L2 * w0
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)

            # linear weights gradient
            for i in range(d):
                np.add.at(g_w[i], xb[:, i], grad)  # (256,)
            g_w /= b
            g_w += L2 * w

            # V gradient: for each sample, grad * (sum_v - v_i) for each i
            # vectorized: (b, d, K)
            diff = sum_v[:, None, :] - V_gather  # (b, d, K)
            g_V_gather = (grad[:, None, None] * diff).mean(axis=0)  # (d, K)
            for i in range(d):
                # accumulate per category
                np.add.at(g_V[i], xb[:, i], g_V_gather[i])  # Wait, g_V_gather[i] is (K,) but we need to add to each category?
                # Actually g_V_gather is already averaged over batch and indexed by feature i, but we need per-category gradients.
                # Let's re-derive: For each sample, gradient for V[i, xi] is grad * (sum_v - V[i, xi]).
                # We can compute per-category by scattering.
            # The above is wrong; we need to scatter per category. Let's do a loop over features for clarity and correctness.
            # Since d=14 and K=8, a loop is fine.

            # Recompute g_V properly:
            g_V = np.zeros_like(V)
            for i in range(d):
                # For each feature i, we have batch indices xi = xb[:, i] (b,)
                # and per-sample diff_i = sum_v - V_gather[:, i, :]  (b, K)
                diff_i = sum_v - V_gather[:, i, :]  # (b, K)
                # grad_i = grad[:, None] * diff_i   (b, K)
                grad_i = grad[:, None] * diff_i
                # scatter-add into g_V[i] at positions xi
                np.add.at(g_V[i], xb[:, i], grad_i)
            g_V /= b
            g_V += L2 * V

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (g_w0 * g_w0)
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * (g_w * g_w)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * (g_V * g_V)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Write weights
    # Flatten in order: w0, w (d*256), V (d*256*K)
    flat = np.concatenate([
        np.array([w0], dtype=np.float32),
        w.ravel(),
        V.ravel()
    ])
    header = struct.pack("<4i", MAGIC, d, K, C)
    out = header + flat.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (order 2) inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0 or 1).
 * Weights format: magic(0x464D4C31), d, k, C; float32 w0, w[d*256], V[d*256*k]. */
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K <= 0 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = (float *)malloc(sizeof(float) * K);
    float *sum_v_sq = (float *)malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float acc = w0;
        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            acc += w[(long)j * 256 + v];
            const float *vec = V + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = vec[f];
                sum_v[f] += val;
                sum_v_sq[f] += val * val;
            }
        }
        float pairwise = 0.0f;
        for (int f = 0; f < K; f++) pairwise += sum_v[f] * sum_v[f] - sum_v_sq[f];
        acc += 0.5f * pairwise;

        unsigned char best = (acc > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    free(sum_v);
    free(sum_v_sq);
    return 0;
}
```