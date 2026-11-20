```hypothesis
We propose a Factorization Machine family "label-fm". FMs model all pairwise feature interactions via low-rank latent vectors (K=8), which are known to be crucial for tabular data like Adult (e.g., education×occupation). This adds only O(d·K) ≈ 112 FLOPs/item over a linear model, far cheaper than GBDT/MLP, while often matching their accuracy. If dev accuracy does not exceed label-linear's 0.8524 or inference cost exceeds ~2000 Ir/item, the hypothesis fails.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D3132, d, K, C; float32 bias, w[d*256], V[d*256*K]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (would be chosen on tuning split, then fixed)
K = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 30
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

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

    # Parameters: w[d,256], V[d,256,K], bias
    w = np.zeros((d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.float32(0.0), np.float32(0.0)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            linear = bias + np.sum(w[np.arange(d), xb.T], axis=0)  # (B,)
            sum_V = np.zeros((B, K), dtype=np.float32)
            sum_sq_V = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                vj = V[j, xb[:, j]]          # (B, K)
                sum_V += vj
                sum_sq_V += vj * vj
            interaction = 0.5 * np.sum(sum_V * sum_V - sum_sq_V, axis=1)
            score = linear + interaction
            prob = 1.0 / (1.0 + np.exp(-score))
            grad_score = prob - yb            # (B,)  dL/dscore

            # Gradients
            grad_w = np.zeros_like(w)
            grad_V = np.zeros_like(V)
            grad_bias = np.float32(0.0)

            # bias
            grad_bias = np.float32(np.sum(grad_score))

            # w and V
            for j in range(d):
                vals = xb[:, j]               # (B,)
                # w gradient: scatter add grad_score to w[j, vals]
                np.add.at(grad_w[j], vals, grad_score)
                # V gradient: for each sample, (sum_V - V[j,val]) * grad_score
                vj = V[j, vals]               # (B, K)
                diff = sum_V - vj              # (B, K)
                # grad_V[j, vals, k] += grad_score[:,None] * diff[:,k]
                # Use loop over K for scatter add
                for k in range(K):
                    np.add.at(grad_V[j, :, k], vals, grad_score * diff[:, k])

            # L2 regularization
            grad_w += L2 * w
            grad_V += L2 * V
            # bias not regularized

            # Adam update
            t += 1
            # w
            m_w = BETA1 * m_w + (1 - BETA1) * grad_w
            v_w = BETA2 * v_w + (1 - BETA2) * (grad_w * grad_w)
            m_w_hat = m_w / (1 - BETA1 ** t)
            v_w_hat = v_w / (1 - BETA2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + EPS)

            # V
            m_V = BETA1 * m_V + (1 - BETA1) * grad_V
            v_V = BETA2 * v_V + (1 - BETA2) * (grad_V * grad_V)
            m_V_hat = m_V / (1 - BETA1 ** t)
            v_V_hat = v_V / (1 - BETA2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + EPS)

            # bias
            m_b = BETA1 * m_b + (1 - BETA1) * grad_bias
            v_b = BETA2 * v_b + (1 - BETA2) * (grad_bias * grad_bias)
            m_b_hat = m_b / (1 - BETA1 ** t)
            v_b_hat = v_b / (1 - BETA2 ** t)
            bias -= LR * m_b_hat / (np.sqrt(v_b_hat) + EPS)

    # Write weights.bin
    magic = 0x464D3132  # "FM12"
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index 0 or 1).
 * Weights format: magic 0x464D3132, int32 d, K, C; float32 bias, w[d*256], V[d*256*K]. */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16 + 4L + 4L * (long)d * 256 + 4L * (long)d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *w = bias + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = (float *)malloc(sizeof(float) * K);
    float *sum_sq_V = (float *)malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = *bias;
        for (int j = 0; j < K; j++) { sum_V[j] = 0.0f; sum_sq_V[j] = 0.0f; }

        for (int j = 0; j < d; j++) {
            uint8_t val = xi[j];
            long idx = (long)j * 256 + val;
            linear += w[idx];
            const float *vec = V + (long)idx * K;
            for (int kk = 0; kk < K; kk++) {
                float v = vec[kk];
                sum_V[kk] += v;
                sum_sq_V[kk] += v * v;
            }
        }

        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            interaction += sum_V[kk] * sum_V[kk] - sum_sq_V[kk];
        }
        interaction *= 0.5f;

        float score = linear + interaction;
        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }

    free(sum_V); free(sum_sq_V);
    return 0;
}
```