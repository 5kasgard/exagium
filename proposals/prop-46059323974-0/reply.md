```hypothesis
Family: label-fm (Factorization Machine). Captures pairwise feature interactions via low-rank embeddings, which is effective for tabular data with mixed categorical/ordinal features. Should outperform the linear baseline at modest compute cost. Failure if dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-mlp16 (~2300 Ir/item).
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: int32 magic 0x464D4C31, d, k, C; float32 bias, W[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
EPOCHS = 20
BATCH = 256
K = 8          # embedding dimension
L2 = 1e-4      # L2 regularization
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "Binary classification only"

    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    bias = np.float32(0.0)
    W = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward pass
            # Linear part
            lin = bias + np.sum(W[np.arange(d), xb.T], axis=0)   # (B,)
            # Embeddings: gather V for each feature
            emb = V[np.arange(d), xb.T]   # (d, B, K)
            sum_v = np.sum(emb, axis=0)   # (B, K)
            sum_v2 = np.sum(emb * emb, axis=0)
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)   # (B,)
            score = lin + inter

            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-score))
            grad_score = (p - yb).astype(np.float32)   # (B,)

            # Gradients
            g_bias = np.mean(grad_score)
            g_W = np.zeros_like(W)
            g_V = np.zeros_like(V)

            # Linear weights gradient
            for f in range(d):
                vals = xb[:, f]
                np.add.at(g_W[f], vals, grad_score)
            g_W /= B

            # Embedding gradients
            # dL/dV_f = grad_score * (sum_v - V_f)
            for f in range(d):
                vals = xb[:, f]
                v_f = V[f, vals]          # (B, K)
                factor = grad_score[:, None] * (sum_v - v_f)   # (B, K)
                np.add.at(g_V[f], vals, factor)
            g_V /= B

            # L2 regularization
            g_W += L2 * W
            g_V += L2 * V

            # Adam update
            t += 1
            # bias
            m_bias = BETA1 * m_bias + (1 - BETA1) * g_bias
            v_bias = BETA2 * v_bias + (1 - BETA2) * g_bias * g_bias
            m_hat = m_bias / (1 - BETA1 ** t)
            v_hat = v_bias / (1 - BETA2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # W
            m_W = BETA1 * m_W + (1 - BETA1) * g_W
            v_W = BETA2 * v_W + (1 - BETA2) * g_W * g_W
            m_hat_W = m_W / (1 - BETA1 ** t)
            v_hat_W = v_W / (1 - BETA2 ** t)
            W -= LR * m_hat_W / (np.sqrt(v_hat_W) + EPS)

            # V
            m_V = BETA1 * m_V + (1 - BETA1) * g_V
            v_V = BETA2 * v_V + (1 - BETA2) * g_V * g_V
            m_hat_V = m_V / (1 - BETA1 ** t)
            v_hat_V = v_V / (1 - BETA2 ** t)
            V -= LR * m_hat_V / (np.sqrt(v_hat_V) + EPS)

    # Pack weights
    # Magic, d, k, C
    header = struct.pack("<4i", 0x464D4C31, d, K, C)
    # bias, W (d*256), V (d*256*k)
    body = struct.pack("<f", bias)
    body += W.astype("<f4").tobytes()
    body += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || k <= 0 || k > 64) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = (float *)malloc(sizeof(float) * k);
    float *sum_v2 = (float *)malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        float score = bias;
        // Linear part
        for (int f = 0; f < d; f++) {
            score += W[(long)f * 256 + xi[f]];
        }
        // Interaction part
        memset(sum_v, 0, sizeof(float) * k);
        memset(sum_v2, 0, sizeof(float) * k);
        for (int f = 0; f < d; f++) {
            const float *emb = V + ((long)f * 256 + xi[f]) * k;
            for (int j = 0; j < k; j++) {
                float v = emb[j];
                sum_v[j] += v;
                sum_v2[j] += v * v;
            }
        }
        for (int j = 0; j < k; j++) {
            score += 0.5f * (sum_v[j] * sum_v[j] - sum_v2[j]);
        }

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_v); free(sum_v2);
    return 0;
}
```