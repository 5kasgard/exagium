```hypothesis
Family: label-fm (factorization machine with degree-2 interactions, factorized with k=8).
Why: Captures pairwise feature interactions efficiently (O(d*k) inference) which should improve over linear model on tabular data with mixed categorical/ordinal features.
Failure: If dev accuracy <= 0.8524 (linear) or inference cost > label-linear (719 Ir/item).
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Learns linear weights and k-dimensional embeddings per feature value.
Exported as float32: magic 0x464D3231, d, k, C, bias, linear[d][256], embed[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 10


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
    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(embed)
    v_emb = np.zeros_like(embed)
    t = 0

    n = X.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            # Forward
            lin_part = bias + np.sum(linear[np.arange(d)[:, None], xb.T], axis=0)  # (bs,)
            sum_vec = np.sum(embed[np.arange(d)[:, None], xb.T], axis=0)           # (bs, K)
            sum_sq = np.sum(embed[np.arange(d)[:, None], xb.T] ** 2, axis=0)       # (bs, K)
            inter = 0.5 * np.sum(sum_vec ** 2 - sum_sq, axis=1)                   # (bs,)
            logit = lin_part + inter
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)   # (bs,)

            # Gradients
            g_bias = np.mean(grad)
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(embed)

            # Vectorized gradient accumulation
            for i in range(bs):
                xi = xb[i]
                g = grad[i]
                g_lin[np.arange(d), xi] += g
                vec = embed[np.arange(d), xi]          # (d, K)
                sv = sum_vec[i]                        # (K,)
                g_emb[np.arange(d), xi] += g * (sv - vec)

            g_lin /= bs
            g_emb /= bs

            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * embed

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            embed -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    # Write weights
    out = struct.pack("<4i", 0x464D3231, d, K, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0/1). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3231) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ptr = (float *)(hd + 4);
    float bias = *ptr++;
    float *linear = ptr; ptr += d * 256;
    float *embed = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_vec[8], sum_sq[8];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = bias;
        for (int j = 0; j < K; j++) { sum_vec[j] = 0.0f; sum_sq[j] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin += linear[j * 256 + v];
            float *e = embed + (j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = e[u];
                sum_vec[u] += val;
                sum_sq[u] += val * val;
            }
        }

        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += 0.5f * (sum_vec[u] * sum_vec[u] - sum_sq[u]);
        float logit = lin + inter;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```