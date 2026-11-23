```hypothesis
Family: label-fm (Factorization Machine with binary cross-entropy).
Why: Adult has 14 byte-wide features mixing ordinal and categorical; a linear model on one-hot bins (label-linear) ignores pairwise interactions. A rank-8 FM adds O(d*k) interactions (d=14, k=8) at ~112 multiply-adds per item, capturing non-linear feature crosses without the cost of a hidden layer. Training uses the same Adam schedule as label-linear for determinism.
Failure signal: Dev accuracy ≤ 0.8524 (label-linear) or inference cost > 1500 Ir/item.
```

```python
"""Factorization Machine for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x464D4C31 ('FML1'), d, k, C (C=2)
  float32 bias[2]
  for each feature j in 0..d-1:
    float32 linear_weight[256][2]   # class 0 weight is kept 0, class 1 learned
    float32 embed[256][k][2]        # class 0 embeddings kept 0
Total floats = 2 + d*256*(2 + 2*k)"""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100
K = 8  # FM rank


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
    assert C == 2
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters: bias[2], linear[d, 256, 2], embed[d, 256, K, 2]
    # We only learn class 1; class 0 stays at 0.
    bias = np.zeros(2, np.float32)
    bias[1] = math.log((y == 1).mean() / (y == 0).mean())  # log-odds init
    linear = np.zeros((d, 256, 2), np.float32)
    embed = np.zeros((d, 256, K, 2), np.float32)
    embed[..., 1] = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    t = 0

    # Precompute feature values for speed
    X_idx = X  # (n, d) uint8

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_idx[idx]          # (B, d)
            yb = y[idx]              # (B,)
            B = len(yb)

            # Forward
            # Linear part: bias + sum_j linear[j, xb[:,j], 1]
            lin_part = bias[1] + np.sum(linear[np.arange(d), xb.T, 1], axis=0)  # (B,)

            # FM part: 0.5 * sum_f ( (sum_j v_jf)^2 - sum_j v_jf^2 )
            # embed[j, xb[:,j], :, 1] -> (B, d, K)
            v = embed[np.arange(d), xb.T, :, 1].transpose(1, 0, 2)  # (B, d, K)
            sum_v = v.sum(axis=1)                 # (B, K)
            sum_v2 = (v * v).sum(axis=1)          # (B, K)
            fm_part = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)  # (B,)

            logit = lin_part + fm_part            # (B,)
            # Sigmoid + BCE gradient
            p = 1.0 / (1.0 + np.exp(-logit))      # (B,)
            grad = (p - yb).astype(np.float32)    # (B,)

            # Backward
            # bias
            g_bias = np.array([0.0, grad.mean()], np.float32)
            # linear
            g_lin = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_lin[j, :, 1], vals, grad)
            g_lin /= B
            g_lin += L2 * linear

            # embed
            g_emb = np.zeros_like(embed)
            for j in range(d):
                vals = xb[:, j]
                vj = v[:, j, :]  # (B, K)
                # dL/dv = grad[:,None] * (sum_v - vj)
                dv = grad[:, None] * (sum_v - vj)  # (B, K)
                for f in range(K):
                    np.add.at(g_emb[j, :, f, 1], vals, dv[:, f])
            g_emb /= B
            g_emb += L2 * embed

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            def adam_step(param, m, v, g):
                m *= beta1; m += (1 - beta1) * g
                v *= beta2; v += (1 - beta2) * g * g
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

            adam_step(bias, m_bias, v_bias, g_bias)
            adam_step(linear, m_lin, v_lin, g_lin)
            adam_step(embed, m_emb, v_emb, g_emb)

    # Write weights
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1).
 * Weights format (little-endian):
 *   int32 magic=0x464D4C31, d, k, C (C=2)
 *   float32 bias[2]
 *   float32 linear[d][256][2]
 *   float32 embed[d][256][k][2] */
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
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    // Expected size: 16 + 4*(2 + d*256*2 + d*256*K*2)
    long expected = 16 + 4L * (2 + (long)d * 256 * 2 + (long)d * 256 * K * 2);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 2;
    const float *embed = linear + (long)d * 256 * 2;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part (class 1 only)
        float logit = bias[1];
        for (int j = 0; j < d; j++) {
            logit += linear[((long)j * 256 + xi[j]) * 2 + 1];
        }

        // FM part
        memset(sum_v, 0, sizeof(float) * K);
        memset(sum_v2, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * K * 2;
            // class 1 embeddings are at odd indices (index 1, 3, 5, ...)
            for (int f = 0; f < K; f++) {
                float v = e[f * 2 + 1];
                sum_v[f] += v;
                sum_v2[f] += v * v;
            }
        }
        for (int f = 0; f < K; f++) {
            logit += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        }

        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```