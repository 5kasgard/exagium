```hypothesis
Factorization Machine (FM) with rank 8 captures pairwise feature interactions via low-rank embeddings, improving over linear models on tabular data with mixed ordinal/categorical features. Inference costs O(d·k) ≈ 112 multiply-adds per item (d=14, k=8), far below MLP cost. If dev accuracy ≤ 0.8524 or Ir/item > 1000, the model does not meet the goal.
```

```python
"""Factorization Machine (FM) for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x464D3130, d, k, then float32 bias, linear[d][256], embed[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: bias, linear[d][256], embed[d][256][K]
    bias = np.float32(0.0)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    embed = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    t = 0

    # Convert y to -1/+1 for gradient derivation
    y_pm = 2.0 * y - 1.0  # -1 for class 0, +1 for class 1

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y_pm[idx]
            bs = len(yb)

            # Forward pass
            score = np.full(bs, bias, dtype=np.float32)
            sum_v = np.zeros((bs, K), dtype=np.float32)
            sum_v2 = np.zeros((bs, K), dtype=np.float32)
            for j in range(d):
                xj = xb[:, j]
                score += linear[j, xj]
                v = embed[j, xj]  # (bs, K)
                sum_v += v
                sum_v2 += v * v
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
            score += inter

            # Logistic loss gradient: dL/dscore = sigmoid(score) - (y+1)/2
            # but with y in {-1,1}: p = 1/(1+exp(-y*score)), dL/dscore = -y * p
            # Equivalently: dL/dscore = sigmoid(score) - t where t in {0,1}
            t_vals = (yb + 1) * 0.5  # 0 or 1
            p = 1.0 / (1.0 + np.exp(-score))
            grad_score = (p - t_vals).astype(np.float32)  # (bs,)

            # Backward pass
            g_bias = np.sum(grad_score)
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(embed)

            for j in range(d):
                xj = xb[:, j]
                # linear gradient
                np.add.at(g_lin[j], xj, grad_score)
                # embedding gradient
                v = embed[j, xj]  # (bs, K)
                # d(inter)/dv = sum_v - v
                dv = sum_v - v  # (bs, K)
                # multiply by grad_score per example
                dv *= grad_score[:, None]
                np.add.at(g_emb[j], xj, dv)

            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * embed

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # embed
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            embed -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    out = struct.pack("<3i", 0x464D3130, d, K)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (0 or 1). */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 12 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2];
    if (d <= 0 || d > 256 || K <= 0 || K > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 12 + 4 + (size_t)d * 256 * 4 + (size_t)d * 256 * K * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(wb + 12);
    float bias = *p++;                     // bias
    const float *linear = p; p += (size_t)d * 256;
    const float *embed = p;                // embed[d][256][K]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias;
        memset(sum_v, 0, sizeof(float) * K);
        memset(sum_v2, 0, sizeof(float) * K);

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            score += linear[(size_t)j * 256 + v];
            const float *e = embed + ((size_t)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                sum_v[f] += val;
                sum_v2[f] += val * val;
            }
        }
        for (int f = 0; f < K; f++) score += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);

        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```