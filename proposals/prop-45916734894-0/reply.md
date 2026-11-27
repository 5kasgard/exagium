```hypothesis
Factorization Machine (label-fm) captures pairwise feature interactions via low-rank embeddings, which are known to be predictive on tabular data like Adult (e.g., education × occupation). With embedding rank k=8, it adds only O(d*k) parameters and compute over a linear model. Expect dev accuracy > 0.8524 (label-linear) at similar latency. If accuracy does not improve, interactions are not beneficial or rank is too low.
```

```python
"""Factorization Machine for tabular label tasks: fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Model: bias + linear per feature-value + pairwise interactions via k-dim embeddings.
Binary classification: sigmoid(score) -> class 1 if score>0 else 0.
Exported weights: magic 0x464D4C31, d, k, C=2; float32 bias, linear[d*256], embed[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 8          # embedding rank
MAGIC = 0x464D4C31  # "FML1"

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

    # Parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            bsize = xb.shape[0]

            # Forward
            # linear term
            lin_vals = linear[np.arange(d), xb.T].T   # (b, d) via advanced indexing
            linear_sum = lin_vals.sum(axis=1)         # (b,)

            # interaction term
            emb_vals = embed[np.arange(d), xb.T]      # (d, b, k) -> transpose to (b, d, k)
            emb_vals = np.transpose(emb_vals, (1, 0, 2))
            sum_v = emb_vals.sum(axis=1)              # (b, k)
            sum_v_sq = (emb_vals ** 2).sum(axis=1)    # (b, k)
            interaction = 0.5 * ((sum_v ** 2 - sum_v_sq).sum(axis=1))  # (b,)

            logits = bias + linear_sum + interaction   # (b,)
            # Sigmoid and loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logits = (p - yb).astype(np.float32)  # (b,)

            # Gradients
            g_bias = grad_logits.sum()

            g_linear = np.zeros_like(linear)
            g_embed = np.zeros_like(embed)

            # For each feature, accumulate gradients via bincount
            for j in range(d):
                vals = xb[:, j]                 # (b,)
                # linear gradient
                g_linear[j] = np.bincount(vals, weights=grad_logits, minlength=256)
                # embedding gradient
                emb_j = emb_vals[:, j, :]       # (b, k)
                sum_v_j = sum_v                 # (b, k)
                factor = grad_logits[:, None] * (sum_v_j - emb_j)  # (b, k)
                for f in range(K):
                    g_embed[j, :, f] = np.bincount(vals, weights=factor[:, f], minlength=256)

            # L2 regularization
            g_linear += L2 * linear
            g_embed += L2 * embed

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * (g_bias * g_bias)
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_linear
            v_lin = beta2 * v_lin + (1 - beta2) * (g_linear * g_linear)
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # embed
            m_emb = beta1 * m_emb + (1 - beta1) * g_embed
            v_emb = beta2 * v_emb + (1 - beta2) * (g_embed * g_embed)
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            embed -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    # Header: magic, d, k, C
    header = struct.pack("<4i", MAGIC, d, K, C)
    # bias (float32), linear (d*256 float32), embed (d*256*K float32)
    body = struct.pack("<f", bias) + linear.astype("<f4").tobytes() + embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1).
 * Weights format: int32 magic(0x464D4C31), d, k, C; float32 bias, linear[d*256], embed[d*256*k]. */
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2) { fprintf(stderr, "expected 2 classes\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[16], sum_v_sq[16];  // K <= 16

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias;
        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            score += linear[(long)j * 256 + val];
            const float *e = embed + ((long)j * 256 + val) * K;
            for (int f = 0; f < K; f++) {
                float v = e[f];
                sum_v[f] += v;
                sum_v_sq[f] += v * v;
            }
        }
        for (int f = 0; f < K; f++) {
            score += 0.5f * (sum_v[f] * sum_v[f] - sum_v_sq[f]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```