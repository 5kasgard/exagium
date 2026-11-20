```hypothesis
Factorization Machine (degree-2) with 8-dimensional embeddings captures pairwise feature interactions crucial for tabular data like Adult, where combinations of categorical values (e.g., education×occupation) are predictive. It adds only O(d·k) inference work (d=14, k=8) over linear models, keeping CPU cost low. If dev accuracy does not exceed label-linear by >0.5%, the interaction modeling is insufficient for this dataset.
```
```python
"""Factorization Machine (degree-2) for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: int32 magic 0x464D3231, d, k, C; float32 bias, linear[d][256], embed[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "FM binary only"
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = (rng.uniform(-1, 1, (d, 256, K)) * math.sqrt(6.0 / (256 + K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(embed)
    v_emb = np.zeros_like(embed)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            lin_part = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (B,)
            emb_sum = np.sum(embed[np.arange(d), xb.T], axis=0)            # (B, K)
            emb_sum_sq = np.sum(embed[np.arange(d), xb.T] ** 2, axis=0)    # (B, K)
            inter = 0.5 * np.sum(emb_sum ** 2 - emb_sum_sq, axis=1)        # (B,)
            score = lin_part + inter
            p = sigmoid(score)
            grad_score = (p - yb).astype(np.float32)                       # (B,)

            # Gradients
            g_bias = np.sum(grad_score)
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(embed)

            # For each feature, accumulate gradients
            for j in range(d):
                vals = xb[:, j]  # (B,)
                # linear grad
                np.add.at(g_lin[j], vals, grad_score)
                # embedding grad: grad_score[:, None] * (emb_sum - embed[j, vals])
                diff = emb_sum - embed[j, vals]  # (B, K)
                np.add.at(g_emb[j], vals, grad_score[:, None] * diff)

            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * embed

            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            m_hat = m_bias / (1 - 0.9 ** t)
            v_hat = v_bias / (1 - 0.999 ** t)
            bias -= LR * m_hat / (math.sqrt(v_hat) + 1e-8)

            # linear
            m_lin = 0.9 * m_lin + 0.1 * g_lin
            v_lin = 0.999 * v_lin + 0.001 * g_lin * g_lin
            m_hat_lin = m_lin / (1 - 0.9 ** t)
            v_hat_lin = v_lin / (1 - 0.999 ** t)
            linear -= LR * m_hat_lin / (np.sqrt(v_hat_lin) + 1e-8)

            # embed
            m_emb = 0.9 * m_emb + 0.1 * g_emb
            v_emb = 0.999 * v_emb + 0.001 * g_emb * g_emb
            m_hat_emb = m_emb / (1 - 0.9 ** t)
            v_hat_emb = v_emb / (1 - 0.999 ** t)
            embed -= LR * m_hat_emb / (np.sqrt(v_hat_emb) + 1e-8)

    # Pack weights
    # header: magic, d, k, C
    header = struct.pack("<4i", 0x464D3231, d, K, C)
    # bias (1), linear (d*256), embed (d*256*K) all float32 little-endian
    body = np.array([bias], dtype=np.float32).tobytes()
    body += linear.astype("<f4").tobytes()
    body += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree-2) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (d from header). Output: one byte class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0 || d > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * (1L + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *emb_sum = calloc(K, sizeof(float));
    float *emb_sum_sq = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias[0];
        memset(emb_sum, 0, K * sizeof(float));
        memset(emb_sum_sq, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += linear[(long)j * 256 + v];
            const float *e = embed + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float ev = e[kk];
                emb_sum[kk] += ev;
                emb_sum_sq[kk] += ev * ev;
            }
        }
        for (int kk = 0; kk < K; kk++) {
            score += 0.5f * (emb_sum[kk] * emb_sum[kk] - emb_sum_sq[kk]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(emb_sum);
    free(emb_sum_sq);
    return 0;
}
```