```hypothesis
Factorization Machine (FM) with k=4 latent factors captures pairwise feature interactions crucial for Adult
(education×occupation, marital×relationship) while retaining O(d·k) inference speed. It extends linear with
field-aware embeddings per feature value (all 14 columns treated as categorical 0-255). If dev accuracy ≤0.8524
(label-linear), interactions are not learnable at this rank or the linear signal dominates.
```
```python
"""Factorization Machine (2nd-order, k=4) for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
All 14 input bytes treated as categorical fields with 256 values. Model: bias + linear[14,256] + latent[14,256,k].
Adam lr=1e-2, L2=1e-4, batch=256, epochs=30. Exported as int32 magic 0x464D3132, d, k, C; float32 bias, linear[d*256], latent[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, EPOCHS = 1, 1e-2, 1e-4, 256, 30
MAGIC = 0x464D3132  # "FM12"

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
    assert C == 2, "FM binary only"
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # params: bias (1), linear (d*256), latent (d*256*K)
    bias = np.array([0.0], dtype=np.float32)
    linear = np.zeros(d * 256, dtype=np.float32)
    latent = (rng.normal(0, 0.01, (d * 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_lat, v_lat = np.zeros_like(latent), np.zeros_like(latent)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    idx = np.arange(n)
    for _ in range(EPOCHS):
        rng.shuffle(idx)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s + BATCH]
            xb = X[batch_idx]
            yb = y[batch_idx]
            bsz = len(yb)

            # forward: compute logit for each example
            # linear term: bias + sum_j linear[j, x_j]
            lin_term = bias[0] + linear[xb.ravel()].reshape(bsz, d).sum(axis=1)
            # interaction term: 0.5 * sum_f ( (sum_j v[j,x_j,f])^2 - sum_j v[j,x_j,f]^2 )
            lat = latent[xb.ravel()].reshape(bsz, d, K)  # (bsz, d, K)
            sum_v = lat.sum(axis=1)                      # (bsz, K)
            sum_v2 = (lat * lat).sum(axis=1)             # (bsz, K)
            inter = 0.5 * ((sum_v * sum_v - sum_v2).sum(axis=1))
            logit = lin_term + inter

            # gradient of logistic loss
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)  # (bsz,)

            # gradients
            g_bias = grad.mean()
            g_lin = np.zeros_like(linear)
            g_lat = np.zeros_like(latent)

            # linear grad: for each example and feature, add grad[i] to linear[j, x_ij]
            # use np.add.at for accumulation
            flat_idx = xb.ravel()
            np.add.at(g_lin, flat_idx, np.repeat(grad, d) / bsz)

            # latent grad: for each factor f
            # d/dv_ijf = grad_i * (sum_v[i,f] - v_ijf)
            for f in range(K):
                sv = sum_v[:, f]  # (bsz,)
                for j in range(d):
                    v_ijf = lat[:, j, f]  # (bsz,)
                    g = grad * (sv - v_ijf) / bsz  # (bsz,)
                    base = j * 256
                    np.add.at(g_lat, base + xb[:, j], g)

            # L2 regularization
            g_lin += L2 * linear
            g_lat += L2 * latent

            # Adam updates
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= LR * (m_bias / (1 - beta1**t)) / (np.sqrt(v_bias / (1 - beta2**t)) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1**t)) / (np.sqrt(v_lin / (1 - beta2**t)) + eps)
            # latent
            m_lat = beta1 * m_lat + (1 - beta1) * g_lat
            v_lat = beta2 * v_lat + (1 - beta2) * g_lat * g_lat
            latent -= LR * (m_lat / (1 - beta1**t)) / (np.sqrt(v_lat / (1 - beta2**t)) + eps)

    # pack weights: magic, d, K, C; bias[1], linear[d*256], latent[d*256*K]
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (k=4) inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1).
 * Weights format: int32 magic(0x464D3132), d, k, C; float32 bias[1], linear[d*256], latent[d*256*k]. */
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
    if (C != 2 || K != 4 || d != 14) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t expected = 16 + 4 * (1 + (size_t)d * 256 + (size_t)d * 256 * K);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *latent = linear + (size_t)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[4], sum_v2[4];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias[0];
        // linear term
        for (int j = 0; j < d; j++) logit += linear[j * 256 + xi[j]];
        // interaction term
        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *v = latent + ((size_t)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float vf = v[f];
                sum_v[f] += vf;
                sum_v2[f] += vf * vf;
            }
        }
        for (int f = 0; f < K; f++) logit += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);

        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```