```hypothesis
Factorization Machines (FM) capture pairwise feature interactions via low-rank embeddings, which are crucial for tabular data with mixed categorical/ordinal features. On Adult, interactions like (education, occupation) or (marital-status, sex) are known to be predictive. FM with k=8 adds only ~14*256*8=28672 parameters, keeping inference fast (O(d*k)) and memory low. If dev accuracy does not exceed 0.855 (label-linear + 0.0026) or inference cost exceeds label-linear by >2x, the family does not justify its complexity.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Format: int32 magic 0x464D2D31, d, k, C; float32 bias; float32 linear[d][256]; float32 emb[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 256, 600000, 20


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: bias, linear[d][256], emb[d][256][K]
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.uniform(-0.01, 0.01, (d, 256, K)) * np.sqrt(6.0 / (256 + K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Linear term: bias + sum_j linear[j, x_j]
            lin_term = np.full(B, bias, dtype=np.float32)
            for j in range(d):
                lin_term += linear[j, xb[:, j]]

            # FM interaction term: 0.5 * sum_f ( (sum_j v_{j,x_j,f})^2 - sum_j v_{j,x_j,f}^2 )
            # Compute sum_j v_{j,x_j,f} for each f
            sum_v = np.zeros((B, K), dtype=np.float32)
            sum_v2 = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                v = emb[j, xb[:, j], :]      # (B, K)
                sum_v += v
                sum_v2 += v * v
            inter_term = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)

            logits = lin_term + inter_term
            probs = sigmoid(logits)

            # Gradient of logistic loss: probs - y (for class 1), but we have 0/1 labels
            # Loss = -[y*log(p) + (1-y)*log(1-p)]; dL/dlogit = p - y
            grad_logit = (probs - yb).astype(np.float32)   # (B,)

            # Bias gradient
            g_bias = grad_logit.mean()

            # Linear gradients
            g_lin = np.zeros_like(linear)
            for j in range(d):
                # For each feature value, accumulate grad_logit
                np.add.at(g_lin[j], xb[:, j], grad_logit)
            g_lin /= B
            g_lin += L2 * linear

            # Embedding gradients
            g_emb = np.zeros_like(emb)
            for j in range(d):
                v = emb[j, xb[:, j], :]          # (B, K)
                # d(inter)/dv = grad_logit[:, None] * (sum_v - v)
                factor = grad_logit[:, None] * (sum_v - v)   # (B, K)
                np.add.at(g_emb[j], xb[:, j], factor)
            g_emb /= B
            g_emb += L2 * emb

            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # Bias update
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # Linear update
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # Embedding update
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    magic = 0x464D2D31  # "FM-1" little-endian
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D2D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1 || d > 256 || K > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = bias;
        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                sum_v[f] += val;
                sum_v2[f] += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float sv = sum_v[f];
            inter += 0.5f * (sv * sv - sum_v2[f]);
        }

        float logit = lin + inter;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sum_v);
    free(sum_v2);
    return 0;
}
```