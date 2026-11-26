```hypothesis
Propose a factorization machine of order 2 (family "label-fm") with embedding dimension k=4.
It adds pairwise feature interactions to the linear model, which are crucial for tabular data like Adult,
while adding only O(d*k) parameters per class. The linear model (label-linear) cannot capture interactions;
FM should improve accuracy with a modest increase in inference cost (still well below label-gbdt).
If it does not beat label-linear on dev, it suggests interactions are not beneficial here or k=4 is too small.
```
```python
"""Factorization Machine (order 2) for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D3132, d, k, C; float32 linear[d][256][C], embed[d][256][C][k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on the tuning split: last 20% of train.in)
K = 4                # embedding dimension for interactions
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 30
SAMPLES = 600000     # max total samples seen (like label-linear)

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters: linear[d, 256, C], embed[d, 256, C, K]
    linear = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    embed  = (rng.normal(0, 0.01, (d, 256, C, K))).astype(np.float32)

    # Adam state
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed),  np.zeros_like(embed)

    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward pass
            scores = np.zeros((B, C), dtype=np.float32)
            sum_v = np.zeros((B, C, K), dtype=np.float32)
            sum_v_sq = np.zeros((B, C, K), dtype=np.float32)

            for j in range(d):
                vals = xb[:, j]                 # (B,)
                # linear contribution
                scores += linear[j, vals, :]    # (B, C)
                # embedding contribution
                v = embed[j, vals, :, :]        # (B, C, K)
                sum_v += v
                sum_v_sq += v * v

            inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=2)  # (B, C)
            scores += inter

            # Softmax cross-entropy
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grads = probs / B                     # dL/dscores (B, C)

            # Gradients for linear and embed
            grad_linear = np.zeros_like(linear)
            grad_embed  = np.zeros_like(embed)

            for j in range(d):
                vals = xb[:, j]                 # (B,)
                # linear gradient
                for c in range(C):
                    grad_linear[j, vals, c] += grads[:, c]
                # embed gradient: dL/dV = grads * (sum_v - v)
                v = embed[j, vals, :, :]        # (B, C, K)
                # sum_v per sample per class per factor already computed
                factor = (sum_v - v) * grads[:, :, np.newaxis]  # (B, C, K)
                for c in range(C):
                    grad_embed[j, vals, c, :] += factor[:, c, :]

            # L2 regularization
            grad_linear += L2 * linear
            grad_embed  += L2 * embed

            # Adam update
            t += 1
            for param, grad, m, v in [
                (linear, grad_linear, m_lin, v_lin),
                (embed,  grad_embed,  m_emb, v_emb)
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    magic = 0x464D3132  # "FM12"
    header = struct.pack("<4i", magic, d, K, C)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(linear.astype("<f4").tobytes())
        f.write(embed.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (order 2) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte: argmax class (first max wins).
 * Weight format: int32 magic(0x464D3132), d, k, C; float32 linear[d][256][C], embed[d][256][C][k]. */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * d * 256 * C * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *embed  = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v  = malloc(sizeof(float) * C * K);
    float *sum_v_sq = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            scores[c] = 0.0f;
            for (int f = 0; f < K; f++) {
                sum_v[c * K + f] = 0.0f;
                sum_v_sq[c * K + f] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_j = linear + ((long)j * 256 + v) * C;
            const float *emb_j = embed  + (((long)j * 256 + v) * C) * K;
            for (int c = 0; c < C; c++) {
                scores[c] += lin_j[c];
                for (int f = 0; f < K; f++) {
                    float e = emb_j[c * K + f];
                    sum_v[c * K + f] += e;
                    sum_v_sq[c * K + f] += e * e;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float s = sum_v[c * K + f];
                inter += s * s - sum_v_sq[c * K + f];
            }
            scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```