```hypothesis
Factorization Machine (FM) on raw fields: each of the 14 columns is a categorical field with 256 values (ordinal columns are already quantized to 255 bins). A field-aware FM with embedding dimension k=8 captures all pairwise interactions with ~130k parameters. It should beat linear (0.8524) by modeling interactions while staying faster than MLP (2325 Ir/item) because inference is O(d^2*k) ≈ 728 multiply-adds per record. If dev accuracy ≤ 0.8524 or Ir/item > 1000, the model does not justify its complexity.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each of the d columns as a field with 256 values. Learns a bias per field-value and an embedding vector of size k.
Score = sum_j bias[j, x_j] + sum_{j1<j2} <emb[j1, x_j1], emb[j2, x_j2]>.
Binary classification: predict 1 if score > 0 else 0.
Exported weights: int32 magic=0x464D0001, d, k, C=2; then float32 bias[d*256], emb[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 8
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 10

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
    rng = np.random.default_rng(SEED)

    # Use last 20% as tuning split to pick K? We fix K=8 per hypothesis.
    n = X.shape[0]
    tune_start = int(n * 0.8)

    # Initialize parameters
    bias = np.zeros((d, 256), dtype=np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias = np.zeros_like(bias); v_bias = np.zeros_like(bias)
    m_emb = np.zeros_like(emb); v_emb = np.zeros_like(emb)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            # bias term
            bias_scores = np.sum(bias[np.arange(d), xb.T], axis=0)  # (B,)
            # pairwise interaction
            inter = np.zeros(B, dtype=np.float32)
            for j1 in range(d):
                e1 = emb[j1, xb[:, j1]]   # (B, K)
                for j2 in range(j1 + 1, d):
                    e2 = emb[j2, xb[:, j2]]   # (B, K)
                    inter += np.sum(e1 * e2, axis=1)
            scores = bias_scores + inter
            # Logistic loss gradients: dL/dscore = -y / (1 + exp(y*score)) with y in {-1,1}
            y_sign = 2 * yb - 1  # 0->-1, 1->+1
            z = y_sign * scores
            sig = 1.0 / (1.0 + np.exp(-z))
            grad_score = -(y_sign * (1 - sig))  # (B,)

            # Gradients
            g_bias = np.zeros_like(bias)
            g_emb = np.zeros_like(emb)

            # bias gradient
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_bias[j], vals, grad_score)

            # embedding gradient
            for j1 in range(d):
                e1 = emb[j1, xb[:, j1]]   # (B, K)
                for j2 in range(j1 + 1, d):
                    e2 = emb[j2, xb[:, j2]]
                    g = grad_score[:, None]  # (B,1)
                    # d/demb[j1] = g * e2
                    np.add.at(g_emb[j1], xb[:, j1], g * e2)
                    # d/demb[j2] = g * e1
                    np.add.at(g_emb[j2], xb[:, j2], g * e1)

            # L2 regularization
            g_bias += L2 * bias
            g_emb += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in [(bias, g_bias, m_bias, v_bias), (emb, g_emb, m_emb, v_emb)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights
    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per field). Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *emb = bias + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = 0.0f;
        for (int j = 0; j < d; j++) {
            score += bias[j * 256 + xi[j]];
        }
        for (int j1 = 0; j1 < d; j1++) {
            const float *e1 = emb + (j1 * 256 + xi[j1]) * K;
            for (int j2 = j1 + 1; j2 < d; j2++) {
                const float *e2 = emb + (j2 * 256 + xi[j2]) * K;
                for (int f = 0; f < K; f++) score += e1[f] * e2[f];
            }
        }
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```