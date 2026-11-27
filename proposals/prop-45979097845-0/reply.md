```hypothesis
Family: label-fm (factorization machine with 8-dimensional embeddings).
Why: Adult has mixed categorical/ordinal features; FM captures pairwise interactions cheaply via low-rank embeddings, improving over linear while staying faster than GBDT/MLP.
Failure: If dev accuracy <= 0.8524 (linear) or inference cost > 2000 Ir/item, the model does not meet the goal.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x464D4C31, d, K, C; float32 bias, W_lin[d][256], V[d][256][K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 256
SAMPLES = 600000
MAX_EPOCHS = 100
K = 8  # embedding dimension


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
    # Parameters
    W_lin = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_W = np.zeros_like(W_lin); v_W = np.zeros_like(W_lin)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    m_b = 0.0; v_b = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch = X[idx]
            y_batch = y[idx]
            B = len(idx)

            lin_sum = np.zeros(B, dtype=np.float32)
            sum_v = np.zeros((B, K), dtype=np.float32)
            sum_v2 = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                vals = X_batch[:, j]
                lin_sum += W_lin[j, vals]
                v_j = V[j, vals]
                sum_v += v_j
                sum_v2 += v_j * v_j

            inter = 0.5 * (np.sum(sum_v * sum_v, axis=1) - np.sum(sum_v2, axis=1))
            logits = bias + lin_sum + inter

            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logits = (p - y_batch).astype(np.float32)

            grad_W = np.zeros_like(W_lin)
            grad_V = np.zeros_like(V)
            grad_b = np.sum(grad_logits)

            for j in range(d):
                vals = X_batch[:, j]
                v_j = V[j, vals]
                np.add.at(grad_W[j], vals, grad_logits)
                factor = grad_logits[:, np.newaxis] * (sum_v - v_j)
                np.add.at(grad_V[j], vals, factor)

            grad_W += L2 * W_lin
            grad_V += L2 * V

            t += 1
            # Update W_lin
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W_lin -= LR * m_hat_W / (np.sqrt(v_hat_W) + eps)

            # Update V
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= LR * m_hat_V / (np.sqrt(v_hat_V) + eps)

            # Update bias
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            bias -= LR * m_hat_b / (np.sqrt(v_hat_b) + eps)

    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias = (const float *)(hd + 4);
    const float *W_lin = bias + 1;
    const float *V = W_lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float lin_sum = *bias;
        for (int j = 0; j < K; j++) { sum_v[j] = 0.0f; sum_v2[j] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin_sum += W_lin[(long)j * 256 + v];
            const float *emb = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = emb[kk];
                sum_v[kk] += val;
                sum_v2[kk] += val * val;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            inter += sum_v[kk] * sum_v[kk] - sum_v2[kk];
        }
        inter *= 0.5f;
        float logit = lin_sum + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```