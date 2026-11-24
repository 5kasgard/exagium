```hypothesis
Family: label-fm (factorization machine with shared embeddings and class-specific linear terms).
Why: Tabular data with mixed ordinal/categorical columns benefits from pairwise feature interactions. A factorization machine captures these with O(k*d) parameters and inference cost, where k=8 and d=14. It should beat label-linear (no interactions) and approach label-gbdt accuracy while being much faster than label-gbdt at inference.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear, the model does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Encoding: each (column, byte_value) gets a unique index. Shared embeddings (k=8) for interactions,
class-specific linear weights. Adam lr=0.01, L2=1e-4, batch=256, epochs=20.
Exported weights: magic 0x464D3130, d, k, C, then float32 arrays:
  bias[C], linear[d*256*C], emb[d*256*k] (row-major)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 20


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

    # Map each (col, value) to a flat index: col*256 + val
    V = d * 256

    # Parameters
    bias = np.zeros(C, np.float32)
    linear = np.zeros((V, C), np.float32)
    emb = (rng.normal(0, 0.01, (V, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    # Precompute indices for all examples
    idx = X.astype(np.int32) + np.arange(d, dtype=np.int32) * 256  # (n, d)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            bsz = len(batch_idx)
            idx_b = idx[batch_idx]          # (bsz, d)
            y_b = y[batch_idx]              # (bsz,)

            # Linear term: bias + sum_j linear[idx_j, :]
            lin_term = bias + np.sum(linear[idx_b], axis=1)  # (bsz, C)

            # Interaction term: 0.5 * ( (sum emb)^2 - sum emb^2 )
            emb_sum = np.sum(emb[idx_b], axis=1)             # (bsz, K)
            emb_sq_sum = np.sum(emb[idx_b] ** 2, axis=1)     # (bsz, K)
            inter = 0.5 * np.sum(emb_sum ** 2 - emb_sq_sum, axis=1, keepdims=True)  # (bsz, 1)

            logits = lin_term + inter                        # (bsz, C)

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), y_b] -= 1.0
            grad_logits = probs / bsz                         # (bsz, C)

            # Gradients
            grad_bias = grad_logits.sum(axis=0)               # (C,)
            grad_lin = np.zeros_like(linear)
            np.add.at(grad_lin, idx_b.reshape(-1), grad_logits.repeat(d, axis=0).reshape(-1, C))
            # grad_lin shape (V, C)

            # Embedding gradients
            # inter = 0.5 * ( (sum_u e_u)^2 - sum_u e_u^2 )
            # d inter / d e_i = (sum_u e_u) - e_i = emb_sum - e_i
            # For each example, each feature j, grad_emb[idx_j] += grad_logits_sum * (emb_sum - emb[idx_j])
            # grad_logits_sum is sum over classes of grad_logits (since inter is shared across classes)
            grad_inter = grad_logits.sum(axis=1, keepdims=True)  # (bsz, 1)
            # For each example and feature, we need to add grad_inter * (emb_sum - emb[idx])
            # Vectorized via np.add.at
            grad_emb = np.zeros_like(emb)
            # emb_sum per example: (bsz, K)
            # For each feature position j:
            for j in range(d):
                idx_j = idx_b[:, j]                 # (bsz,)
                emb_j = emb[idx_j]                  # (bsz, K)
                diff = emb_sum - emb_j              # (bsz, K)
                g = grad_inter * diff               # (bsz, K)
                np.add.at(grad_emb, idx_j, g)

            # L2 regularization
            grad_lin += L2 * linear
            grad_emb += L2 * emb

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1 - beta2) * grad_bias ** 2
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * grad_lin
            v_lin = beta2 * v_lin + (1 - beta2) * grad_lin ** 2
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * grad_emb ** 2
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: magic 0x464D3130, int32 d, k, C; float32 bias[C], linear[d*256*C], emb[d*256*k]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long V = (long)d * 256;
    long expected = 16 + 4L * (C + V * C + V * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + V * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear term
        for (int c = 0; c < C; c++) logits[c] = bias[c];
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            const float *lin_row = linear + (long)idx * C;
            for (int c = 0; c < C; c++) logits[c] += lin_row[c];
        }

        // Interaction term
        memset(emb_sum, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            const float *emb_row = emb + (long)idx * K;
            for (int u = 0; u < K; u++) emb_sum[u] += emb_row[u];
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) {
            float e = emb_sum[u];
            // We need sum of squares of embeddings for each feature
            // Compute on the fly: sum_j emb[idx_j][u]^2
            float sq_sum = 0.0f;
            for (int j = 0; j < d; j++) {
                int idx = j * 256 + xi[j];
                float v = emb[(long)idx * K + u];
                sq_sum += v * v;
            }
            inter += 0.5f * (e * e - sq_sum);
        }
        for (int c = 0; c < C; c++) logits[c] += inter;

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(emb_sum); free(logits);
    return 0;
}
```