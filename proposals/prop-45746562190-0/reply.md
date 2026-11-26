```hypothesis
Family: label-fm (factorization machine with shared embeddings).
Why: FMs model pairwise feature interactions via low-rank embeddings, which are crucial for tabular data with mixed categorical/ordinal features. A small latent dimension (k=4) keeps inference fast (O(d*k)) while capturing non-linear effects that linear models miss.
Failure condition: Dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ 2325 (label-mlp16).
```

```python
"""Factorization machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Encodes each of the d bytes as a categorical value (0-255).
Model: score_c = bias_c + sum_f linear[f][x_f][c] + 0.5 * ( (sum_f emb[f][x_f])^2 - sum_f emb[f][x_f]^2 )
Exported as: int32 magic 0x464D4C31, d, k, C; float32 bias[C], linear[d][256][C], emb[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 4                 # latent dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters
    bias = np.zeros(C, dtype=np.float32)
    linear = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    params = [bias, linear, emb]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Linear part: (B, C)
            lin_scores = bias + np.sum(linear[np.arange(d)[:, None], xb.T, :], axis=0).T  # (B, C)
            # Embedding part: sum_f emb[f][x_f] -> (B, K)
            emb_sum = np.sum(emb[np.arange(d)[:, None], xb.T, :], axis=0)                # (B, K)
            emb_sq_sum = np.sum(emb[np.arange(d)[:, None], xb.T, :] ** 2, axis=0)        # (B, K)
            interaction = 0.5 * np.sum(emb_sum ** 2 - emb_sq_sum, axis=1, keepdims=True)  # (B, 1)
            scores = lin_scores + interaction                                             # (B, C)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (B, C)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B  # (B, C)

            # Gradients
            # bias
            g_bias = grad_scores.sum(axis=0).astype(np.float32)
            # linear
            g_linear = np.zeros_like(linear)
            for f in range(d):
                vals = xb[:, f]
                for c in range(C):
                    g_linear[f, vals, c] += grad_scores[:, c]
            g_linear = g_linear.astype(np.float32)
            # emb
            g_emb = np.zeros_like(emb)
            # Interaction gradient w.r.t. emb_sum: dL/d(emb_sum) = interaction_grad * emb_sum
            # interaction_grad = sum_c grad_scores[:, c]  (since interaction added to all classes)
            inter_grad = grad_scores.sum(axis=1, keepdims=True)  # (B, 1)
            # dL/d(emb_f) = inter_grad * (emb_sum - emb_f)
            for f in range(d):
                vals = xb[:, f]
                emb_f = emb[f, vals]  # (B, K)
                # (B, K) * (B, 1) -> (B, K)
                grad_emb_f = inter_grad * (emb_sum - emb_f)
                for k in range(K):
                    g_emb[f, vals, k] += grad_emb_f[:, k]
            g_emb = g_emb.astype(np.float32)

            # Add L2
            g_bias += L2 * bias
            g_linear += L2 * linear
            g_emb += L2 * emb

            grads = [g_bias, g_linear, g_emb]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic 0x464D4C31, d, k, C; float32 bias[C], linear[d][256][C], emb[d][256][k]. */
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
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * K);
    float *emb_sq_sum = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int f = 0; f < d; f++) {
            const float *lin_f = linear + ((long)f * 256 + xi[f]) * C;
            for (int c = 0; c < C; c++) scores[c] += lin_f[c];
        }

        // Embedding sum and square sum
        for (int kk = 0; kk < K; kk++) {
            emb_sum[kk] = 0.0f;
            emb_sq_sum[kk] = 0.0f;
        }
        for (int f = 0; f < d; f++) {
            const float *emb_f = emb + ((long)f * 256 + xi[f]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = emb_f[kk];
                emb_sum[kk] += v;
                emb_sq_sum[kk] += v * v;
            }
        }
        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            interaction += emb_sum[kk] * emb_sum[kk] - emb_sq_sum[kk];
        }
        interaction *= 0.5f;
        for (int c = 0; c < C; c++) scores[c] += interaction;

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(emb_sq_sum); free(scores);
    return 0;
}
```