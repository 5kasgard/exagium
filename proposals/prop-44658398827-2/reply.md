```hypothesis
Family: label-fm (factorization machine with shared embeddings and output projection).
Why: Captures pairwise interactions between context positions via low-rank embeddings, improving over linear model while keeping inference fast (O(d*K + K*C) per item). The tinyshakespeare task benefits from modeling character co-occurrences.
Failure: If dev accuracy <= 0.3375 (label-linear) or CPU cost per item exceeds label-linear by >2x.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Model: score_c = bias_c + sum_j W_lin[j, x_j, c] + 0.5 * ( (sum_j V[j, x_j])^2 - sum_j V[j, x_j]^2 ) @ W_out[:, c]
Exported as: magic 0x464D4C31, d, K, C; float32 bias[C], W_lin[d*256, C], V[d*256, K], W_out[K, C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters
K = 16
SEED = 1
LR = 0.01
L2 = 1e-5
BATCH = 128
MAX_EPOCHS = 20
SAMPLES = 600000


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
    assert d == 16

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.zeros(C, dtype=np.float32)
    W_lin = rng.normal(0, 0.01, size=(d * 256, C)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(d * 256, K)).astype(np.float32)
    W_out = rng.normal(0, 0.01, size=(K, C)).astype(np.float32)

    params = [bias, W_lin, V, W_out]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            linear_sum = np.zeros((B, C), dtype=np.float32)
            sumV = np.zeros((B, K), dtype=np.float32)
            sumV2 = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                idxs = xb[:, j]                     # (B,)
                base = j * 256
                linear_sum += W_lin[base + idxs]    # (B, C)
                vj = V[base + idxs]                 # (B, K)
                sumV += vj
                sumV2 += vj * vj

            interaction = 0.5 * (sumV * sumV - sumV2)          # (B, K)
            scores = bias + linear_sum + interaction @ W_out   # (B, C)

            # Softmax + cross-entropy gradient
            max_scores = scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dscores = probs / B                               # (B, C)

            # Backward
            dbias = dscores.sum(axis=0)                       # (C,)
            dW_out = interaction.T @ dscores                  # (K, C)
            dinteraction = dscores @ W_out.T                  # (B, K)

            dsumV = dinteraction * sumV                       # (B, K)
            dsumV2 = -0.5 * dinteraction                      # (B, K)

            dW_lin = np.zeros_like(W_lin)
            dV = np.zeros_like(V)

            for j in range(d):
                idxs = xb[:, j]
                base = j * 256
                dW_lin[base + idxs] += dscores
                vj = V[base + idxs]
                dvj = dsumV + 2.0 * vj * dsumV2
                dV[base + idxs] += dvj

            # L2 regularization
            dW_lin += L2 * W_lin
            dV += L2 * V
            dW_out += L2 * W_out
            # no L2 on bias

            grads = [dbias, dW_lin, dV, dW_out]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    MAGIC = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (d != 16 || C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K + (long)K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *W_lin = bias + C;
    const float *V = W_lin + (long)d * 256 * C;
    const float *W_out = V + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *linear_sum = malloc(sizeof(float) * C);
    float *sumV = malloc(sizeof(float) * K);
    float *sumV2 = malloc(sizeof(float) * K);
    float *interaction = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) linear_sum[c] = 0.0f;
        for (int kk = 0; kk < K; kk++) { sumV[kk] = 0.0f; sumV2[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            long base = (long)j * 256 + byte;
            const float *wlin = W_lin + base * C;
            const float *v = V + base * K;
            for (int c = 0; c < C; c++) linear_sum[c] += wlin[c];
            for (int kk = 0; kk < K; kk++) {
                float val = v[kk];
                sumV[kk] += val;
                sumV2[kk] += val * val;
            }
        }

        for (int kk = 0; kk < K; kk++)
            interaction[kk] = 0.5f * (sumV[kk] * sumV[kk] - sumV2[kk]);

        for (int c = 0; c < C; c++) scores[c] = bias[c] + linear_sum[c];
        for (int kk = 0; kk < K; kk++) {
            float ival = interaction[kk];
            const float *wout = W_out + (long)kk * C;
            for (int c = 0; c < C; c++) scores[c] += ival * wout[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(linear_sum); free(sumV); free(sumV2); free(interaction); free(scores);
    if (x) free(x);
    return 0;
}
```