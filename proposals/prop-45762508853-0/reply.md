```hypothesis
Factorization Machine with class-specific embeddings (label-fm) captures pairwise feature interactions crucial for tabular data like Adult, where combinations (education×occupation, marital-status×relationship) are predictive. With embedding dim k=8 it adds ~2× the parameters of label-linear but models interactions in O(d·k) time. It should exceed 0.8524 dev accuracy; if it matches or underperforms linear, the interaction hypothesis fails for this quantization.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Each of the d input bytes is treated as a categorical feature with 256 values.
Model: per-class bias, linear weights W_c[f,v], and embeddings V_c[f,v] (dim k).
Score_c = bias_c + sum_f W_c[f,x_f] + 0.5 * sum_l[(sum_f V_c[f,x_f]_l)^2 - sum_f V_c[f,x_f]_l^2].
Exported as: magic 0x464D4C31, d, k, C; then per-class bias, W_c[d*256], V_c[d*256*k] (float32 LE)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-5, 64, 600000, 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters per class: bias (C), W (C, d, 256), V (C, d, 256, K)
    bias = np.zeros(C, np.float32)
    W = np.zeros((C, d, 256), np.float32)
    V = rng.normal(0, 0.01, (C, d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            # linear_c[b] = bias_c + sum_f W_c[f, xb[f]]
            linear = bias.reshape(1, C) + np.sum(W[:, np.arange(d), xb.T], axis=1).T  # (B, C)
            # interaction_c[b] = 0.5 * sum_l[ (sum_f V_c[f,xb[f],l])^2 - sum_f V_c[f,xb[f],l]^2 ]
            # V_c[f, xb[f]] -> (C, B, d, K)
            V_gather = V[:, np.arange(d), xb.T]  # (C, d, B, K)
            V_gather = np.transpose(V_gather, (0, 2, 1, 3))  # (C, B, d, K)
            sum_V = V_gather.sum(axis=2)           # (C, B, K)
            sum_V_sq = (V_gather ** 2).sum(axis=2) # (C, B, K)
            interaction = 0.5 * ((sum_V ** 2 - sum_V_sq).sum(axis=2)).T  # (B, C)

            scores = linear + interaction  # (B, C)
            # Softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (B, C)

            # Gradient dL/dscores = probs - one_hot(y)
            grad_scores = probs.copy()
            grad_scores[np.arange(B), yb] -= 1.0  # (B, C)
            grad_scores = grad_scores.astype(np.float32) / B  # average over batch

            # Backward
            # bias grad: sum_b grad_scores[b,c]
            g_bias = grad_scores.sum(axis=0)  # (C,)

            # W grad: for each f, scatter-add grad_scores[b,c] to W_c[f, xb[b,f]]
            g_W = np.zeros_like(W)
            for f in range(d):
                vals = xb[:, f]  # (B,)
                for c in range(C):
                    np.add.at(g_W[c, f], vals, grad_scores[:, c])

            # V grad: for each c, f, b: grad_scores[b,c] * (sum_V[c,b] - V_c[f, xb[b,f]])
            g_V = np.zeros_like(V)
            for c in range(C):
                for f in range(d):
                    vals = xb[:, f]  # (B,)
                    # sum_V[c] is (B, K)
                    diff = sum_V[c] - V_gather[c, :, f, :]  # (B, K)
                    # grad_scores[:,c] is (B,), broadcast to (B,K)
                    g = (grad_scores[:, c:c+1] * diff).astype(np.float32)  # (B, K)
                    np.add.at(g_V[c, f], vals, g)

            # Adam update
            t += 1
            # bias
            m_bias = BETA1 * m_bias + (1 - BETA1) * g_bias
            v_bias = BETA2 * v_bias + (1 - BETA2) * g_bias * g_bias
            m_hat = m_bias / (1 - BETA1 ** t)
            v_hat = v_bias / (1 - BETA2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # W
            m_W = BETA1 * m_W + (1 - BETA1) * g_W
            v_W = BETA2 * v_W + (1 - BETA2) * g_W * g_W
            m_hat = m_W / (1 - BETA1 ** t)
            v_hat = v_W / (1 - BETA2 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # V
            m_V = BETA1 * m_V + (1 - BETA1) * g_V
            v_V = BETA2 * v_V + (1 - BETA2) * g_V * g_V
            m_hat = m_V / (1 - BETA1 ** t)
            v_hat = v_V / (1 - BETA2 ** t)
            V -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # L2 weight decay (applied after Adam step as in label-linear)
            bias *= (1 - LR * L2)
            W *= (1 - LR * L2)
            V *= (1 - LR * L2)

    # Pack weights
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    out = bytearray(header)
    for c in range(C):
        out += bias[c:c+1].astype("<f4").tobytes()
        out += W[c].astype("<f4").tobytes()
        out += V[c].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weight format: magic 0x464D4C31, d, k, C; then per-class bias, W[d*256], V[d*256*k] (float32 LE). */
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
    if (d != 14 || K != 8 || C != 2) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L + 4L * C * (1L + (long)d * 256 * (1L + K));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *W = bias + C;
    const float *V = W + C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_V[2][8], sum_V_sq[2][8];
    float linear[2], score[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            linear[c] = bias[c];
            for (int l = 0; l < K; l++) {
                sum_V[c][l] = 0.0f;
                sum_V_sq[c][l] = 0.0f;
            }
        }

        for (int f = 0; f < d; f++) {
            unsigned char v = xi[f];
            for (int c = 0; c < C; c++) {
                linear[c] += W[c * d * 256 + f * 256 + v];
                const float *Vcf = V + c * d * 256 * K + f * 256 * K + v * K;
                for (int l = 0; l < K; l++) {
                    float val = Vcf[l];
                    sum_V[c][l] += val;
                    sum_V_sq[c][l] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int l = 0; l < K; l++) {
                float s = sum_V[c][l];
                inter += s * s - sum_V_sq[c][l];
            }
            score[c] = linear[c] + 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```