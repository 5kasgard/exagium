```hypothesis
Factorization Machines (FM) model pairwise feature interactions via low-rank embeddings, which are known to excel on tabular data like Adult where feature combinations (e.g., education × occupation) are predictive. With a small embedding dimension (k=4) the model stays compact and fast: 14 features × 256 values × 2 classes × (1 bias + 4 embedding) ≈ 143 KB weights. Expect dev accuracy >0.8524 (label-linear) at similar or lower CPU cost. If accuracy ≤0.8524 or instructions/item exceed label-linear by >20%, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: each of the 14 bytes is a categorical value (0..255). We learn per-feature per-value
a bias and a k-dim embedding for each class. Logit_c = sum_j bias[j,x_j,c] + 0.5*sum_d((sum_j e_{j,x_j,c,d})^2 - sum_j e_{j,x_j,c,d}^2).
Adam lr=3e-3, L2=1e-4, minibatch=64, epochs=min(100, ceil(600000/n)). Deterministic seed=1.
Weights format: int32 magic=0x464D4C31, d, k, C; float32 bias[d][256][C]; float32 emb[d][256][C][k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    rng = np.random.default_rng(SEED)

    # parameters: bias[d, 256, C], emb[d, 256, C, K]
    bias = np.zeros((d, 256, C), dtype=np.float32)
    emb = (rng.normal(0, 1.0 / math.sqrt(K), (d, 256, C, K)).astype(np.float32))

    params = [bias, emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = xb.shape[0]

            # forward
            linear = np.zeros((B, C), dtype=np.float32)
            sum_e = np.zeros((B, C, K), dtype=np.float32)
            sum_e_sq = np.zeros((B, C, K), dtype=np.float32)
            for j in range(d):
                v = xb[:, j]  # (B,)
                linear += bias[j, v]  # (B, C)
                e = emb[j, v]  # (B, C, K)
                sum_e += e
                sum_e_sq += e * e
            interaction = 0.5 * np.sum(sum_e * sum_e - sum_e_sq, axis=2)  # (B, C)
            logits = linear + interaction  # (B, C)

            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)  # (B, C)

            # backward
            grad_bias = np.zeros_like(bias)
            grad_emb = np.zeros_like(emb)

            for c in range(C):
                g_lin = grad_logits[:, c]  # (B,)
                g_inter = grad_logits[:, c]  # (B,)
                for j in range(d):
                    v = xb[:, j]
                    # bias gradient
                    np.add.at(grad_bias[j, :, c], v, g_lin)
                    # embedding gradient
                    e_j = emb[j, v, c]  # (B, K)
                    # d(interaction)/d(e_j) = g_inter * (sum_e[:,c] - e_j)
                    grad_e = g_inter[:, None] * (sum_e[:, c] - e_j)  # (B, K)
                    np.add.at(grad_emb[j, :, c], v, grad_e)

            # L2 regularization
            grad_bias += L2 * bias
            grad_emb += L2 * emb

            # Adam update
            grads = [grad_bias, grad_emb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    out = header + bias.astype("<f4").tobytes() + emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * d * 256 * C * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *emb = bias + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = malloc(sizeof(float) * C * K);
    float *sum_e_sq = malloc(sizeof(float) * C * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            logits[c] = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                sum_e[c * K + kk] = 0.0f;
                sum_e_sq[c * K + kk] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                logits[c] += bias[((long)j * 256 + v) * C + c];
                const float *e = emb + (((long)j * 256 + v) * C + c) * K;
                for (int kk = 0; kk < K; kk++) {
                    float val = e[kk];
                    sum_e[c * K + kk] += val;
                    sum_e_sq[c * K + kk] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float se = sum_e[c * K + kk];
                inter += 0.5f * (se * se - sum_e_sq[c * K + kk]);
            }
            logits[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(sum_e); free(sum_e_sq); free(logits);
    return 0;
}
```