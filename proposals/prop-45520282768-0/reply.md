```hypothesis
Family: label-fm (Factorization Machine). FMs capture pairwise feature interactions via low-rank embeddings, which are crucial for tabular data with mixed categorical/ordinal features. With embedding dim K=4, inference adds only O(d*K) operations per item, keeping CPU cost near linear models. If dev accuracy does not exceed label-linear's 0.8524, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D4C31, d, K, C; float32 bias[C], linear[d][256][C], embed[d][256][K][C]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 600000, 100


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

    # parameters
    bias = np.zeros(C, np.float32)
    linear = (rng.normal(0, 0.01, (d, 256, C)) * 0.01).astype(np.float32)
    embed = (rng.normal(0, 0.01, (d, 256, K, C)) * 0.01).astype(np.float32)

    params = [bias, linear, embed]
    shapes = [p.shape for p in params]
    # Adam state
    m = [np.zeros(s, np.float32) for s in shapes]
    v = [np.zeros(s, np.float32) for s in shapes]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            bsize = len(yb)

            # forward
            # linear part: (b, C)
            lin_part = bias + np.sum(linear[np.arange(d)[:, None, None], xb.T, :], axis=0)  # (b, C)
            # interaction part
            # embeddings for each feature: (b, d, K, C)
            emb = embed[np.arange(d)[:, None, None], xb.T]  # (d, b, K, C) -> transpose to (b, d, K, C)
            emb = np.transpose(emb, (1, 0, 2, 3))
            sum_emb = np.sum(emb, axis=1)        # (b, K, C)
            sum_emb_sq = np.sum(emb * emb, axis=1)
            inter = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)  # (b, C)

            logits = lin_part + inter  # (b, C)

            # softmax
            max_logit = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # gradients
            grad_logits = probs.copy()
            grad_logits[np.arange(bsize), yb] -= 1.0
            grad_logits /= bsize

            # bias grad
            g_bias = np.sum(grad_logits, axis=0) + L2 * bias

            # linear grad: (d, 256, C)
            g_linear = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                for c in range(C):
                    np.add.at(g_linear[j, :, c], vals, grad_logits[:, c])
            g_linear += L2 * linear

            # embed grad: (d, 256, K, C)
            g_embed = np.zeros_like(embed)
            # sum_emb is (b, K, C), we need to multiply grad_logits (b, C) with sum_emb
            # d/dv_{j,v,k,c} = grad_logits[:,c] * (sum_emb[:,k,c] - v_{j,v,k,c})
            for j in range(d):
                vals = xb[:, j]
                for c in range(C):
                    g = grad_logits[:, c]  # (b,)
                    for k in range(K):
                        # sum_emb[:, k, c] is (b,)
                        diff = sum_emb[:, k, c] - embed[j, vals, k, c]
                        # g * diff added to g_embed[j, vals, k, c]
                        np.add.at(g_embed[j, :, k, c], vals, g * diff)
            g_embed += L2 * embed

            grads = [g_bias, g_linear, g_embed]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # pack weights
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    body = bias.astype("<f4").tobytes()
    body += linear.astype("<f4").tobytes()
    body += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }

    long total_floats = (long)C + (long)d * 256 * C + (long)d * 256 * K * C;
    if (nw != 16 + 4 * total_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *embed = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K * C);
    float *sum_emb_sq = malloc(sizeof(float) * K * C);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_ptr = linear + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += lin_ptr[c];
        }

        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                sum_emb[c * K + kk] = 0.0f;
                sum_emb_sq[c * K + kk] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *emb_ptr = embed + ((long)j * 256 + v) * K * C;
            for (int c = 0; c < C; c++) {
                for (int kk = 0; kk < K; kk++) {
                    float e = emb_ptr[c * K + kk];
                    sum_emb[c * K + kk] += e;
                    sum_emb_sq[c * K + kk] += e * e;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float s = sum_emb[c * K + kk];
                float sq = sum_emb_sq[c * K + kk];
                inter += 0.5f * (s * s - sq);
            }
            scores[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(sum_emb); free(sum_emb_sq); free(scores);
    if (x) free(x);
    return 0;
}
```