```hypothesis
Family: label-fm (Factorization Machine with shared interactions)
Reason: Adult has 14 mixed-type features; pairwise interactions capture non-linearities missed by linear models.
A small embedding size (k=8) adds ~36k parameters (vs ~7k for linear) but computes interactions in O(d*k) via the FM trick.
If dev accuracy does not exceed label-linear by ≥0.5% (≈0.857), the extra complexity is not justified.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weight format: magic 0x464D3132, int32 d,k,C; float32 bias[C], linear[d,256,C], embed[d,256,k]."""
import json, math, struct, sys

import numpy as np

K = 8
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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # parameters
    bias = np.zeros(C, np.float32)
    linear = (rng.normal(0, 0.01, (d, 256, C)) * 0).astype(np.float32)  # zero init for linear
    embed = (rng.normal(0, 0.01, (d, 256, K)) * math.sqrt(2.0 / K)).astype(np.float32)

    # Adam state
    params = [bias, linear, embed]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = len(yb)

            # gather embeddings and linear weights for this batch
            emb_batch = embed[np.arange(d)[:, None], xb.T]          # (d, B, K)
            lin_batch = linear[np.arange(d)[:, None], xb.T]         # (d, B, C)

            # forward
            linear_part = lin_batch.sum(axis=0)                     # (B, C)
            sum_emb = emb_batch.sum(axis=0)                         # (B, K)
            sum_sq = (emb_batch ** 2).sum(axis=0)                   # (B, K)
            interaction = 0.5 * ((sum_emb ** 2 - sum_sq).sum(axis=1))  # (B,)
            scores = bias + linear_part + interaction[:, None]      # (B, C)

            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B                                 # (B, C)

            # gradients
            dbias = grad_scores.sum(axis=0)                         # (C,)
            dlinear_part = grad_scores                              # (B, C)
            dinteraction = grad_scores.sum(axis=1)                  # (B,)

            # scatter linear gradients
            dlinear = np.zeros_like(linear)
            for i in range(d):
                np.add.at(dlinear[i], xb[:, i], dlinear_part)

            # embedding gradients: d_interaction/d_emb = (sum_emb - emb_batch) * dinteraction
            dembed = np.zeros_like(embed)
            for i in range(d):
                diff = sum_emb - emb_batch[i]                       # (B, K)
                grad_emb_i = diff * dinteraction[:, None]           # (B, K)
                np.add.at(dembed[i], xb[:, i], grad_emb_i)

            # L2 regularization
            dlinear += L2 * linear
            dembed += L2 * embed

            grads = [dbias, dlinear, dembed]

            # Adam update
            t += 1
            for p, gr, (m, v) in zip(params, grads, state):
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    magic = 0x464D3132
    out = struct.pack("<4i", magic, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class index.
 * Weight layout: magic, d, k, C; float32 bias[C], linear[d][256][C], embed[d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *embed = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // linear part
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *w = linear + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        // FM interaction
        memset(sum_emb, 0, sizeof(float) * K);
        memset(sum_sq, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) {
                float val = e[u];
                sum_emb[u] += val;
                sum_sq[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) {
            float diff = sum_emb[u] * sum_emb[u] - sum_sq[u];
            inter += 0.5f * diff;
        }
        for (int c = 0; c < C; c++) scores[c] += inter;

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```