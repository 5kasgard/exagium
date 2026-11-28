```hypothesis
Family: label-fm (Factorization Machine with small latent dimension).
Why: Adult benefits from pairwise feature interactions (e.g., education×occupation). A lightweight FM adds O(d·k) inference cost (d=14, k=8) while capturing interactions. It should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at near-linear speed.
Failure: If dev accuracy ≤ 0.8524 or Ir/item > 1000, the model is not worth the complexity.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encoding: each of the d columns is a byte (0..255). We learn per-column per-value
  linear weights (d × 256 × C) and FM embeddings (d × 256 × C × k).
Exported binary format (little-endian):
  uint32 magic 0x464D0001, int32 d, int32 k, int32 C,
  float32 linear_weights[d*256*C], float32 linear_bias[C],
  float32 fm_emb[d*256*C*k]  (layout: [d][256][C][k])."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K = 8  # latent dimension, chosen on tuning split


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # parameters
    lin_w = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    lin_b = np.zeros(C, np.float32)
    fm_emb = rng.normal(0, 0.01, (d, 256, C, K)).astype(np.float32)

    # Adam state
    params = [lin_w, lin_b, fm_emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(idx):
        xb = X[idx]  # (B, d)
        B = xb.shape[0]
        # linear logits
        logits = lin_b + np.sum(lin_w[np.arange(d), xb.T, :], axis=0).T  # (B, C)
        # FM part per class
        for c in range(C):
            # sum of embeddings and sum of squares across features
            emb = fm_emb[np.arange(d), xb.T, c, :]  # (d, B, K)
            sum_emb = emb.sum(axis=0)               # (B, K)
            sum_sq = (emb * emb).sum(axis=0)        # (B, K)
            fm = 0.5 * np.sum(sum_emb * sum_emb - sum_sq, axis=1)  # (B,)
            logits[:, c] += fm
        return logits

    def softmax_cross_entropy(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(len(yy)), yy] + 1e-12).mean()
        grad = probs.copy()
        grad[np.arange(len(yy)), yy] -= 1.0
        grad /= len(yy)
        return loss, grad.astype(np.float32)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            logits = forward(idx)
            loss, g = softmax_cross_entropy(logits, y[idx])

            # gradients
            B = len(idx)
            xb = X[idx]  # (B, d)
            # linear gradients
            glin_w = np.zeros_like(lin_w)
            glin_b = g.sum(axis=0)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(glin_w[j], vals, g)

            # FM gradients
            gfm = np.zeros_like(fm_emb)
            for c in range(C):
                emb = fm_emb[np.arange(d), xb.T, c, :]  # (d, B, K)
                sum_emb = emb.sum(axis=0)               # (B, K)
                # dL/d(sum_emb) = g[:,c][:,None] * sum_emb
                d_sum = g[:, c][:, None] * sum_emb      # (B, K)
                # dL/d(emb) = d_sum - g[:,c][:,None] * emb
                for j in range(d):
                    vals = xb[:, j]
                    # d_sum broadcast to (B, K) added to each feature's gradient
                    grad_emb = d_sum - g[:, c][:, None] * emb[j]  # (B, K)
                    np.add.at(gfm[j, vals, c, :], np.arange(B)[:, None], grad_emb)

            grads = [glin_w, glin_b, gfm]
            # L2 regularization
            for p, gr in zip(params, grads):
                gr += L2 * p

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    magic = 0x464D0001
    header = struct.pack("<4i", magic, d, K, C)
    body = lin_w.astype("<f4").tobytes() + lin_b.astype("<f4").tobytes() + fm_emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (one per column). Output: one byte class index.
 * Weights format: magic 0x464D0001, int32 d, int32 k, int32 C,
 *   float32 lin_w[d*256*C], float32 lin_b[C], float32 fm_emb[d*256*C*k]. */
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
    if (d < 1 || d > 256 || K < 1 || K > 64 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L
        + 4L * d * 256 * C          // lin_w
        + 4L * C                    // lin_b
        + 4L * d * 256 * C * K;     // fm_emb
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *lin_w = (const float *)(hd + 4);
    const float *lin_b = lin_w + (long)d * 256 * C;
    const float *fm_emb = lin_b + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *sum_emb = malloc(sizeof(float) * C * K);
    float *sum_sq  = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // linear part
        for (int c = 0; c < C; c++) logits[c] = lin_b[c];
        for (int j = 0; j < d; j++) {
            const float *w = lin_w + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }

        // FM part
        for (int c = 0; c < C; c++) {
            float *se = sum_emb + (long)c * K;
            float *ss = sum_sq  + (long)c * K;
            for (int kk = 0; kk < K; kk++) { se[kk] = 0.0f; ss[kk] = 0.0f; }
        }
        for (int j = 0; j < d; j++) {
            const float *emb = fm_emb + ((long)j * 256 + xi[j]) * C * K;
            for (int c = 0; c < C; c++) {
                const float *ec = emb + (long)c * K;
                float *se = sum_emb + (long)c * K;
                float *ss = sum_sq  + (long)c * K;
                for (int kk = 0; kk < K; kk++) {
                    float v = ec[kk];
                    se[kk] += v;
                    ss[kk] += v * v;
                }
            }
        }
        for (int c = 0; c < C; c++) {
            float *se = sum_emb + (long)c * K;
            float *ss = sum_sq  + (long)c * K;
            float fm = 0.0f;
            for (int kk = 0; kk < K; kk++) fm += se[kk] * se[kk] - ss[kk];
            logits[c] += 0.5f * fm;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```