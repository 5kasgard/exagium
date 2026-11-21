```hypothesis
Factorization Machine (FM) with k=8 latent dimensions captures pairwise feature interactions
crucial for tabular data (e.g., education×occupation, marital-status×sex) while keeping
inference at ~14·k ops/item. It should beat label-linear's 0.8524 accuracy on Adult; if
dev accuracy ≤0.852 or Ir/item ≥1000, the approach fails for this task.
```
```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Encodes each column as 256-value categorical (255=unseen).
FM: score = w·x + 0.5[(Σ v_i)² - Σ ||v_i||²] with latent dim k=8. Adam lr=0.01, L2=1e-4, batch=256, epochs=min(50, ceil(300000/n)).
Exports: magic 0x464D3100, d, k, C=1; float32 w[d][256], v[d][256][k], v_norm_sq[d][256]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 300000, 50


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
    # Convert labels to -1/+1 for logistic loss
    y_pm = np.where(y == 0, -1.0, 1.0).astype(np.float32)

    d = X.shape[1]          # 14 columns
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # FM parameters
    w = np.zeros((d, 256), dtype=np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    mw, mv = np.zeros_like(w), np.zeros_like(v)
    vw, vv = np.zeros_like(w), np.zeros_like(v)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]          # (b, d)
            yi = y_pm[idx]       # (b,)

            # Forward
            # Linear term: sum_j w[j, xi[:,j]]
            linear = np.sum(w[np.arange(d), xi.T], axis=0)   # (b,)

            # Interaction: 0.5 * (||sum_j v[j, xi[:,j]]||^2 - sum_j ||v[j, xi[:,j]]||^2)
            # sum_v shape (b, K)
            sum_v = np.sum(v[np.arange(d), xi.T].transpose(1, 0, 2), axis=1)  # (b, K)
            sum_v_sq = np.sum(sum_v * sum_v, axis=1)                         # (b,)
            v_norm_sq = np.sum(v[np.arange(d), xi.T] ** 2, axis=(1, 2))      # (b,)
            interaction = 0.5 * (sum_v_sq - v_norm_sq)

            score = linear + interaction                       # (b,)
            # Logistic loss gradient: -y * sigmoid(-y * score)
            margin = -yi * score
            # sigmoid(margin) = 1 / (1 + exp(-margin))
            # Use stable computation
            z = np.maximum(-margin, 0)
            sigmoid = np.exp(-z) / (1 + np.exp(-np.abs(margin)))
            grad = -yi * sigmoid                               # (b,)

            # Backward
            # dw[j, val] += mean(grad * indicator)
            # dv[j, val, :] += mean(grad * (sum_v - v[j, val]) * indicator)
            # Vectorized over batch
            for j in range(d):
                vals = xi[:, j]
                # Linear gradient
                np.add.at(mw[j], vals, grad)
                # Interaction gradient
                # sum_v shape (b, K), v[j, vals] shape (b, K)
                gv = grad[:, None] * (sum_v - v[j, vals])      # (b, K)
                np.add.at(mv[j], vals, gv)

            t += 1
            # Adam update
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # Linear weights
            mw *= beta1; mw += (1 - beta1) * (mw / BATCH + L2 * w)  # mw holds mean grad
            vw *= beta2; vw += (1 - beta2) * (mw / BATCH + L2 * w) ** 2
            # Wait, need correct Adam: m = beta1*m + (1-beta1)*g, v = beta2*v + (1-beta2)*g^2
            # Let's redo properly

    # Re-implement Adam correctly
    mw, mv = np.zeros_like(w), np.zeros_like(v)
    vw, vv = np.zeros_like(w), np.zeros_like(v)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]
            yi = y_pm[idx]

            linear = np.sum(w[np.arange(d), xi.T], axis=0)
            sum_v = np.sum(v[np.arange(d), xi.T].transpose(1, 0, 2), axis=1)
            sum_v_sq = np.sum(sum_v * sum_v, axis=1)
            v_norm_sq = np.sum(v[np.arange(d), xi.T] ** 2, axis=(1, 2))
            interaction = 0.5 * (sum_v_sq - v_norm_sq)
            score = linear + interaction
            margin = -yi * score
            z = np.maximum(-margin, 0)
            sigmoid = np.exp(-z) / (1 + np.exp(-np.abs(margin)))
            grad = -yi * sigmoid

            # Gradients
            gw = np.zeros_like(w)
            gv = np.zeros_like(v)
            for j in range(d):
                vals = xi[:, j]
                np.add.at(gw[j], vals, grad)
                g = grad[:, None] * (sum_v - v[j, vals])
                np.add.at(gv[j], vals, g)

            t += 1
            beta1, beta2 = 0.9, 0.999
            mw = beta1 * mw + (1 - beta1) * (gw / BATCH + L2 * w)
            vw = beta2 * vw + (1 - beta2) * (gw / BATCH + L2 * w) ** 2
            mv = beta1 * mv + (1 - beta1) * (gv / BATCH + L2 * v)
            vv = beta2 * vv + (1 - beta2) * (gv / BATCH + L2 * v) ** 2
            m_hat_w = mw / (1 - beta1 ** t)
            v_hat_w = vw / (1 - beta2 ** t)
            m_hat_v = mv / (1 - beta1 ** t)
            v_hat_v = vv / (1 - beta2 ** t)
            w -= LR * m_hat_w / (np.sqrt(v_hat_w) + eps)
            v -= LR * m_hat_v / (np.sqrt(v_hat_v) + eps)

    # Precompute squared norms of latent vectors for fast inference
    v_norm_sq = np.sum(v ** 2, axis=2).astype(np.float32)  # (d, 256)

    # Pack weights: magic, d, K, C=1
    out = struct.pack("<4i", 0x464D3100, d, K, 1)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    out += v_norm_sq.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d=14 bytes per record. Output: 1 byte class (0/1).
 * Weights: magic 0x464D3100, d, K, C=1; float32 w[d][256], v[d][256][K], v_norm_sq[d][256]. */
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
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K != 8 || C != 1) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * d * 256 * (1 + K + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *w = (const float *)(hd + 4);
    const float *v = w + (long)d * 256;
    const float *v_norm_sq = v + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = 0.0f;
        for (int c = 0; c < K; c++) sum_v[c] = 0.0f;
        float sum_v_norm_sq = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            long base_w = (long)j * 256 + val;
            linear += w[base_w];
            long base_v = base_w * K;
            for (int c = 0; c < K; c++) sum_v[c] += v[base_v + c];
            sum_v_norm_sq += v_norm_sq[base_w];
        }

        float dot = 0.0f;
        for (int c = 0; c < K; c++) dot += sum_v[c] * sum_v[c];
        float interaction = 0.5f * (dot - sum_v_norm_sq);
        float score = linear + interaction;

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```