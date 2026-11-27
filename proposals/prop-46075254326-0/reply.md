```hypothesis
Factorization Machine (FM) on the 14 raw byte fields (k=8 latent factors). Each of the 14 columns has 256 values; every value gets a linear weight and a k-dim embedding. Pairwise interactions are computed via the FM trick: 0.5*(||Σv_j||² - Σ||v_j||²). This captures feature crosses crucial for tabular data (e.g., education×occupation) with only ~32K parameters. Deterministic Adam training. Should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at lower inference cost (simple lookups + vector ops). Failure: dev accuracy ≤ 0.8524.
```
```python
"""Factorization Machine for tabular byte features: python3 -s -P train.py card.json train.in train.key weights.bin
Each of the 14 input bytes (0-255) gets a linear weight and a k-dim embedding. Logit = linear + FM pairwise interactions.
Binary logistic loss, Adam optimizer, deterministic (seed 1). Exports: magic, d, k, C; float32 w[d][256], E[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2_W, L2_E, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 1e-6, 256, 200000, 50


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
    assert C == 2
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # w: [d, 256], E: [d, 256, K]
    w = np.zeros((d, 256), dtype=np.float32)
    E = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    mw, mE = np.zeros_like(w), np.zeros_like(E)
    vw, vE = np.zeros_like(w), np.zeros_like(E)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Gather embeddings and weights for this batch
            # w_b: [bsz, d], E_b: [bsz, d, K]
            w_b = w[np.arange(d), xb.T].T  # [bsz, d]
            E_b = E[np.arange(d), xb.T].transpose(1, 0, 2)  # [bsz, d, K]

            # Linear term
            linear = w_b.sum(axis=1)  # [bsz]

            # FM interaction: 0.5 * (||sum_j v_j||^2 - sum_j ||v_j||^2)
            sum_v = E_b.sum(axis=1)  # [bsz, K]
            sum_sq = (E_b * E_b).sum(axis=(1, 2))  # [bsz]
            inter = 0.5 * ((sum_v * sum_v).sum(axis=1) - sum_sq)  # [bsz]

            logit = linear + inter
            p = 1.0 / (1.0 + np.exp(-logit))
            tgt = (yb + 1) * 0.5  # 0 or 1
            g = (p - tgt).astype(np.float32)  # [bsz]

            # Gradients
            # dw[j, x_j] += g
            # dE[j, x_j] += g * (sum_v - v_j)
            # Vectorized scatter-add
            gw = np.zeros_like(w)
            gE = np.zeros_like(E)

            # For each feature j
            for j in range(d):
                xj = xb[:, j]  # [bsz]
                np.add.at(gw[j], xj, g)
                # sum_v - E_b[:, j] -> [bsz, K]
                diff = sum_v - E_b[:, j]  # [bsz, K]
                gEj = g[:, None] * diff  # [bsz, K]
                np.add.at(gE[j], xj, gEj)

            # Regularization
            gw += L2_W * w
            gE += L2_E * E

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            mw = beta1 * mw + (1 - beta1) * gw
            mE = beta1 * mE + (1 - beta1) * gE
            vw = beta2 * vw + (1 - beta2) * (gw * gw)
            vE = beta2 * vE + (1 - beta2) * (gE * gE)

            mw_hat = mw / (1 - beta1 ** t)
            mE_hat = mE / (1 - beta1 ** t)
            vw_hat = vw / (1 - beta2 ** t)
            vE_hat = vE / (1 - beta2 ** t)

            w -= LR * mw_hat / (np.sqrt(vw_hat) + eps)
            E -= LR * mE_hat / (np.sqrt(vE_hat) + eps)

    # Pack weights: magic, d, K, C; w[d][256], E[d][256][K] (all float32 little-endian)
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, K, C)
    out += w.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: 14-byte records. Output: 1-byte class (0/1). FM with k=8 latent factors. */
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
    if (d != 14 || K != 8 || C != 2) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t w_sz = (size_t)d * 256;
    size_t e_sz = (size_t)d * 256 * K;
    if (nw != 16 + 4L * (w_sz + e_sz)) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w = (const float *)(hd + 4);
    const float *E = w + w_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[8];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = 0.0f;
        for (int u = 0; u < K; u++) sum_v[u] = 0.0f;
        float sum_sq = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            size_t base_w = (size_t)j * 256 + v;
            size_t base_e = ((size_t)j * 256 + v) * K;
            linear += w[base_w];
            for (int u = 0; u < K; u++) {
                float e = E[base_e + u];
                sum_v[u] += e;
                sum_sq += e * e;
            }
        }

        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += sum_v[u] * sum_v[u];
        inter = 0.5f * (inter - sum_sq);

        float logit = linear + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```