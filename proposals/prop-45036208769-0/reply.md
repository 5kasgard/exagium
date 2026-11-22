```hypothesis
Factorization Machine (FM) with k=8 latent factors captures pairwise feature interactions
efficiently, which is crucial for tabular data with mixed categorical/ordinal features like Adult.
It should beat label-linear (0.8524) by modeling interactions (e.g., education×occupation)
while staying faster than MLP/GBDT at inference. If dev accuracy ≤0.8524 or latency ≥label-linear,
the approach fails to justify its complexity.
```
```python
"""Factorization Machine for tabular label tasks: 2nd-order interactions via factorized weights.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1)
Encoding: ordinal z-scored (1 dim), categorical one-hot over training values (≤256 dims).
FM: logit_c = b_c + Σ_j linear_c[j][x_j] + ½ Σ_f ( (Σ_j factor_c[j][x_j][f])² - Σ_j factor_sq_c[j][x_j][f] )
Exported as per-byte lookup tables: magic 0x464D31, d, k, C; float32 linear[d][256][C], factor[d][256][k][C],
factor_sq[d][256][k][C], bias[C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.03, 1e-4, 64, 600000, 100


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
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    rng = np.random.default_rng(SEED)

    # Build expanded design matrix and per-feature encoding maps
    enc_maps = []          # list of (expanded_indices, encoder_matrix)
    # encoder_matrix: for ordinal (256,1), for categorical (256, n_vals)
    expanded_dim = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256, dtype=np.float64) - mu) / sd)[:, None]  # (256,1)
            idx = np.arange(expanded_dim, expanded_dim + 1)
            expanded_dim += 1
        else:
            vals = np.unique(X[:, j])
            n_vals = len(vals)
            E = np.zeros((256, n_vals), dtype=np.float64)
            E[vals, np.arange(n_vals)] = 1.0
            idx = np.arange(expanded_dim, expanded_dim + n_vals)
            expanded_dim += n_vals
        enc_maps.append((idx, E.astype(np.float32)))

    # Expanded feature matrix Z (n, expanded_dim)
    n = X.shape[0]
    Z = np.zeros((n, expanded_dim), dtype=np.float32)
    for j, (idx, E) in enumerate(enc_maps):
        Z[:, idx] = E[X[:, j]]

    # Initialize parameters per class
    w = (rng.uniform(-0.01, 0.01, (C, expanded_dim)) * math.sqrt(6.0 / expanded_dim)).astype(np.float32)
    V = (rng.uniform(-0.01, 0.01, (C, expanded_dim, K)) * math.sqrt(6.0 / expanded_dim)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    mw, mV, mb = np.zeros_like(w), np.zeros_like(V), np.zeros_like(b)
    vw, vV, vb = np.zeros_like(w), np.zeros_like(V), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def softmax_grad(logits, yy):
        # logits: (batch, C), yy: (batch,)
        P = np.exp(logits - logits.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            bs = len(yy)

            # Forward
            linear = z @ w.T + b  # (bs, C)
            inter = np.zeros((bs, C), dtype=np.float32)
            for f in range(K):
                vf = V[:, :, f]  # (C, expanded_dim)
                z_vf = z @ vf.T   # (bs, C)
                z2_vf2 = (z * z) @ (vf * vf).T  # (bs, C)
                inter += 0.5 * (z_vf * z_vf - z2_vf2)
            logits = linear + inter

            # Gradients
            g = softmax_grad(logits, yy)  # (bs, C)
            # grad_w = g.T @ z + L2 * w  (C, expanded_dim)
            grad_w = g.T @ z + L2 * w
            # grad_b = g.sum(0)
            grad_b = g.sum(axis=0)
            # grad_V: for each f, g_c * (z_vf_c * z - z * vf_c) + L2 * V_c
            grad_V = np.zeros_like(V)
            for f in range(K):
                vf = V[:, :, f]  # (C, expanded_dim)
                z_vf = z @ vf.T   # (bs, C)
                # g_c * z_vf_c[:, None] * z  -  g_c[:, None] * z * vf_c[None, :]
                # First term: (bs, C, 1) * (bs, expanded_dim) -> (bs, C, expanded_dim)
                # Second term: (bs, C, 1) * (bs, expanded_dim) * (C, expanded_dim)
                term1 = (g[:, :, None] * z_vf[:, :, None]) * z[:, None, :]  # (bs, C, expanded_dim)
                term2 = (g[:, :, None] * z[:, None, :]) * vf[None, :, :]    # (bs, C, expanded_dim)
                grad_V[:, :, f] = (term1 - term2).sum(axis=0) + L2 * vf

            # Adam update
            t += 1
            for param, grad, m, v in [(w, grad_w, mw, vw), (V, grad_V, mV, vV), (b, grad_b, mb, vb)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Precompute lookup tables for export
    # linear_table[j][v][c] = sum_{i in feature j} w[c, i] * E_j[v, i_local]
    # factor_table[j][v][f][c] = sum_i V[c, i, f] * E_j[v, i_local]
    # factor_sq_table[j][v][f][c] = sum_i V[c, i, f]^2 * E_j[v, i_local]^2
    linear_table = np.zeros((d, 256, C), dtype=np.float32)
    factor_table = np.zeros((d, 256, K, C), dtype=np.float32)
    factor_sq_table = np.zeros((d, 256, K, C), dtype=np.float32)

    for j, (idx, E) in enumerate(enc_maps):
        # E: (256, n_local)
        # w[:, idx]: (C, n_local) -> (n_local, C)
        linear_table[j] = E @ w[:, idx].T  # (256, C)
        for f in range(K):
            # V[:, idx, f]: (C, n_local) -> (n_local, C)
            factor_table[j, :, f, :] = E @ V[:, idx, f].T
            factor_sq_table[j, :, f, :] = (E * E) @ (V[:, idx, f] * V[:, idx, f]).T

    # Pack weights
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D31, d, K, C)
    out += linear_table.astype("<f4").tobytes()
    out += factor_table.astype("<f4").tobytes()
    out += factor_sq_table.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class index (argmax).
 * Weights format: magic 0x464D31, int32 d, k, C; float32 linear[d][256][C],
 * factor[d][256][k][C], factor_sq[d][256][k][C], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }

    // Expected size: 16 + 4*(d*256*C + d*256*K*C + d*256*K*C + C)
    long expected = 16L + 4L * ((long)d * 256 * C * (1 + 2L * K) + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *factor = linear + (long)d * 256 * C;
    const float *factor_sq = factor + (long)d * 256 * K * C;
    const float *bias = factor_sq + (long)d * 256 * K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *sum_f = malloc(sizeof(float) * C * K);
    float *sum_sq_f = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize logits with bias
        for (int c = 0; c < C; c++) logits[c] = bias[c];

        // Linear terms
        for (int j = 0; j < d; j++) {
            const float *lj = linear + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += lj[c];
        }

        // Factor terms: for each factor f, accumulate sum and sum_sq per class
        for (int f = 0; f < K; f++) {
            for (int c = 0; c < C; c++) {
                sum_f[c * K + f] = 0.0f;
                sum_sq_f[c * K + f] = 0.0f;
            }
            for (int j = 0; j < d; j++) {
                const float *fj = factor + ((long)j * 256 + xi[j]) * K * C + f * C;
                const float *fsqj = factor_sq + ((long)j * 256 + xi[j]) * K * C + f * C;
                for (int c = 0; c < C; c++) {
                    sum_f[c * K + f] += fj[c];
                    sum_sq_f[c * K + f] += fsqj[c];
                }
            }
        }

        // Add interactions
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float sf = sum_f[c * K + f];
                float ssf = sum_sq_f[c * K + f];
                inter += 0.5f * (sf * sf - ssf);
            }
            logits[c] += inter;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(logits); free(sum_f); free(sum_sq_f);
    return 0;
}
```