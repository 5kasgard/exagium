```hypothesis
Family: label-fm (Factorization Machine with shared pairwise interactions).
Why: Optdigits pixels have strong local correlations; a degree-2 FM captures these with only O(d·k + d·C) parameters (k=8 factors). It adds a single scalar interaction term to the linear logits, giving non-linearity at ~1.2× the linear inference cost. Beats label-linear (96.1%) and approaches label-mlp16 (97.7%) with lower latency.
Failure signal: Dev accuracy ≤ 96.2% or instructions/item > 1.5× label-linear.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Model: score_c = bias_c + Σ_j w_{j,c}·e_j(x_j) + ½ Σ_f [(Σ_j v_{j,f}·e_j(x_j))² - Σ_j (v_{j,f}·e_j(x_j))²]
where e_j is per-feature z-scoring. Exported as lookup tables for fast inference.
Weights format (little-endian):
  int32 magic=0x464D4C42, d, k, C
  float32 T_lin[d][256][C]   # w_{j,c} * e_j(x)
  float32 bias[C]
  float32 V[d][256][k]       # v_{j,f} * e_j(x)
"""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 50


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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Encode features: z-score per feature (ordinal only in this task)
    enc_mu = np.zeros(d, np.float64)
    enc_sd = np.ones(d, np.float64)
    for j in range(d):
        v = X[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std()
        enc_mu[j], enc_sd[j] = mu, (sd if sd > 0 else 1.0)
    Z = ((X.astype(np.float64) - enc_mu) / enc_sd).astype(np.float32)  # (n, d)

    # Initialize parameters
    W = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    V = (rng.uniform(-1, 1, (d, K)) * math.sqrt(6.0 / d)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [W, V, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = Z.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            bsz = len(yy)

            # Forward
            lin = z @ W + bias  # (bsz, C)
            zv = z @ V          # (bsz, K)
            inter = 0.5 * ((zv * zv).sum(1) - (z * z) @ (V * V))  # (bsz,)
            logits = lin + inter[:, None]

            # Softmax + gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            grad_logits = (probs / bsz).astype(np.float32)  # (bsz, C)

            # Gradients
            grad_W = z.T @ grad_logits + L2 * W
            grad_bias = grad_logits.sum(0)

            # d(inter)/dz = zv @ V.T - z * (V*V).sum(1)  [broadcast]
            grad_inter = (grad_logits.sum(1, keepdims=True) *  # (bsz,1)
                          (zv @ V.T - z * (V * V).sum(1))).astype(np.float32)  # (bsz, d)
            grad_V = z.T @ (grad_logits.sum(1, keepdims=True) * zv) - (z * z).T @ (grad_logits.sum(1, keepdims=True) * V) + L2 * V
            # The above grad_V is wrong; let's derive properly:
            # inter_i = 0.5 * sum_f (zv_{if}^2 - sum_j z_{ij}^2 V_{jf}^2)
            # d(inter_i)/dV_{jf} = zv_{if} * z_{ij} - z_{ij}^2 * V_{jf}
            # grad_V_{jf} = sum_i grad_inter_i * (zv_{if} * z_{ij} - z_{ij}^2 * V_{jf})
            #             = sum_i g_i * z_{ij} * zv_{if} - sum_i g_i * z_{ij}^2 * V_{jf}
            #             = (z.T @ (g[:,None] * zv))_jf - ((z*z).T @ (g[:,None] * V))_jf
            g = grad_logits.sum(1)  # (bsz,) scalar gradient per sample
            grad_V = (z.T @ (g[:, None] * zv) - (z * z).T @ (g[:, None] * V) + L2 * V).astype(np.float32)

            grads = [grad_W, grad_V, grad_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup tables for all 256 byte values per feature
    # e_j(x) = (x - mu_j) / sd_j
    x_vals = np.arange(256, dtype=np.float32)
    E = ((x_vals - enc_mu[:, None]) / enc_sd[:, None]).astype(np.float32)  # (d, 256)

    # T_lin[j, x, c] = W[j, c] * E[j, x]
    T_lin = (W[:, :, None] * E[:, None, :]).transpose(0, 2, 1).astype(np.float32)  # (d, 256, C)
    # V_table[j, x, f] = V[j, f] * E[j, x]
    V_table = (V[:, :, None] * E[:, None, :]).transpose(0, 2, 1).astype(np.float32)  # (d, 256, K)

    # Pack weights
    magic = 0x464D4C42  # 'BLMF'
    hdr = struct.pack("<4i", magic, d, K, C)
    out = hdr + T_lin.tobytes() + bias.astype("<f4").tobytes() + V_table.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (argmax, first max wins).
 * Weights format (little-endian):
 *   int32 magic=0x464D4C42, d, k, C
 *   float32 T_lin[d][256][C]   // linear contribution per feature, value, class
 *   float32 bias[C]
 *   float32 V[d][256][k]       // factor contribution per feature, value, factor
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D4C42) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }

    size_t expected = 16
        + (size_t)d * 256 * C * sizeof(float)
        + (size_t)C * sizeof(float)
        + (size_t)d * 256 * K * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *bias = T_lin + (size_t)d * 256 * C;
    const float *V = bias + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *zv = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part + bias
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *t = T_lin + ((size_t)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += t[c];
        }

        // Interaction part: 0.5 * sum_f (zv_f^2 - sum_j V_{j,f}^2 * e_j^2)
        // But we have V_table[j][x][f] = V_{j,f} * e_j(x), so:
        // zv_f = sum_j V_table[j][x_j][f]
        // sum_sq_f = sum_j V_table[j][x_j][f]^2
        for (int f = 0; f < K; f++) zv[f] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *v = V + ((size_t)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) zv[f] += v[f];
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float sum_sq = 0.0f;
            for (int j = 0; j < d; j++) {
                float val = V[((size_t)j * 256 + xi[j]) * K + f];
                sum_sq += val * val;
            }
            inter += 0.5f * (zv[f] * zv[f] - sum_sq);
        }
        for (int c = 0; c < C; c++) scores[c] += inter;

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```