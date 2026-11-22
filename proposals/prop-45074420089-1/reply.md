```hypothesis
Factorization Machine family (label-fm): adds pairwise pixel interactions via low-rank factors (rank 4) to a linear model, capturing stroke correlations crucial for 8x8 digits. With ~3200 params (5× linear) it should exceed label-linear's 0.9613 dev accuracy at ~5× inference cost (still far below mlp16). Failure: dev accuracy ≤ 0.962 or Ir/item > 30000.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Ordinal features z-scored. Rank-4 FM per class: bias + linear + factor vectors.
Adam lr 3e-3, L2 1e-4, minibatch 64, epochs min(100, ceil(600000/n)). Seed 1.
Exports: magic 0x464D4C31, d, k, C; float32 bias[C], linear[C][d], factors[C][d][k]."""
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
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # z-score ordinal features
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xz = ((X - mu) / sd).astype(np.float32)

    # params per class: bias (1), linear (d), factors (d, K)
    bias = np.zeros(C, np.float32)
    linear = (rng.uniform(-1, 1, (C, d)) * math.sqrt(6.0 / d)).astype(np.float32)
    factors = (rng.uniform(-1, 1, (C, d, K)) * math.sqrt(6.0 / (d * K))).astype(np.float32)

    params = [bias, linear, factors]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = Xz.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = Xz[idx]           # (B, d)
            yb = y[idx]            # (B,)
            B = len(yb)

            # forward: compute scores for all classes
            # linear part: (B, C)
            scores = xb @ linear.T + bias  # (B, C)
            # FM interaction part
            # for each class: 0.5 * (sum_f (sum_j v_cjf * x_j)^2 - sum_f sum_j (v_cjf * x_j)^2)
            # xb: (B, d), factors: (C, d, K) -> want (B, C, K) = sum_j xb[b,j] * factors[c,j,:]
            # einsum: 'bd,cdk->bck'
            xv = np.einsum('bd,cdk->bck', xb, factors)  # (B, C, K)
            sum_xv = xv.sum(axis=2)                     # (B, C)
            sum_xv_sq = (xv * xv).sum(axis=2)           # (B, C)
            scores += 0.5 * (sum_xv * sum_xv - sum_xv_sq)

            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (B, C)
            probs[np.arange(B), yb] -= 1.0
            probs /= B  # (B, C) gradient wrt scores

            # gradients
            # bias grad: sum_b probs[b,c] -> (C,)
            g_bias = probs.sum(axis=0).astype(np.float32)
            # linear grad: probs.T @ xb -> (C, d)
            g_linear = (probs.T @ xb).astype(np.float32) + L2 * linear
            # factors grad: for each class c, factor f: sum_b probs[b,c] * (xv[b,c,f] * xb[b,:] - xv[b,c,f] * xb[b,:]? wait)
            # d/dV_cjf of 0.5*( (sum_j V_cjf x_j)^2 - sum_j (V_cjf x_j)^2 ) = (sum_j V_cjf x_j) * x_j - (V_cjf x_j) * x_j
            # = x_j * (sum_l V_clf x_l - V_cjf x_j) = x_j * (xv_cf - V_cjf x_j)
            # So gradient for factors[c,j,f] = sum_b probs[b,c] * xb[b,j] * (xv[b,c,f] - factors[c,j,f] * xb[b,j])
            # Vectorized: (B, C, K) * (B, d, 1) * ( (B, C, K) - (C, d, K) * (B, d, 1) ) -> sum over b
            # Let's compute per class to avoid huge intermediate
            g_factors = np.zeros((C, d, K), np.float32)
            for c in range(C):
                pc = probs[:, c:c+1]  # (B, 1)
                xv_c = xv[:, c, :]    # (B, K)
                # (B, K) - (d, K) * (B, d, 1) -> (B, d, K) broadcast
                # term = xv_c[:, None, :] - factors[c][None, :, :] * xb[:, :, None]
                term = xv_c[:, None, :] - factors[c][None, :, :] * xb[:, :, None]  # (B, d, K)
                # sum_b pc[b] * xb[b, :, None] * term[b, :, :]
                g_factors[c] = (pc * xb[:, :, None] * term).sum(axis=0).astype(np.float32)
            g_factors += L2 * factors

            grads = [g_bias, g_linear, g_factors]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # export
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += factors.astype("<f4").tobytes()
    # also export mu, sd for inference z-scoring
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (rank-K) for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (z-scored using stored mu/sd). Output: 1 byte class (argmax). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 16 || d != 64) { fprintf(stderr, "bad header dims\n"); return 3; }
    size_t exp = 16
        + (size_t)C * 4
        + (size_t)C * d * 4
        + (size_t)C * d * K * 4
        + (size_t)d * 4
        + (size_t)d * 4;
    if (nw != (long)exp) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, exp); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *factors = linear + (size_t)C * d;
    const float *mu = factors + (size_t)C * d * K;
    const float *sd = mu + d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xz = (float *)malloc(d * sizeof(float));
    float *xv = (float *)malloc(C * K * sizeof(float));
    float *scores = (float *)malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // z-score
        for (int j = 0; j < d; j++) xz[j] = (xi[j] - mu[j]) / sd[j];
        // linear + bias
        for (int c = 0; c < C; c++) {
            float s = bias[c];
            const float *lc = linear + (size_t)c * d;
            for (int j = 0; j < d; j++) s += lc[j] * xz[j];
            scores[c] = s;
        }
        // FM interaction
        // xv[c*K + f] = sum_j factors[c][j][f] * xz[j]
        for (int c = 0; c < C; c++) {
            for (int f = 0; f < K; f++) xv[c * K + f] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            float xj = xz[j];
            if (xj == 0.0f) continue;
            for (int c = 0; c < C; c++) {
                const float *fc = factors + ((size_t)c * d + j) * K;
                for (int f = 0; f < K; f++) xv[c * K + f] += fc[f] * xj;
            }
        }
        for (int c = 0; c < C; c++) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int f = 0; f < K; f++) {
                float v = xv[c * K + f];
                sum += v;
                sum_sq += v * v;
            }
            scores[c] += 0.5f * (sum * sum - sum_sq);
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```