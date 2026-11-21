```hypothesis
Family: label-fm (Factorization Machine, degree 2, rank 8).
Why: Adult has strong pairwise interactions (e.g., education×occupation, marital-status×relationship) that a linear model misses. A rank-8 FM adds only 8 latent factors per feature-value, so inference stays O(d·R) ≈ 112 FMA/item (vs 14 for linear) while capturing all pairwise crosses. Training uses Adam on the full train set (deterministic seed), with R=8 chosen on the tuning split.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 1200, the family does not justify its complexity.
```

```python
"""Factorization Machine (degree 2, rank R) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib + numpy only.
Exports: magic 0x464D3231, d, R, C; float32 linear[d][256], fm[d][256][R], bias."""
import json, math, struct, sys

import numpy as np

R = 8
SEED, LR, L2, L2_FM, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 1e-5, 256, 600000, 100


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

    # Build one-hot indices per feature (256 values each)
    # Linear weights: (d, 256)
    W = (rng.uniform(-0.01, 0.01, (d, 256))).astype(np.float32)
    # FM factors: (d, 256, R)
    V = (rng.uniform(-0.01, 0.01, (d, 256, R))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    mb, vb = np.zeros_like(bias), np.zeros_like(bias)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    idx_all = np.arange(n)

    for _ in range(epochs):
        rng.shuffle(idx_all)
        for s in range(0, n, BATCH):
            idx = idx_all[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Linear logits: sum_j W[j, xb[:, j]] + bias
            lin = np.zeros((B, C), np.float32)
            for j in range(d):
                lin += W[j, xb[:, j]][:, None]  # broadcast over C
            lin += bias

            # FM logits: 0.5 * sum_r (sum_j V[j, xb[:, j], r])^2 - sum_j V[j, xb[:, j], r]^2
            # Efficient: for each r, compute sum_j V and sum_j V^2
            fm = np.zeros((B, C), np.float32)
            for r in range(R):
                sum_v = np.zeros(B, np.float32)
                sum_v2 = np.zeros(B, np.float32)
                for j in range(d):
                    vjr = V[j, xb[:, j], r]   # (B,)
                    sum_v += vjr
                    sum_v2 += vjr * vjr
                inter = 0.5 * (sum_v * sum_v - sum_v2)  # (B,)
                fm += inter[:, None]  # broadcast over C

            logits = lin + fm  # (B, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad = probs / B  # (B, C)

            # Gradients
            gW = np.zeros_like(W)
            gV = np.zeros_like(V)
            gb = grad.sum(axis=0)

            # Linear gradient
            for j in range(d):
                # For each sample, add grad to W[j, xb[i, j]]
                np.add.at(gW[j], xb[:, j], grad)

            # FM gradient
            for r in range(R):
                sum_v = np.zeros(B, np.float32)
                for j in range(d):
                    sum_v += V[j, xb[:, j], r]
                # d(inter)/dV = (sum_v - V) * grad  (broadcast over C, but grad is per-class)
                # Since inter is shared across classes, gradient wrt V is sum_c grad[:, c] * (sum_v - V)
                g_sum = grad.sum(axis=1)  # (B,)
                for j in range(d):
                    vjr = V[j, xb[:, j], r]
                    gv = (sum_v - vjr) * g_sum  # (B,)
                    np.add.at(gV[j, :, r], xb[:, j], gv)

            # L2 regularization
            gW += L2 * W
            gV += L2_FM * V

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            # Update W
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * (gW * gW)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + eps)

            # Update V
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * (gV * gV)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= LR * mV_hat / (np.sqrt(vV_hat) + eps)

            # Update bias
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * (gb * gb)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            bias -= LR * mb_hat / (np.sqrt(vb_hat) + eps)

    # Write weights: magic, d, R, C; then linear[d][256], fm[d][256][R], bias[C]
    out = struct.pack("<4i", 0x464D3231, d, R, C)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (degree 2, rank R) inference.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes. Output: 1 byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C < 1 || C > 256 || R < 1 || R > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad header values\n"); return 3;
    }
    size_t lin_sz = (size_t)d * 256;
    size_t fm_sz = (size_t)d * 256 * R;
    size_t bias_sz = C;
    size_t expected = 16 + 4 * (lin_sz + fm_sz + bias_sz);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *lin = (const float *)(hd + 4);
    const float *fm = lin + lin_sz;
    const float *bias = fm + fm_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * R);
    float *sum_v2 = malloc(sizeof(float) * R);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *w = lin + ((size_t)j * 256 + xi[j]);
            for (int c = 0; c < C; c++) scores[c] += w[0];  // linear weight per feature-value (shared across classes)
        }

        // FM part: 0.5 * sum_r (sum_j v_jr)^2 - sum_j v_jr^2
        // v_jr is per feature-value per rank (shared across classes)
        for (int r = 0; r < R; r++) {
            sum_v[r] = 0.0f;
            sum_v2[r] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            const float *v = fm + ((size_t)j * 256 + xi[j]) * R;
            for (int r = 0; r < R; r++) {
                float val = v[r];
                sum_v[r] += val;
                sum_v2[r] += val * val;
            }
        }
        for (int r = 0; r < R; r++) {
            float inter = 0.5f * (sum_v[r] * sum_v[r] - sum_v2[r]);
            for (int c = 0; c < C; c++) scores[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(sum_v); free(sum_v2);
    return 0;
}
```