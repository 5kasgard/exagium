```hypothesis
Family: label-fm (Factorization Machine with low-rank interaction layer).
Reason: Character-level language modeling benefits from modeling pairwise interactions between context bytes (e.g., bigrams, trigrams). A linear model (label-linear) cannot capture these; a full MLP is heavy. FM computes all pairwise interactions via a shared low-rank embedding (size K=32), adding only O(d*K + K*C) operations per item. This should beat label-linear's 0.3375 accuracy while keeping weights ~500KB and inference fast.
Failure signal: Dev accuracy ≤ 0.3375 or instructions/item > 15000.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Model: score = b + FM_interaction(V) @ W, where FM_interaction computes
0.5*((sum V)^2 - sum V^2) over the d positions. V shape (d,256,K), W (K,C), b (C).
Exported weights: magic 0x464D4C31, d, K, C; then V, W, b as float32 LE."""
import json, math, struct, sys

import numpy as np

K = 32
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 20


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    W = rng.normal(0, 0.01, (K, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Gather embeddings: V[xb] -> (d, B, K)
            Vemb = V[np.arange(d)[:, None], xb.T]  # (d, B, K)
            sumV = Vemb.sum(axis=0)                # (B, K)
            sumV2 = (Vemb * Vemb).sum(axis=0)      # (B, K)
            inter = 0.5 * (sumV * sumV - sumV2)    # (B, K)

            scores = inter @ W + b                 # (B, C)
            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B                # (B, C)

            # Gradients
            grad_W = inter.T @ grad_scores + L2 * W          # (K, C)
            grad_b = grad_scores.sum(axis=0)                 # (C,)
            grad_inter = grad_scores @ W.T                   # (B, K)

            # Backprop through FM interaction
            # inter = 0.5*(sumV^2 - sumV2)
            # d inter / d sumV = sumV
            # d inter / d sumV2 = -0.5
            grad_sumV = grad_inter * sumV                    # (B, K)
            grad_sumV2 = -0.5 * grad_inter                   # (B, K)

            # sumV = sum_d Vemb[d], sumV2 = sum_d Vemb[d]^2
            # So grad_Vemb[d] = grad_sumV + 2 * Vemb[d] * grad_sumV2
            grad_Vemb = grad_sumV[None, :, :] + 2.0 * Vemb * grad_sumV2[None, :, :]  # (d, B, K)
            grad_Vemb += L2 * Vemb

            # Scatter grad_Vemb into grad_V (d, 256, K)
            grad_V = np.zeros_like(V)
            np.add.at(grad_V, (np.arange(d)[:, None], xb.T), grad_Vemb)

            # Adam updates
            t += 1
            for param, grad, m, v in [
                (V, grad_V, mV, vV),
                (W, grad_W, mW, vW),
                (b, grad_b, mb, vb)
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    body = V.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record: d bytes. Output: one byte (class index 0..C-1).
 * Weights format: magic 0x464D4C31, int32 d, K, C; float32 V[d][256][K], W[K][C], b[C]. */
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
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * K + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *V = (const float *)(hd + 4);
    const float *W = V + (long)d * 256 * K;
    const float *b = W + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sumV = (float *)malloc(sizeof(float) * K);
    float *sumV2 = (float *)malloc(sizeof(float) * K);
    float *inter = (float *)malloc(sizeof(float) * K);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int kk = 0; kk < K; kk++) { sumV[kk] = 0.0f; sumV2[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            const float *v_ptr = V + ((long)j * 256 + byte) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = v_ptr[kk];
                sumV[kk] += v;
                sumV2[kk] += v * v;
            }
        }

        for (int kk = 0; kk < K; kk++) {
            inter[kk] = 0.5f * (sumV[kk] * sumV[kk] - sumV2[kk]);
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int kk = 0; kk < K; kk++) {
            float w = inter[kk];
            if (w == 0.0f) continue;
            const float *w_row = W + (long)kk * C;
            for (int c = 0; c < C; c++) scores[c] += w * w_row[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(sumV); free(sumV2); free(inter); free(scores);
    return 0;
}
```