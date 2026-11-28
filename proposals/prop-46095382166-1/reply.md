```hypothesis
Family: label-fm (Factorization Machine with shared embeddings and class-specific interactions).
Why: Optdigits pixels (0-16) have strong pairwise spatial correlations (strokes, loops) that linear models miss. A rank-K FM captures all pairwise interactions via shared latent embeddings (V) plus class-specific linear (W) and interaction weights (U), adding only ~2K ops/item vs label-linear's ~0.6K. With K=8 it should reach ~0.975 accuracy (near MLP) at ~2x linear's latency.
Failure signal: Dev accuracy ≤ 0.965 or instructions/item > 12000 (2.4× label-linear).
```
```python
"""label-fm: Factorization Machine for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Shared per-pixel/value embeddings V[d][256][K], class-specific linear W[C][d][256],
class-specific interaction weights U[C][K], bias b[C].
Score_c = b_c + sum_j W_c[j][x_j] + 0.5 * sum_k U_c[k] * (S_k^2 - S_sq_k)
where S_k = sum_j V[j][x_j]_k, S_sq_k = sum_j V[j][x_j]_k^2.
Adam lr=1e-2, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)), seed=1."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100
MAGIC = 0x464D31  # "FM1"

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

    # Parameters: V[d, 256, K], W[C, d, 256], U[C, K], b[C]
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    W = np.zeros((C, d, 256), np.float32)
    U = (rng.normal(0, 0.01, (C, K))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [V, W, U, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            m = len(yb)

            # Forward
            # S[m, K] = sum_j V[j, xb[:,j], :]
            # S_sq[m, K] = sum_j V[j, xb[:,j], :]^2
            S = np.zeros((m, K), np.float32)
            S_sq = np.zeros((m, K), np.float32)
            linear = b.astype(np.float32) + np.zeros((m, C), np.float32)
            for j in range(d):
                v = xb[:, j]
                S += V[j, v]
                vv = V[j, v]
                S_sq += vv * vv
                for c in range(C):
                    linear[:, c] += W[c, j, v]

            # Interaction per class
            scores = linear.copy()
            for c in range(C):
                inter = 0.5 * np.sum(U[c] * (S * S - S_sq), axis=1)
                scores[:, c] += inter

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(m), yb] -= 1.0
            grad_scores = (probs / m).astype(np.float32)  # [m, C]

            # Gradients
            g_b = grad_scores.sum(axis=0)  # [C]
            g_W = np.zeros_like(W)
            g_U = np.zeros_like(U)
            g_V = np.zeros_like(V)

            # dL/dlinear = grad_scores
            for c in range(C):
                g_W[c] += np.zeros_like(W[c])  # placeholder, accumulate per j
            # Efficient accumulation:
            for j in range(d):
                v = xb[:, j]
                for c in range(C):
                    g_W[c, j, v] += grad_scores[:, c]

            # dL/dS_k = sum_c grad_scores[:,c] * U[c,k] * S_k
            # dL/dS_sq_k = -sum_c grad_scores[:,c] * U[c,k] * 0.5
            dS = np.zeros((m, K), np.float32)
            dS_sq = np.zeros((m, K), np.float32)
            for c in range(C):
                uc = U[c]
                gs = grad_scores[:, c]
                dS += gs[:, None] * uc * S
                dS_sq -= 0.5 * gs[:, None] * uc
                # g_U[c,k] = 0.5 * sum_i gs_i * (S_ik^2 - S_sq_ik)
                g_U[c] += 0.5 * np.sum(gs[:, None] * (S * S - S_sq), axis=0)

            # Backprop to V
            for j in range(d):
                v = xb[:, j]
                g_V[j, v] += dS
                g_V[j, v] += 2.0 * V[j, v] * dS_sq

            # Adam update
            grads = [g_V, g_W, g_U, g_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_adam, v_adam = state[i]
                m_adam *= 0.9; m_adam += 0.1 * gr
                v_adam *= 0.999; v_adam += 0.001 * gr * gr
                m_hat = m_adam / (1 - 0.9 ** t)
                v_hat = v_adam / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                # L2 on W and V (not on U, b typically)
                if i in (0, 1):  # V, W
                    p -= LR * L2 * p

    # Pack weights: magic, d, K, C, then V[d][256][K], W[C][d][256], U[C][K], b[C] (all float32 LE)
    total_floats = d * 256 * K + C * d * 256 + C * K + C
    header = struct.pack("<4i", MAGIC, d, K, C)
    body = (V.astype("<f4").tobytes() +
            W.astype("<f4").tobytes() +
            U.astype("<f4").tobytes() +
            b.astype("<f4").tobytes())
    assert len(body) == total_floats * 4
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Factorization Machine with shared embeddings.
 * Weights format: int32 magic(0x464D31), d, K, C; float32 V[d][256][K], W[C][d][256], U[C][K], b[C]. */
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
    if (C < 1 || C > 256 || K < 1 || K > 64 || d != 64) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t exp = 16 + 4L * ((long)d * 256 * K + (long)C * d * 256 + (long)C * K + C);
    if (nw != (long)exp) { fprintf(stderr, "bad weights size %ld vs %zu\n", nw, exp); return 3; }

    const float *V = (const float *)(hd + 4);
    const float *W = V + (long)d * 256 * K;
    const float *U = W + (long)C * d * 256;
    const float *b = U + (long)C * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = calloc(K, sizeof(float));
    float *S_sq = calloc(K, sizeof(float));
    float *linear = malloc(C * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int kk = 0; kk < K; kk++) { S[kk] = 0.0f; S_sq[kk] = 0.0f; }
        for (int c = 0; c < C; c++) linear[c] = b[c];

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Vjv = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = Vjv[kk];
                S[kk] += val;
                S_sq[kk] += val * val;
            }
            for (int c = 0; c < C; c++) {
                linear[c] += W[((long)c * d + j) * 256 + v];
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            const float *Uc = U + (long)c * K;
            for (int kk = 0; kk < K; kk++) {
                inter += Uc[kk] * (S[kk] * S[kk] - S_sq[kk]);
            }
            scores[c] = linear[c] + 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```