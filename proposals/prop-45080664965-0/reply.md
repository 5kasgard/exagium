```hypothesis
Family: label-fm (Factorization Machine with per-class embeddings, K=8).
Why: Adult income has known feature interactions (education×occupation, marital-status×relationship) that linear models miss; FM captures all pairwise interactions with O(d·K·C) params (≈65K), faster than MLP/GBDT at inference.
Failure signal: dev accuracy ≤ 0.853 (does not beat label-linear) or Ir/item ≥ 2000 (not cheaper than label-mlp16).
```
```python
"""Factorization Machine for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. FM with per-class embeddings (K=8), Adam lr=3e-3, L2=1e-4.
Weights: int32 magic 0x464D3130, d, K, C; float32 bias[C], linear[d][256][C], V[d][256][K][C]."""
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
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.zeros(C, np.float32)
    linear = np.zeros((d, 256, C), np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K, C))).astype(np.float32)

    params = [bias, linear, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]          # (B, d)
            yy = y[idx]         # (B,)
            B = len(yy)

            # Forward: scores[B, C]
            scores = np.zeros((B, C), np.float32)
            scores += bias
            for j in range(d):
                scores += linear[j, x[:, j], :]      # (B, C)
            for c in range(C):
                inter = np.zeros(B, np.float32)
                for k in range(K):
                    v = V[np.arange(d)[:, None], x.T, k, c]   # (d, B)
                    sum_v = v.sum(axis=0)
                    sum_v2 = (v * v).sum(axis=0)
                    inter += 0.5 * (sum_v * sum_v - sum_v2)
                scores[:, c] += inter

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_scores = (probs / B).astype(np.float32)     # (B, C)

            # Gradients
            g_bias = grad_scores.sum(axis=0)                 # (C,)
            g_linear = np.zeros_like(linear)
            g_V = np.zeros_like(V)

            for j in range(d):
                xj = x[:, j]                                 # (B,)
                for c in range(C):
                    g_linear[j, xj, c] += grad_scores[:, c].sum()
            for c in range(C):
                for k in range(K):
                    v = V[np.arange(d)[:, None], x.T, k, c]  # (d, B)
                    sum_v = v.sum(axis=0)                    # (B,)
                    # d(inter)/dV_{j} = sum_v - v_j
                    for j in range(d):
                        vj = v[j]                            # (B,)
                        gv = (sum_v - vj) * grad_scores[:, c]   # (B,)
                        g_V[j, xj, k, c] += gv

            # L2 regularization
            g_bias += L2 * bias
            g_linear += L2 * linear
            g_V += L2 * V

            grads = [g_bias, g_linear, g_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D3130
    out = struct.pack("<4i", magic, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class index (argmax). */
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
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *bias = p; p += C;
    const float *linear = p; p += (long)d * 256 * C;
    const float *V = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        scores[0] = bias[0]; scores[1] = bias[1];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            scores[0] += linear[((long)j * 256 + v) * C + 0];
            scores[1] += linear[((long)j * 256 + v) * C + 1];
        }
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float sum_v = 0.0f, sum_v2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    int v = xi[j];
                    float val = V[(((long)j * 256 + v) * K + kk) * C + c];
                    sum_v += val;
                    sum_v2 += val * val;
                }
                inter += 0.5f * (sum_v * sum_v - sum_v2);
            }
            scores[c] += inter;
        }
        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```