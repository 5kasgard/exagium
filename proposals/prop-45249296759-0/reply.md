```hypothesis
Propose a factorization machine (FM) family: per-feature embeddings (k=8) plus class-specific linear weights.
FMs capture pairwise interactions efficiently (O(d*k) inference via the FM trick) and are strong on tabular data.
This should beat label-linear (0.8524) by modeling interactions while staying faster than MLPs/GBDTs.
If dev accuracy does not exceed label-linear, interactions are not helpful or k is too small.
```
```python
"""Factorization Machine for tabular label tasks: shared embeddings + class-specific linear weights.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exported weights: magic 0x464D4C31, d, k, C; float32 b[C], W_lin[d][256][C], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 30


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
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters: b[C], W_lin[d,256,C], V[d,256,K]
    b = np.zeros(C, np.float32)
    W_lin = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    m_W, v_W = np.zeros_like(W_lin), np.zeros_like(W_lin)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    t = 0
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            # Linear part: (B, C)
            lin = b + np.sum(W_lin[np.arange(d)[:, None], xb.T], axis=0).T  # (B, C)
            # FM interaction
            # sum_V: (B, K)
            sum_V = np.sum(V[np.arange(d)[:, None], xb.T], axis=0).T
            # sum_sq_V: (B, K)
            sum_sq_V = np.sum(V[np.arange(d)[:, None], xb.T] ** 2, axis=0).T
            inter = 0.5 * (np.sum(sum_V ** 2, axis=1) - np.sum(sum_sq_V, axis=1))  # (B,)
            scores = lin + inter[:, None]  # (B, C)

            # Softmax + cross-entropy gradient
            max_s = np.max(scores, axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B  # (B, C)

            # Gradients
            # b
            g_b = np.sum(grad_scores, axis=0)
            # W_lin
            g_W = np.zeros_like(W_lin)
            for j in range(d):
                vals = xb[:, j]
                for c in range(C):
                    np.add.at(g_W[j, :, c], vals, grad_scores[:, c])
            # V
            g_V = np.zeros_like(V)
            # dI/dV_j = sum_V - V_j
            for j in range(d):
                vals = xb[:, j]
                V_j = V[j, vals]                 # (B, K)
                factor = (sum_V - V_j) * grad_scores.sum(axis=1)[:, None]  # (B, K)
                for k in range(K):
                    np.add.at(g_V[j, :, k], vals, factor[:, k])

            # L2 regularization
            g_W += L2 * W_lin
            g_V += L2 * V

            t += 1
            # Adam updates
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # b
            m_b = beta1 * m_b + (1 - beta1) * g_b
            v_b = beta2 * v_b + (1 - beta2) * g_b * g_b
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            b -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)
            # W_lin
            m_W = beta1 * m_W + (1 - beta1) * g_W
            v_W = beta2 * v_W + (1 - beta2) * g_W * g_W
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            W_lin -= LR * m_W_hat / (np.sqrt(v_W_hat) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += b.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class index (argmax). */
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
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * (C + K));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *b = (const float *)(hd + 4);
    const float *W_lin = b + C;
    const float *V = W_lin + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_V = malloc(sizeof(float) * K);
    float *sum_sq_V = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *w = W_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        // FM interaction
        for (int kk = 0; kk < K; kk++) { sum_V[kk] = 0.0f; sum_sq_V[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = v[kk];
                sum_V[kk] += val;
                sum_sq_V[kk] += val * val;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += sum_V[kk] * sum_V[kk] - sum_sq_V[kk];
        inter *= 0.5f;
        for (int c = 0; c < C; c++) scores[c] += inter;

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(sum_V); free(sum_sq_V);
    return 0;
}
```