```hypothesis
Factorization Machine (FM) with shared K=8 interaction factors and per-class linear weights. 
Directly uses the 14 input bytes as indices into lookup tables (no re-encoding), modeling all pairwise 
interactions via factorized vectors. FMs excel on tabular data with mixed ordinal/categorical features 
and are faster than MLPs/GBDTs. If dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 1500, it fails.
```

```python
"""Factorization Machine family for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. 
Input bytes used directly as indices: linear[d][256][C] + factors[d][256][K] (K=8, shared across classes).
Adam lr=3e-3, L2=1e-4 on linear, L2=1e-3 on factors, batch=64, epochs=min(100, ceil(600000/n)).
Weights: magic 0x464D4C31, int32 d,K,C; float32 bias[C], linear[d][256][C], V[d][256][K]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2_LIN, L2_FAC, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 1e-3, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_grad(logits, y):
    # logits: (B, C), y: (B,) -> grad (B, C)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    p[np.arange(len(y)), y] -= 1.0
    return (p / len(y)).astype(np.float32)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # params: bias[C], lin[d,256,C], V[d,256,K]
    bias = np.zeros(C, np.float32)
    lin = (rng.normal(0, 0.01, (d, 256, C)) * math.sqrt(2.0 / (d * 256))).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K)) * math.sqrt(2.0 / (d * 256))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            # Linear part: (B, C)
            logits = bias + np.sum(lin[xb, np.arange(256)[:, None], :], axis=1)  # WRONG: need proper indexing
            # Fix: lin[xb, :, :] doesn't work directly. Use advanced indexing.
            # xb is (B, d). We want lin[xb[:,j], j, :] for each j.
            # Let's compute properly:
            logits = np.full((B, C), bias, dtype=np.float32)
            for j in range(d):
                logits += lin[j, xb[:, j], :]

            # FM interaction part: 0.5 * sum_k (sum_j V_jk)^2 - sum_j V_jk^2
            # sum_V: (B, K)
            sum_V = np.zeros((B, K), np.float32)
            sum_V_sq = np.zeros((B, K), np.float32)
            for j in range(d):
                vj = V[j, xb[:, j], :]  # (B, K)
                sum_V += vj
                sum_V_sq += vj * vj
            fm = 0.5 * np.sum(sum_V * sum_V - sum_V_sq, axis=1, keepdims=True)  # (B, 1)
            logits += fm  # broadcast to (B, C) - same interaction for all classes

            g = softmax_grad(logits, yb)  # (B, C)

            # Gradients
            g_bias = g.sum(axis=0)
            g_lin = np.zeros_like(lin)
            g_V = np.zeros_like(V)

            for j in range(d):
                xj = xb[:, j]
                # linear grad: for each class, sum over batch
                for c in range(C):
                    g_lin[j, xj, c] += g[:, c]
                # V grad: g_fm = sum_c g[:,c] = g.sum(axis=1) -> (B,)
                g_fm = g.sum(axis=1)  # (B,)
                vj = V[j, xj, :]  # (B, K)
                # d/dV_jk = g_fm * (sum_V_k - vj_k)
                for k in range(K):
                    g_V[j, xj, k] += g_fm * (sum_V[:, k] - vj[:, k])

            # L2 regularization
            g_lin += L2_LIN * lin
            g_V += L2_FAC * V

            t += 1
            b1, b2 = 0.9, 0.999
            # bias
            m_bias = b1 * m_bias + (1 - b1) * g_bias
            v_bias = b2 * v_bias + (1 - b2) * g_bias * g_bias
            m_hat = m_bias / (1 - b1 ** t)
            v_hat = v_bias / (1 - b2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            # lin
            m_lin = b1 * m_lin + (1 - b1) * g_lin
            v_lin = b2 * v_lin + (1 - b2) * g_lin * g_lin
            m_hat = m_lin / (1 - b1 ** t)
            v_hat = v_lin / (1 - b2 ** t)
            lin -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            # V
            m_V = b1 * m_V + (1 - b1) * g_V
            v_V = b2 * v_V + (1 - b2) * g_V * g_V
            m_hat = m_V / (1 - b1 ** t)
            v_hat = v_V / (1 - b2 ** t)
            V -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class index (argmax).
 * Weights: magic 0x464D4C31, int32 d,K,C; float32 bias[C], linear[d][256][C], V[d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || d > 256 || K < 1 || K > 256) {
        fprintf(stderr, "bad dims\n"); return 3;
    }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *lin = bias + C;
    const float *V = lin + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = bias[c];
        for (int kk = 0; kk < K; kk++) sum_V[kk] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_jv = lin + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) logits[c] += lin_jv[c];
            const float *V_jv = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) sum_V[kk] += V_jv[kk];
        }

        float fm = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float sv = sum_V[kk];
            fm += sv * sv;
        }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *V_jv = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float vjk = V_jv[kk];
                fm -= vjk * vjk;
            }
        }
        fm *= 0.5f;
        for (int c = 0; c < C; c++) logits[c] += fm;

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```