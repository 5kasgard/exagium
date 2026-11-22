```hypothesis
Family: label-fm (Factorization Machine degree-2). Why: Adult has strong categorical interactions (education×occupation, marital-status×relationship); FM captures all pairwise interactions with O(d·K) params (d=14, K=8) — more expressive than linear, cheaper than MLP/GBDT. Quantile-binned ordinals work naturally as categorical. Failure: dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear.
```
```python
"""Factorization Machine (degree 2) for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
All 14 input bytes treated as categorical (0-255). FM: logit = w0 + sum_i w[i,xi] + 0.5*sum_k((sum_i V[i,xi,k])^2 - sum_i V[i,xi,k]^2).
Adam lr=1e-2, L2=1e-5 on w/V, minibatch 256, epochs min(50, ceil(300000/n)), seed=1, deterministic."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 256, 300000, 50
MAGIC = 0x464D3231  # "FM21" little-endian

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
    assert C == 2, "FM binary only"
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # params: w0 (1), w (d, 256), V (d, 256, K)
    w0 = np.float32(0.0)
    w = (rng.normal(0, 0.01, (d, 256)) * math.sqrt(1.0 / 256)).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K)) * math.sqrt(1.0 / (256 * K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]           # (b, d)
            yi = y[idx]           # (b,)
            b = xi.shape[0]

            # Linear term: w0 + sum_i w[i, xi]
            lin = w0 + np.sum(w[np.arange(d), xi.T], axis=0)  # (b,)

            # FM interaction term
            # sum_k ( (sum_i V[i,xi,k])^2 - sum_i V[i,xi,k]^2 ) * 0.5
            # V_xi shape: (b, d, K)
            V_xi = V[np.arange(d), xi.T]  # (d, b, K) -> transpose to (b, d, K)
            V_xi = np.transpose(V_xi, (1, 0, 2))
            sum_V = np.sum(V_xi, axis=1)           # (b, K)
            sum_V2 = np.sum(V_xi * V_xi, axis=1)   # (b, K)
            inter = 0.5 * np.sum(sum_V * sum_V - sum_V2, axis=1)  # (b,)

            logit = lin + inter
            # Sigmoid + BCE gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yi).astype(np.float32)  # (b,)

            # Gradients
            # w0
            gw0 = np.mean(grad) + L2 * w0
            # w: (d, 256)
            gw = np.zeros_like(w)
            for i in range(d):
                np.add.at(gw[i], xi[:, i], grad)
            gw = (gw / b) + L2 * w
            # V: (d, 256, K)
            gV = np.zeros_like(V)
            # For each sample, each feature, each k: grad * (sum_V[b,k] - V[i,xi,k])
            # Vectorized: (b, d, K) * (b, 1, K) -> (b, d, K)
            sum_V_exp = sum_V[:, None, :]  # (b, 1, K)
            V_xi_exp = V_xi                # (b, d, K)
            gV_xi = grad[:, None, None] * (sum_V_exp - V_xi_exp)  # (b, d, K)
            for i in range(d):
                np.add.at(gV[i], xi[:, i], gV_xi[:, i, :])
            gV = (gV / b) + L2 * V

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Export: magic, d, K, C, w0, w[d][256], V[d][256][K] all float32 little-endian
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += np.array([w0], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree 2) inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1).
 * Weights format: int32 magic(0x464D3231), d, K, C; float32 w0, w[d][256], V[d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0 || d > 256 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;                    // [d][256]
    const float *V = p + 1 + (long)d * 256;    // [d][256][K]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear term
        float lin = w0;
        for (int j = 0; j < d; j++) lin += w[(long)j * 256 + xi[j]];

        // Interaction term: 0.5 * sum_k (sum_V[k]^2 - sum_V2[k])
        memset(sum_V, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) sum_V[kk] += v[kk];
        }
        float inter = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float s = sum_V[kk];
                inter += 0.5f * (s * s - v[kk] * v[kk]);
            }
        }
        // The above double-counts: for each j we add 0.5*(s^2 - v_j^2), but s includes v_j.
        // Correct FM trick: 0.5 * sum_k ( (sum_i v_ik)^2 - sum_i v_ik^2 )
        // Compute correctly:
        inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float s = sum_V[kk];
            float s2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[((long)j * 256 + xi[j]) * K + kk];
                s2 += v * v;
            }
            inter += 0.5f * (s * s - s2);
        }

        float logit = lin + inter;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```