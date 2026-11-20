```hypothesis
Factorization Machine on raw byte features (14 fields × 256 values) with K=8 embeddings.
Captures pairwise interactions (e.g., education × occupation) crucial for Adult, while the FM trick
keeps inference at O(d·K) ≈ 112 FLOPs/item. Trained with Adam (lr=0.01, L2=1e-4, 30 epochs).
If dev accuracy ≤ 0.8524, interactions aren't sufficiently exploited or regularization too strong.
```
```python
"""Factorization Machine for tabular byte features: ./train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only. Learns per-field per-value linear weight
and K-dim embedding; uses FM trick for O(d·K) interaction term. Adam lr=0.01, L2=1e-4, batch=256."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 30

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
    n, d = X.shape
    C = card["classes"]
    assert C == 2 and d == 14

    rng = np.random.default_rng(SEED)
    # Parameters: w0, W_lin[d,256], Emb[d,256,K]
    w0 = np.log((y == 1).sum() / max(1, (y == 0).sum())).astype(np.float32)
    W_lin = np.zeros((d, 256), np.float32)
    Emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_W, v_W = np.zeros_like(W_lin), np.zeros_like(W_lin)
    m_E, v_E = np.zeros_like(Emb), np.zeros_like(Emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Linear term: w0 + sum_j W_lin[j, xb[:,j]]
            lin = w0 + np.sum(W_lin[np.arange(d), xb.T], axis=0)   # (B,)

            # FM interaction: 0.5 * (||sum_j e_j||^2 - sum_j ||e_j||^2)
            # e_j shape (B, K)
            e = Emb[np.arange(d), xb.T]          # (d, B, K) -> transpose to (B, d, K)
            e = e.transpose(1, 0, 2)             # (B, d, K)
            sum_e = e.sum(axis=1)                # (B, K)
            sum_sq = (e * e).sum(axis=1)         # (B, K)
            inter = 0.5 * ((sum_e * sum_e - sum_sq).sum(axis=1))   # (B,)

            logits = lin + inter
            p = sigmoid(logits)
            grad = (p - yb).astype(np.float32)   # (B,)

            # Gradients
            g_w0 = grad.mean()
            g_W = np.zeros_like(W_lin)
            g_E = np.zeros_like(Emb)

            # Linear gradients
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_W[j], vals, grad)
            g_W /= B

            # Embedding gradients
            # For each sample i, for each field j: dL/de_{ij} = grad[i] * (sum_e[i] - e_{ij})
            # sum_e[i] is (K,), e_{ij} is (K,)
            for j in range(d):
                vals = xb[:, j]
                # g_E[j, v] += sum_{i: xb[i,j]==v} grad[i] * (sum_e[i] - Emb[j, v])
                for v in range(256):
                    mask = (vals == v)
                    if not np.any(mask):
                        continue
                    g = grad[mask]                       # (m,)
                    diff = sum_e[mask] - Emb[j, v]       # (m, K)
                    g_E[j, v] += (g[:, None] * diff).sum(axis=0)
            g_E /= B

            # L2 regularization
            g_W += L2 * W_lin
            g_E += L2 * Emb

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_hat = m_w0 / (1 - beta1 ** t)
            v_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # W_lin
            m_W = beta1 * m_W + (1 - beta1) * g_W
            v_W = beta2 * v_W + (1 - beta2) * g_W * g_W
            m_hat = m_W / (1 - beta1 ** t)
            v_hat = v_W / (1 - beta2 ** t)
            W_lin -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # Emb
            m_E = beta1 * m_E + (1 - beta1) * g_E
            v_E = beta2 * v_E + (1 - beta2) * g_E * g_E
            m_hat = m_E / (1 - beta1 ** t)
            v_hat = v_E / (1 - beta2 ** t)
            Emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin: magic, d, K, C, w0, W_lin, Emb
    magic = 0x464D4C31  # FML1
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0)
    out += W_lin.astype("<f4").tobytes()
    out += Emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output: 1 byte class (0/1).
 * Weights format: magic(0x464D4C31), d, K, C, w0, W_lin[d*256], Emb[d*256*K] all float32 little-endian. */
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
    if (d != 14 || C != 2 || K <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *W_lin = p + 1;
    const float *Emb = W_lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);
    float *e_ptr = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize accumulators
        for (int u = 0; u < K; u++) { sum_e[u] = 0.0f; sum_sq[u] = 0.0f; }
        float linear = w0;

        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            linear += W_lin[(long)j * 256 + v];
            const float *emb = Emb + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = emb[u];
                sum_e[u] += val;
                sum_sq[u] += val * val;
            }
        }

        float inter = 0.0f;
        for (int u = 0; u < K; u++) {
            float diff = sum_e[u] * sum_e[u] - sum_sq[u];
            inter += diff;
        }
        inter *= 0.5f;

        float score = linear + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```