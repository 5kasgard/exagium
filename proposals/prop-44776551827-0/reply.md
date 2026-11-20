```hypothesis
We propose a Factorization Machine (FM) with embedding dimension 8. FMs capture pairwise feature interactions via low-rank embeddings, which is crucial for tabular data with mixed categorical/ordinal features. The model has ~32k parameters (w0, linear weights, embeddings) and inference costs O(d*k) with d=14, k=8. It should beat label-linear (0.8524) by modeling interactions, approaching label-mlp16 (0.8591) at lower CPU cost. If dev accuracy is ≤0.8524 or instructions/item exceed label-mlp16 (~2300), the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D3130, d, k, C; float32 w0, W_lin[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 8  # embedding dimension

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
    # Parameters
    w0 = np.float32(0.0)
    W_lin = np.zeros((d, 256), dtype=np.float32)
    V = (rng.uniform(-0.01, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_W = np.zeros_like(W_lin); v_W = np.zeros_like(W_lin)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Linear part
            lin = w0 + W_lin[np.arange(d), xb.T].sum(axis=0)   # (B,)

            # FM part: gather embeddings (B, d, K)
            V_batch = V[np.arange(d), xb.T].transpose(1, 0, 2)
            sum_v = V_batch.sum(axis=1)                 # (B, K)
            sum_sq_v = (V_batch ** 2).sum(axis=1)       # (B, K)
            inter = 0.5 * ((sum_v ** 2).sum(axis=1) - sum_sq_v.sum(axis=1))  # (B,)

            score = lin + inter
            # Logistic loss gradient
            sig = 1.0 / (1.0 + np.exp(-score))
            grad = (sig - yb).astype(np.float32)        # (B,)

            # Gradients
            g_w0 = grad.mean()
            g_W = np.zeros_like(W_lin)
            g_V = np.zeros_like(V)

            # Linear weight gradients
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_W[j], vals, grad)

            # Embedding gradients
            # dL/dV_ijk = grad_i * (sum_v_i - V_ijk)
            for j in range(d):
                v_j = V_batch[:, j, :]                  # (B, K)
                diff = sum_v - v_j                      # (B, K)
                g_vj = grad[:, None] * diff             # (B, K)
                vals = xb[:, j]
                for k in range(K):
                    np.add.at(g_V[j, :, k], vals, g_vj[:, k])

            # Add L2
            g_w0 += L2 * w0
            g_W += L2 * W_lin
            g_V += L2 * V

            # Adam update
            t += 1
            # w0
            m_w0 = 0.9 * m_w0 + 0.1 * g_w0
            v_w0 = 0.999 * v_w0 + 0.001 * g_w0 * g_w0
            m_hat = m_w0 / (1 - 0.9 ** t)
            v_hat = v_w0 / (1 - 0.999 ** t)
            w0 -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            # W_lin
            m_W = 0.9 * m_W + 0.1 * g_W
            v_W = 0.999 * v_W + 0.001 * g_W * g_W
            m_hat = m_W / (1 - 0.9 ** t)
            v_hat = v_W / (1 - 0.999 ** t)
            W_lin -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            # V
            m_V = 0.9 * m_V + 0.1 * g_V
            v_V = 0.999 * v_V + 0.001 * g_V * g_V
            m_hat = m_V / (1 - 0.9 ** t)
            v_hat = v_V / (1 - 0.999 ** t)
            V -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D3130  # 'FM10'
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += np.array([w0], dtype="<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3130) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *W_lin = p + 1;
    const float *V = W_lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        float sum_v[8] = {0};
        float sum_sq_v[8] = {0};

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += W_lin[(long)j * 256 + v];
            const float *emb = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = emb[kk];
                sum_v[kk] += val;
                sum_sq_v[kk] += val * val;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            inter += sum_v[kk] * sum_v[kk] - sum_sq_v[kk];
        }
        score += 0.5f * inter;

        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```