```hypothesis
Family: label-fm (factorization machine). It models pairwise feature interactions via low-rank embeddings, which are crucial for tabular data like Adult where combinations of categorical values (e.g., education × occupation) are predictive. Logistic regression (label-linear) misses these interactions; FM adds them with only O(d * k) extra parameters (k=8). Inference remains fast: O(d * k) per item. If dev accuracy does not exceed label-linear's 0.8524, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each of the 14 input bytes as a categorical feature with 256 possible values.
Learns a bias w0, per-value weights w[j][v], and per-value embeddings V[j][v] (dim=k).
Optimizes binary cross-entropy with Adam (lr=0.01, L2=1e-4, batch=256, epochs=50).
Exports: magic 0x464D0001, d=14, k, C=2; float32 w0, w[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen a priori; could be tuned on last 20% of train.in)
SEED = 1
K = 8                # embedding dimension
LR = 0.01
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 50
MAGIC = 0x464D0001   # 'FM\x00\x01'

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
    assert d == 14 and C == 2

    # Load data: X (n, d) uint8, y (n,) int64
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 1.0 / math.sqrt(K), size=(d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Gather weights and embeddings for each feature
            w_vals = np.zeros((B, d), dtype=np.float32)
            V_vals = np.zeros((B, d, K), dtype=np.float32)
            for j in range(d):
                w_vals[:, j] = w[j, xb[:, j]]
                V_vals[:, j, :] = V[j, xb[:, j], :]

            sum_w = w0 + w_vals.sum(axis=1)                     # (B,)
            sum_v = V_vals.sum(axis=1)                          # (B, K)
            sum_v_sq = (V_vals ** 2).sum(axis=1)                # (B, K)
            interaction = 0.5 * ((sum_v ** 2 - sum_v_sq).sum(axis=1))  # (B,)
            logit = sum_w + interaction                         # (B,)
            prob = sigmoid(logit)
            dlogit = prob - yb                                  # (B,)

            # Gradients
            grad_w0 = dlogit.sum()
            grad_w = np.zeros_like(w)
            grad_V = np.zeros_like(V)
            for j in range(d):
                # weight gradient
                np.add.at(grad_w[j], xb[:, j], dlogit)
                # embedding gradient: dlogit * (sum_v - V_j)
                # sum_v is (B, K), V_vals[:,j] is (B, K)
                g = dlogit[:, None] * (sum_v - V_vals[:, j, :])  # (B, K)
                np.add.at(grad_V[j], xb[:, j], g)

            # L2 regularization
            grad_w += L2 * w
            grad_V += L2 * V

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (grad_w0 * grad_w0)
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * (grad_w * grad_w)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * (grad_V * grad_V)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Pack weights: magic, d, K, C; then w0, w (d*256), V (d*256*K)
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: 14 bytes (one per feature). Output: one byte class (0 or 1).
 * Weights format: magic 0x464D0001, int32 d, K, C; float32 w0, w[d*256], V[d*256*K]. */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || C != 2 || K <= 0 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float sum_w = w0;
        float sum_v[64] = {0};
        float sum_v_sq[64] = {0};

        for (int j = 0; j < d; j++) {
            unsigned char b = xi[j];
            sum_w += w[(long)j * 256 + b];
            const float *v_ptr = V + ((long)j * 256 + b) * K;
            for (int u = 0; u < K; u++) {
                float v = v_ptr[u];
                sum_v[u] += v;
                sum_v_sq[u] += v * v;
            }
        }
        float interaction = 0.0f;
        for (int u = 0; u < K; u++) {
            interaction += sum_v[u] * sum_v[u] - sum_v_sq[u];
        }
        interaction *= 0.5f;
        float logit = sum_w + interaction;
        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```