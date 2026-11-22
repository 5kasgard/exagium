```hypothesis
Family: label-fm (factorization machine with per-value embeddings). Captures pairwise feature interactions via low-rank embeddings (k=8), which are crucial for tabular data with mixed ordinal/categorical features. The model adds only O(d*256*k) parameters (~32k) and O(d*k) inference cost, staying fast. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds label-linear by >2x, the approach fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
K = 8                # embedding dimension
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 20
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x464D4C31   # 'FML1'

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

    # Parameters: bias, W_lin (d*256), V (d*256, K)
    bias = np.array(0.0, dtype=np.float32)
    W_lin = np.zeros(d * 256, dtype=np.float32)
    V = (rng.normal(0, 0.01, size=(d * 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_W = np.zeros_like(W_lin); v_W = np.zeros_like(W_lin)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)

    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            # Linear part
            W_batch = W_lin[xb.ravel()].reshape(B, d)          # (B, d)
            linear = W_batch.sum(axis=1)                       # (B,)

            # Interaction part
            V_batch = V[xb.ravel()].reshape(B, d, K)           # (B, d, K)
            sum_V = V_batch.sum(axis=1)                        # (B, K)
            sum_V_sq = (V_batch ** 2).sum(axis=1)              # (B, K)
            interaction = 0.5 * ((sum_V ** 2 - sum_V_sq).sum(axis=1))  # (B,)

            logit = bias + linear + interaction                # (B,)
            p = 1.0 / (1.0 + np.exp(-logit))                   # (B,)

            # Gradient of loss w.r.t logit
            dlogit = (p - yb).astype(np.float32)               # (B,)

            # Gradients
            dbias = dlogit.mean()

            # Gradient for W_lin: scatter add dlogit per feature value
            grad_W = np.zeros_like(W_lin)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(grad_W, j * 256 + vals, dlogit)

            # Gradient for V: for each feature j, grad_V[j, val] += dlogit * (sum_V - V_batch[:, j])
            grad_V = np.zeros_like(V)
            for j in range(d):
                vals = xb[:, j]
                V_j = V_batch[:, j, :]                         # (B, K)
                diff = sum_V - V_j                             # (B, K)
                for l in range(K):
                    np.add.at(grad_V[:, l], j * 256 + vals, dlogit * diff[:, l])

            # L2 regularization
            grad_W += L2 * W_lin
            grad_V += L2 * V

            # Adam updates
            t += 1
            # bias
            m_bias = BETA1 * m_bias + (1 - BETA1) * dbias
            v_bias = BETA2 * v_bias + (1 - BETA2) * (dbias * dbias)
            m_hat = m_bias / (1 - BETA1 ** t)
            v_hat = v_bias / (1 - BETA2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # W_lin
            m_W = BETA1 * m_W + (1 - BETA1) * grad_W
            v_W = BETA2 * v_W + (1 - BETA2) * (grad_W * grad_W)
            m_hat_W = m_W / (1 - BETA1 ** t)
            v_hat_W = v_W / (1 - BETA2 ** t)
            W_lin -= LR * m_hat_W / (np.sqrt(v_hat_W) + EPS)

            # V
            m_V = BETA1 * m_V + (1 - BETA1) * grad_V
            v_V = BETA2 * v_V + (1 - BETA2) * (grad_V * grad_V)
            m_hat_V = m_V / (1 - BETA1 ** t)
            v_hat_V = v_V / (1 - BETA2 ** t)
            V -= LR * m_hat_V / (np.sqrt(v_hat_V) + EPS)

    # Write weights.bin
    # Format: magic, d, K, C (int32), bias (float32), W_lin (float32), V (float32)
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, class index (0 or 1). */
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
    if (C != 2 || K <= 0 || d <= 0 || nw != 16 + 4 + 4L * d * 256 * (1 + K)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *p = (const float *)(hd + 4);
    const float bias = p[0];
    const float *W_lin = p + 1;
    const float *V = W_lin + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);
    float *sum_V_sq = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear_sum = 0.0f;
        for (int l = 0; l < K; l++) { sum_V[l] = 0.0f; sum_V_sq[l] = 0.0f; }
        for (int j = 0; j < d; j++) {
            int val = xi[j];
            linear_sum += W_lin[(long)j * 256 + val];
            const float *v = V + ((long)j * 256 + val) * K;
            for (int l = 0; l < K; l++) {
                float vv = v[l];
                sum_V[l] += vv;
                sum_V_sq[l] += vv * vv;
            }
        }
        float logit = bias + linear_sum;
        for (int l = 0; l < K; l++) {
            logit += 0.5f * (sum_V[l] * sum_V[l] - sum_V_sq[l]);
        }
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```