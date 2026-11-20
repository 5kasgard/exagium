```hypothesis
Family: label-fm (Factorization Machine). Adult has mixed categorical/ordinal features where pairwise interactions (e.g., education × occupation) are predictive. FM captures all pairwise interactions with O(d*k) parameters (k=8) and computes them in O(d*k) time per item, much cheaper than a 16-unit MLP. If dev accuracy ≤ label-linear (0.8524) or Ir/item > 2000, the model fails its goal.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
SAMPLES = 600000
MAX_EPOCHS = 50
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
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)  # 0 or 1
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    # Flat index for each feature value
    idx = np.empty_like(X, dtype=np.int32)
    for j in range(d):
        idx[:, j] = j * 256 + X[:, j]

    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.float32(0.0)
    W = np.zeros(d * 256, dtype=np.float32)
    V = rng.normal(0, math.sqrt(2.0 / (d * 256)), size=(d * 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    beta1, beta2, eps = 0.9, 0.999, 1e-8

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            b = len(batch_idx)
            idx_b = idx[batch_idx]  # (b, d)
            y_b = y[batch_idx]      # (b,)

            # Forward
            linear = bias + np.sum(W[idx_b], axis=1)  # (b,)
            V_b = V[idx_b]  # (b, d, K)
            S = np.sum(V_b, axis=1)  # (b, K)
            sum_sq = np.sum(V_b * V_b, axis=1)  # (b, K)
            fm = 0.5 * np.sum(S * S - sum_sq, axis=1)  # (b,)
            logit = linear + fm
            p = 1.0 / (1.0 + np.exp(-logit))
            delta = p - y_b  # (b,)

            # Gradients
            grad_bias = np.mean(delta)
            grad_W = np.zeros_like(W)
            np.add.at(grad_W, idx_b.ravel(), np.repeat(delta, d))
            grad_W /= b
            grad_V = np.zeros_like(V)
            for j in range(d):
                idx_j = idx_b[:, j]  # (b,)
                V_j = V_b[:, j, :]   # (b, K)
                factor = delta[:, None] * (S - V_j)  # (b, K)
                np.add.at(grad_V, idx_j, factor)
            grad_V /= b

            # L2 regularization
            grad_W += L2 * W
            grad_V += L2 * V

            # Adam update
            t += 1
            m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1 - beta2) * grad_bias * grad_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W -= LR * m_hat_W / (np.sqrt(v_hat_W) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= LR * m_hat_V / (np.sqrt(v_hat_V) + eps)

    # Write weights
    magic = 0x464D3132  # "FM12"
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += struct.pack("<f", bias)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    unsigned char *weights_data = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)weights_data;
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2) { fprintf(stderr, "only binary classification supported\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *W = bias + 1;
    const float *V = W + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias[0];
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            logit += W[idx];
        }
        for (int f = 0; f < K; f++) {
            S[f] = 0.0f;
            sum_sq[f] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            const float *v = V + (long)idx * K;
            for (int f = 0; f < K; f++) {
                float val = v[f];
                S[f] += val;
                sum_sq[f] += val * val;
            }
        }
        for (int f = 0; f < K; f++) {
            logit += 0.5f * (S[f] * S[f] - sum_sq[f]);
        }

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```