```hypothesis
Factorization Machine (FM) with rank-8 pairwise interactions on the 14 byte-wise categorical features.
FM captures non-linear feature interactions while keeping inference fast (O(d*K) per item).
On Adult, pairwise interactions are known to boost accuracy over linear models.
If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds label-linear (719 Ir/item), the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each of the d bytes as a categorical feature with 256 categories.
Learns a linear weight per category, a global bias, and a factor vector (rank K) per category.
Adam lr=0.01, L2=1e-4, minibatch=64, epochs=50.
Exported weights: int32 magic 0x464D3130, d, K, C; float32 linear[d*256], bias, factors[d*256*K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 50
K = 8  # FM rank


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
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    n = X.shape[0]

    # Parameters: linear[d*256], bias, factors[d*256*K]
    linear = rng.normal(0, 0.01, size=(d * 256,)).astype(np.float32)
    bias = np.float32(0.0)
    factors = rng.normal(0, 0.01, size=(d * 256, K)).astype(np.float32)

    # Adam state
    m_linear, v_linear = np.zeros_like(linear), np.zeros_like(linear)
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_factors, v_factors = np.zeros_like(factors), np.zeros_like(factors)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            # Gradients
            g_linear = np.zeros_like(linear)
            g_bias = np.float32(0.0)
            g_factors = np.zeros_like(factors)

            for i in range(bs):
                xi = xb[i]
                yi = yb[i]

                # Forward pass
                lin_term = bias
                f_sum = np.zeros(K, dtype=np.float32)
                f_sq_sum = np.zeros(K, dtype=np.float32)
                for j in range(d):
                    cat = xi[j]
                    idx_flat = j * 256 + cat
                    lin_term += linear[idx_flat]
                    f_vec = factors[idx_flat]
                    f_sum += f_vec
                    f_sq_sum += f_vec * f_vec
                inter = 0.5 * np.sum(f_sum * f_sum - f_sq_sum)
                logit = lin_term + inter
                p = 1.0 / (1.0 + math.exp(-logit))
                d_logit = p - yi  # gradient of logistic loss w.r.t logit

                # Backward pass
                g_bias += d_logit
                for j in range(d):
                    cat = xi[j]
                    idx_flat = j * 256 + cat
                    g_linear[idx_flat] += d_logit
                    f_vec = factors[idx_flat]
                    # d_inter/d_f_vec = f_sum - f_vec
                    g_factors[idx_flat] += d_logit * (f_sum - f_vec)

            # Average gradients + L2
            g_linear = (g_linear / bs) + L2 * linear
            g_bias = g_bias / bs
            g_factors = (g_factors / bs) + L2 * factors

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_linear = beta1 * m_linear + (1 - beta1) * g_linear
            v_linear = beta2 * v_linear + (1 - beta2) * (g_linear * g_linear)
            m_hat = m_linear / (1 - beta1 ** t)
            v_hat = v_linear / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)

            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * (g_bias * g_bias)
            m_hat_b = m_bias / (1 - beta1 ** t)
            v_hat_b = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat_b / (math.sqrt(v_hat_b) + eps)

            m_factors = beta1 * m_factors + (1 - beta1) * g_factors
            v_factors = beta2 * v_factors + (1 - beta2) * (g_factors * g_factors)
            m_hat_f = m_factors / (1 - beta1 ** t)
            v_hat_f = v_factors / (1 - beta2 ** t)
            factors -= LR * m_hat_f / (np.sqrt(v_hat_f) + eps)

    # Write weights.bin
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += np.array([bias], dtype="<f4").tobytes()
    out += factors.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (categorical indices 0..255). Output: one byte class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 + 1 + (long)d * 256 * K;
    if (nw != 16 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float bias = linear[(long)d * 256];
    const float *factors = linear + (long)d * 256 + 1;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *f_sum = malloc(sizeof(float) * K);
    float *f_sq_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_term = bias;
        for (int u = 0; u < K; u++) { f_sum[u] = 0.0f; f_sq_sum[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            int cat = xi[j];
            long idx = (long)j * 256 + cat;
            lin_term += linear[idx];
            const float *f_vec = factors + idx * K;
            for (int u = 0; u < K; u++) {
                float v = f_vec[u];
                f_sum[u] += v;
                f_sq_sum[u] += v * v;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += f_sum[u] * f_sum[u] - f_sq_sum[u];
        inter *= 0.5f;
        float logit = lin_term + inter;
        unsigned char best = (logit >= 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```