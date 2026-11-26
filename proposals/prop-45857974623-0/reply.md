```hypothesis
Family: label-fm (Factorization Machine). Adult has many categorical features; FM captures pairwise interactions with low-rank factors, improving over linear while keeping inference O(d*D). Chosen D=8, Adam lr=0.01, L2=1e-4, 10 epochs. Failure if dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2x label-linear.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 10
D = 8          # factor dimension
MAGIC = 0x464D4C31  # 'FML1'

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
    assert d == 14
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    # Parameters: bias, linear[d][256], factor[d][256][D]
    bias = np.float32(0.0)
    L = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_L, v_L = np.zeros_like(L), np.zeros_like(L)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    y_pm = (y * 2 - 1).astype(np.float32)  # -1 / +1

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y_pm[idx]       # (b,)
            b = xb.shape[0]

            # Gather embeddings
            L_emb = L[np.arange(d), xb.T]          # (d, b)
            V_emb = V[np.arange(d), xb.T, :]       # (d, b, D)

            # Linear part
            lin = bias + L_emb.sum(axis=0)         # (b,)

            # Interaction part
            sum_V = V_emb.sum(axis=0)              # (b, D)
            sum_V_sq = (V_emb ** 2).sum(axis=0)    # (b, D)
            inter = 0.5 * ((sum_V ** 2 - sum_V_sq).sum(axis=1))  # (b,)

            score = lin + inter                    # (b,)
            # Logistic loss gradient
            sig = 1.0 / (1.0 + np.exp(-yb * score))
            grad_loss = -yb * (1.0 - sig)          # (b,)

            # Gradients
            g_bias = grad_loss.mean()
            g_L = np.zeros_like(L)
            g_V = np.zeros_like(V)

            # For each feature j, we need to accumulate gradients for the used categories
            # We'll use a loop over features (d=14 small) for clarity and memory
            for j in range(d):
                cats = xb[:, j]                    # (b,)
                # Linear gradient
                np.add.at(g_L[j], cats, grad_loss / b)
                # Factor gradient
                # inter gradient w.r.t V_emb[j] = grad_loss[:,None] * (sum_V - V_emb[j])
                factor_grad = grad_loss[:, None] * (sum_V - V_emb[j]) / b   # (b, D)
                np.add.at(g_V[j], cats, factor_grad)

            # L2 regularization
            g_L += L2 * L
            g_V += L2 * V

            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # L
            m_L = beta1 * m_L + (1 - beta1) * g_L
            v_L = beta2 * v_L + (1 - beta2) * g_L * g_L
            m_hat_L = m_L / (1 - beta1 ** t)
            v_hat_L = v_L / (1 - beta2 ** t)
            L -= LR * m_hat_L / (np.sqrt(v_hat_L) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= LR * m_hat_V / (np.sqrt(v_hat_V) + eps)

    # Write weights.bin
    # Header: magic, d, D, C (all int32)
    header = struct.pack("<4i", MAGIC, d, D, C)
    # bias (float32)
    bias_bytes = np.array([bias], dtype=np.float32).tobytes()
    # linear weights: d * 256 floats (row-major)
    L_bytes = L.astype("<f4").tobytes()
    # factor vectors: d * 256 * D floats (row-major: j, cat, f)
    V_bytes = V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + bias_bytes + L_bytes + V_bytes)

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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 14 || D != 8 || C != 2) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * D;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *L = p + 1;
    const float *V = L + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_V[8], sum_V_sq[8];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias;
        for (int f = 0; f < D; f++) { sum_V[f] = 0.0f; sum_V_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *L_row = L + (long)j * 256;
            const float *V_row = V + ((long)j * 256 + v) * D;
            score += L_row[v];
            for (int f = 0; f < D; f++) {
                float val = V_row[f];
                sum_V[f] += val;
                sum_V_sq[f] += val * val;
            }
        }
        for (int f = 0; f < D; f++) {
            score += 0.5f * (sum_V[f] * sum_V[f] - sum_V_sq[f]);
        }

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```