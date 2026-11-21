```hypothesis
Propose a factorization machine (FM) of rank 4. It adds low-rank pairwise interactions to a linear model, capturing feature combinations that logistic regression misses. On tabular data with mixed categorical/ordinal features like Adult, FMs often outperform linear models with only a modest increase in inference cost. If dev accuracy does not exceed label-linear (0.8524) or inference cost per item is more than 2× label-linear, the family does not meet the goal.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen a priori)
K = 4              # FM rank
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 10
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 14
    C = card["classes"]                 # 2

    # Load data
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    bias = np.float32(0.0)
    L = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_L, v_L = np.zeros_like(L), np.zeros_like(L)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Gather embeddings
            L_emb = L[np.arange(d), xb.T].T          # (B, d)
            V_emb = V[np.arange(d), xb.T].transpose(1,0,2)  # (B, d, K)

            # Forward
            linear = bias + L_emb.sum(axis=1)                         # (B,)
            sum_V = V_emb.sum(axis=1)                                 # (B, K)
            sum_V_sq = (V_emb**2).sum(axis=1)                         # (B, K)
            inter = 0.5 * ((sum_V**2 - sum_V_sq).sum(axis=1))         # (B,)
            logit = linear + inter                                    # (B,)

            # Logistic loss gradient
            prob = 1.0 / (1.0 + np.exp(-logit))
            grad = (prob - yb).astype(np.float32)                     # (B,)

            # Gradients
            g_bias = grad.mean()
            g_L = np.zeros_like(L)
            g_V = np.zeros_like(V)

            # Linear part gradients
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_L[j], vals, grad)

            # Interaction gradients
            for j in range(d):
                vals = xb[:, j]
                for f in range(K):
                    # d inter / d V_{j, v, f} = sum_V[:, f] - V_emb[:, j, f]
                    g = grad * (sum_V[:, f] - V_emb[:, j, f])
                    np.add.at(g_V[j, :, f], vals, g)

            # L2 regularization
            g_L += L2 * L
            g_V += L2 * V

            # Adam update
            t += 1
            # bias
            m_bias = BETA1 * m_bias + (1 - BETA1) * g_bias
            v_bias = BETA2 * v_bias + (1 - BETA2) * g_bias * g_bias
            m_hat = m_bias / (1 - BETA1**t)
            v_hat = v_bias / (1 - BETA2**t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # L
            m_L = BETA1 * m_L + (1 - BETA1) * g_L
            v_L = BETA2 * v_L + (1 - BETA2) * g_L * g_L
            m_hat_L = m_L / (1 - BETA1**t)
            v_hat_L = v_L / (1 - BETA2**t)
            L -= LR * m_hat_L / (np.sqrt(v_hat_L) + EPS)

            # V
            m_V = BETA1 * m_V + (1 - BETA1) * g_V
            v_V = BETA2 * v_V + (1 - BETA2) * g_V * g_V
            m_hat_V = m_V / (1 - BETA1**t)
            v_hat_V = v_V / (1 - BETA2**t)
            V -= LR * m_hat_V / (np.sqrt(v_hat_V) + EPS)

    # Write weights.bin
    # Magic: 0x464D31 ('FM1'), d, K, C
    header = struct.pack("<4i", 0x464D31, d, K, C)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(bias.astype("<f4").tobytes())
        f.write(L.astype("<f4").tobytes())
        f.write(V.astype("<f4").tobytes())

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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *L = bias + 1;
    const float *V = L + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_V[16], sum_V_sq[16];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = bias[0];
        for (int f = 0; f < K; f++) { sum_V[f] = 0.0f; sum_V_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += L[(long)j * 256 + v];
            const float *Vj = V + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = Vj[f];
                sum_V[f] += val;
                sum_V_sq[f] += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += sum_V[f] * sum_V[f] - sum_V_sq[f];
        inter *= 0.5f;
        float logit = linear + inter;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```