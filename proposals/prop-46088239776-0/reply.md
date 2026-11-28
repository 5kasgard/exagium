```hypothesis
Family: label-fm (factorization machine with small embeddings). Adult has 14 mixed-type features; pairwise interactions are known to help on this dataset. A rank-4 FM adds ~72k parameters (vs ~14k for linear) and O(d^2*k) inference (14^2*4=784 ops), keeping CPU cost low. If dev accuracy does not exceed label-linear's 0.8524, the interactions are not beneficial at this rank or the optimization fails.
```

```python
"""label-fm: Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, k, C; float32 bias, W[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 4              # embedding rank
SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 256
EPOCHS = 20

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
    # Parameters: bias, W[d,256], V[d,256,K]
    bias = np.float32(0.0)
    W = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    # Precompute indices for gathering
    idx = X  # (n, d)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s+BATCH]
            xb = idx[batch_idx]          # (b, d)
            yb = y[batch_idx]            # (b,)
            b = xb.shape[0]

            # Gather embeddings
            W_x = W[np.arange(d), xb.T].T          # (b, d)
            V_x = V[np.arange(d), xb.T].transpose(1,0,2)  # (b, d, K)

            # Linear part
            linear = bias + W_x.sum(axis=1)        # (b,)

            # Interaction part: 0.5 * ( (sum V)^2 - sum V^2 )
            sum_V = V_x.sum(axis=1)                # (b, K)
            sum_V2 = (V_x**2).sum(axis=1)          # (b, K)
            inter = 0.5 * ((sum_V**2 - sum_V2).sum(axis=1))  # (b,)

            logits = linear + inter                # (b,)
            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))      # (b,)
            grad = (p - yb).astype(np.float32)     # (b,)

            # Gradients
            g_bias = grad.mean()
            g_W = np.zeros_like(W)
            g_V = np.zeros_like(V)

            # Vectorized scatter-add
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_W[j], vals, grad)
                for k in range(K):
                    np.add.at(g_V[j,:,k], vals, grad * V_x[:, j, k])
            g_W /= b
            g_V /= b
            g_W += L2 * W
            g_V += L2 * V

            t += 1
            # Adam updates
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_W = beta1 * m_W + (1 - beta1) * g_W
            v_W = beta2 * v_W + (1 - beta2) * g_W * g_W
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V

            m_bias_hat = m_bias / (1 - beta1**t)
            v_bias_hat = v_bias / (1 - beta2**t)
            m_W_hat = m_W / (1 - beta1**t)
            v_W_hat = v_W / (1 - beta2**t)
            m_V_hat = m_V / (1 - beta1**t)
            v_V_hat = v_V / (1 - beta2**t)

            bias -= LR * m_bias_hat / (np.sqrt(v_bias_hat) + eps)
            W -= LR * m_W_hat / (np.sqrt(v_W_hat) + eps)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Write weights.bin
    # header: magic, d, K, C
    header = struct.pack("<4i", 0x464D4C31, d, K, C)
    out = header
    out += struct.pack("<f", bias)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (0 or 1). */
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
    if (C != 2 || d != 14 || K != 4) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4 + 4L*d*256 + 4L*d*256*K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = bias;
        float sum_V[4] = {0};
        float sum_V2[4] = {0};

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += W[(long)j * 256 + v];
            const float *Vj = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = Vj[kk];
                sum_V[kk] += val;
                sum_V2[kk] += val * val;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            inter += 0.5f * (sum_V[kk] * sum_V[kk] - sum_V2[kk]);
        }

        float logit = linear + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```