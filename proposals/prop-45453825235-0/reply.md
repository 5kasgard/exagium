```hypothesis
Family: label-fm (Factorization Machine with 2-way interactions).
Why: Adult has mixed categorical/ordinal features; FM learns pairwise interactions via low-rank embeddings, capturing non-linearities missed by linear models while staying lightweight (d=14, k=8 → ~130KB weights, ~1K FLOPs/item).
Failure: If dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-mlp16 (2325 Ir/item), the model does not justify its complexity.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (fixed, no tuning on tuning split)
K = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 20
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
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = xb.shape[0]

            # Gather linear and embeddings
            lin_vals = linear[np.arange(d), xb.T].T   # (b, d)
            emb_vals = emb[np.arange(d), xb.T].transpose(1,0,2)  # (b, d, K)

            # FM forward using sum-emb trick
            sum_emb = emb_vals.sum(axis=1)                 # (b, K)
            sum_emb_sq = (emb_vals**2).sum(axis=1)         # (b, K)
            inter = 0.5 * ((sum_emb**2 - sum_emb_sq).sum(axis=1))  # (b,)
            lin_sum = lin_vals.sum(axis=1)                 # (b,)
            logit = bias + lin_sum + inter                 # (b,)

            # Sigmoid and gradient
            pred = 1.0 / (1.0 + np.exp(-logit))
            g = (pred - yb).astype(np.float32)             # (b,)

            # Gradients
            g_bias = g.sum()
            g_lin = np.zeros_like(linear)
            for i in range(d):
                np.add.at(g_lin[i], xb[:, i], g)
            g_emb = np.zeros_like(emb)
            for i in range(d):
                for v in range(256):
                    mask = (xb[:, i] == v)
                    if not mask.any():
                        continue
                    g_emb[i, v] = (g[mask, None] * (sum_emb[mask] - emb_vals[mask, i])).sum(axis=0)

            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * emb

            # Adam update
            t += 1
            # bias
            m_bias = BETA1 * m_bias + (1 - BETA1) * g_bias
            v_bias = BETA2 * v_bias + (1 - BETA2) * g_bias * g_bias
            m_hat = m_bias / (1 - BETA1**t)
            v_hat = v_bias / (1 - BETA2**t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # linear
            m_lin = BETA1 * m_lin + (1 - BETA1) * g_lin
            v_lin = BETA2 * v_lin + (1 - BETA2) * g_lin * g_lin
            m_hat_lin = m_lin / (1 - BETA1**t)
            v_hat_lin = v_lin / (1 - BETA2**t)
            linear -= LR * m_hat_lin / (np.sqrt(v_hat_lin) + EPS)

            # emb
            m_emb = BETA1 * m_emb + (1 - BETA1) * g_emb
            v_emb = BETA2 * v_emb + (1 - BETA2) * g_emb * g_emb
            m_hat_emb = m_emb / (1 - BETA1**t)
            v_hat_emb = v_emb / (1 - BETA2**t)
            emb -= LR * m_hat_emb / (np.sqrt(v_hat_emb) + EPS)

    # Pack weights
    magic = 0x464D3130
    header = struct.pack("<4i", magic, d, K, C)
    out_bytes = header
    out_bytes += struct.pack("<f", bias)
    out_bytes += linear.astype("<f4").tobytes()
    out_bytes += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out_bytes)

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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3130) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K != 8 || C != 2) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = *bias;
        float sum_emb[8] = {0};
        float sum_emb_sq[8] = {0};

        for (int j = 0; j < d; j++) {
            int val = xi[j];
            score += linear[(long)j * 256 + val];
            const float *e = emb + ((long)j * 256 + val) * K;
            for (int f = 0; f < K; f++) {
                float v = e[f];
                sum_emb[f] += v;
                sum_emb_sq[f] += v * v;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            inter += sum_emb[f] * sum_emb[f] - sum_emb_sq[f];
        }
        score += 0.5f * inter;

        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```