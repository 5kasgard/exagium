```hypothesis
Family: label-fm (Factorization Machine). 
Why: Adult has mixed ordinal/categorical features; pairwise interactions capture correlations (e.g., education×occupation) that linear models miss. FMs factorize interactions via low-rank embeddings, adding only O(d·k) parameters and O(d·k) inference cost (d=14, k=8 → ~112 ops). 
Failure signal: If dev accuracy ≤ label-linear (0.8524) or inference cost > 2× label-linear, the interaction benefit does not justify complexity.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: int32 magic 0x464D4C31, d, k, C; float32 bias, linear[d*256], emb[d*256*k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
SEED = 1
K = 8                # embedding dimension
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 50
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)   # 0 or 1
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2, "FM binary only"
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            # linear part: sum_j linear[j, xb[:,j]]
            lin_vals = linear[np.arange(d), xb.T]   # (d, B)
            lin_sum = lin_vals.sum(axis=0)          # (B,)

            # embedding part: sum_j emb[j, xb[:,j], :]
            emb_vals = emb[np.arange(d), xb.T]      # (d, B, K)
            sum_vec = emb_vals.sum(axis=0)          # (B, K)
            sum_sq_vec = (emb_vals**2).sum(axis=0)  # (B, K)
            inter = 0.5 * np.sum(sum_vec**2 - sum_sq_vec, axis=1)   # (B,)

            logit = bias + lin_sum + inter
            sig = 1.0 / (1.0 + np.exp(-logit))
            # Binary cross-entropy gradient wrt logit
            dlogit = (sig - yb).astype(np.float32)    # (B,)

            # Gradients
            dbias = dlogit.mean()
            # linear gradients: for each feature j, add dlogit to linear[j, value]
            dlin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(dlin[j], xb[:, j], dlogit)
            dlin /= B
            dlin += L2 * linear

            # embedding gradients: for each feature j, grad = dlogit[:,None] * (sum_vec - emb_val)
            demb = np.zeros_like(emb)
            for j in range(d):
                vals = xb[:, j]
                ev = emb[j, vals]               # (B, K)
                grad = dlogit[:, None] * (sum_vec - ev)   # (B, K)
                np.add.at(demb[j], vals, grad)
            demb /= B
            demb += L2 * emb

            # Adam updates
            t += 1
            # bias
            m_bias = BETA1 * m_bias + (1 - BETA1) * dbias
            v_bias = BETA2 * v_bias + (1 - BETA2) * (dbias * dbias)
            m_hat = m_bias / (1 - BETA1**t)
            v_hat = v_bias / (1 - BETA2**t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # linear
            m_lin = BETA1 * m_lin + (1 - BETA1) * dlin
            v_lin = BETA2 * v_lin + (1 - BETA2) * (dlin * dlin)
            m_hat = m_lin / (1 - BETA1**t)
            v_hat = v_lin / (1 - BETA2**t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + EPS)

            # emb
            m_emb = BETA1 * m_emb + (1 - BETA1) * demb
            v_emb = BETA2 * v_emb + (1 - BETA2) * (demb * demb)
            m_hat = m_emb / (1 - BETA1**t)
            v_hat = v_emb / (1 - BETA2**t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(np.array([bias], dtype=np.float32).tobytes())
        f.write(linear.astype("<f4").tobytes())
        f.write(emb.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1).
 * Weights format: int32 magic, d, k, C; float32 bias, linear[d*256], emb[d*256*k]. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t*)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4 + 4L*d*256 + 4L*d*256*k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float*)(hd + 4);
    const float *linear = bias + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_vec = calloc(k, sizeof(float));
    float *sum_sq_vec = calloc(k, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float logit = bias[0];
        memset(sum_vec, 0, k*sizeof(float));
        memset(sum_sq_vec, 0, k*sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * k;
            for (int u = 0; u < k; u++) {
                float val = e[u];
                sum_vec[u] += val;
                sum_sq_vec[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < k; u++) {
            float sv = sum_vec[u];
            inter += sv * sv - sum_sq_vec[u];
        }
        logit += 0.5f * inter;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char cls = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    free(sum_vec); free(sum_sq_vec);
    return 0;
}
```