```hypothesis
Family: label-fm (Factorization Machine). The Adult dataset mixes categorical and ordinal features; FM captures pairwise interactions via low-rank latent vectors, improving over linear models without the cost of a full MLP. The one-hot encoding yields sparse binary features, ideal for FM. If dev accuracy ≤ 0.8524 (label-linear) or inference cost per item exceeds label-linear significantly, the model does not meet the goal.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x464D4C31, d, k, C, float32 w0, then d*256*(1+k) float32 parameters
(per feature j, per value v: w_jv, V_jv[0..k-1])."""
import json, math, struct, sys

import numpy as np

K = 8                 # latent dimension
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    assert C == 2

    rng = np.random.default_rng(SEED)

    # Parameters: w0, w[d,256], V[d,256,K]
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = Xb.shape[0]

            # Forward
            linear = w0 + np.sum(w[np.arange(d), Xb.T], axis=0)   # (b,)
            s_vec = np.sum(V[np.arange(d), Xb.T], axis=0)         # (b, K)
            sq_vec = np.sum(V[np.arange(d), Xb.T] ** 2, axis=0)   # (b, K)
            inter = 0.5 * np.sum(s_vec ** 2 - sq_vec, axis=1)     # (b,)
            logit = linear + inter
            p = 1.0 / (1.0 + np.exp(-logit))                      # (b,)
            grad_logit = (p - yb).astype(np.float32)              # (b,)

            # Gradients
            gw0 = np.sum(grad_logit)
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)

            # Accumulate gradients per feature
            for j in range(d):
                vals = Xb[:, j]
                # linear weight gradient
                np.add.at(gw[j], vals, grad_logit)
                # V gradient: for each example, grad_logit * (s_vec - V[j, val])
                # We loop over latent dim for vectorized add.at
                for u in range(K):
                    # s_vec[:,u] - V[j, vals, u]
                    diff = s_vec[:, u] - V[j, vals, u]
                    np.add.at(gV[j, :, u], vals, grad_logit * diff)

            # L2 regularization
            gw += L2 * w
            gV += L2 * V

            # Adam update
            t += 1
            # w0
            m_w0 = 0.9 * m_w0 + 0.1 * gw0
            v_w0 = 0.999 * v_w0 + 0.001 * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - 0.9 ** t)) / (math.sqrt(v_w0 / (1 - 0.999 ** t)) + 1e-8)
            # w
            m_w = 0.9 * m_w + 0.1 * gw
            v_w = 0.999 * v_w + 0.001 * gw * gw
            w -= LR * (m_w / (1 - 0.9 ** t)) / (np.sqrt(v_w / (1 - 0.999 ** t)) + 1e-8)
            # V
            m_V = 0.9 * m_V + 0.1 * gV
            v_V = 0.999 * v_V + 0.001 * gV * gV
            V -= LR * (m_V / (1 - 0.9 ** t)) / (np.sqrt(v_V / (1 - 0.999 ** t)) + 1e-8)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    params = bytearray()
    params += struct.pack("<f", w0)
    # Flatten w and V in order: feature 0 value 0..255, feature 1 value 0..255, ...
    for j in range(d):
        for v in range(256):
            params += struct.pack("<f", w[j, v])
            for u in range(K):
                params += struct.pack("<f", V[j, v, u])
    open(sys.argv[4], "wb").write(header + params)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte: class index (0 or 1). */
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
    if (C != 2 || K < 1 || d < 1 || nw != 16 + 4 + 4L * d * 256 * (1 + K)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *params = (const float *)(hd + 4);  // after header (4 int32)
    float w0 = params[0];
    const float *w = params + 1;
    const float *V = params + 1 + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *s = malloc(sizeof(float) * K);
    float *sq = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = w0;
        for (int u = 0; u < K; u++) { s[u] = 0.0f; sq[u] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            long base = ((long)j * 256 + v) * (1 + K);
            logit += w[base];
            const float *Vjv = V + (long)j * 256 * K + (long)v * K;
            for (int u = 0; u < K; u++) {
                float val = Vjv[u];
                s[u] += val;
                sq[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += s[u] * s[u] - sq[u];
        logit += 0.5f * inter;
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    free(s); free(sq);
    return 0;
}
```