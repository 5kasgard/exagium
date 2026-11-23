```hypothesis
Family: label-fm (Factorization Machine with 8 latent factors). It captures pairwise feature interactions which are crucial for tabular data like Adult, while remaining a linear model in the embedded space. The one-hot encoding for categoricals and z-scoring for ordinals is kept from label-linear; we add a low-rank interaction term per class. If dev accuracy does not exceed label-linear's 0.8524 by at least 0.002, the family does not justify its extra parameters.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal -> z-scored scalar, categorical -> one-hot over training values.
Model: two-class FM with k=8 factors per class. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights layout: magic 0x464D3100, d, k, C; float32 w0[C], linear[d][256][C], fsum[d][256][C][k], fsq[d][256][C][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Build encoders and compute encoded dimension
    enc = []
    dims = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)  # (256,1)
            enc.append(E)
            dims.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            dims.append(len(vals))

    d_enc = sum(dims)
    offsets = np.cumsum([0] + dims)

    # Initialize parameters
    w0 = np.zeros(C, np.float32)
    W = (rng.normal(0, 0.01, (d_enc, C)) * math.sqrt(6.0 / d_enc)).astype(np.float32)
    V = (rng.normal(0, 0.01, (d_enc, K, C)) * math.sqrt(6.0 / d_enc)).astype(np.float32)

    params = [w0, W, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute encoded design matrix Z (n x d_enc)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]          # (b, d_enc)
            yy = y[idx]         # (b,)

            # Forward
            # Linear logits: (b, C)
            lin = w0 + z @ W
            # Factor logits: (b, C)
            fac = np.zeros((z.shape[0], C), np.float32)
            for c in range(C):
                for f in range(K):
                    v = V[:, f, c]                 # (d_enc,)
                    s_val = z @ v                  # (b,)
                    s_sq = z @ (v * v)             # (b,)
                    fac[:, c] += 0.5 * (s_val * s_val - s_sq)
            logits = lin + fac                      # (b, C)

            # Softmax and loss gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # (b, C)
            grad_logits = probs
            grad_logits[np.arange(len(yy)), yy] -= 1.0
            grad_logits /= len(yy)                # (b, C)

            # Gradients
            grad_w0 = grad_logits.sum(axis=0)                     # (C,)
            grad_W = z.T @ grad_logits + L2 * W                   # (d_enc, C)

            grad_V = np.zeros_like(V)                             # (d_enc, K, C)
            for c in range(C):
                g_c = grad_logits[:, c]                           # (b,)
                for f in range(K):
                    v = V[:, f, c]                                # (d_enc,)
                    s_val = z @ v                                 # (b,)
                    # d/dV = z.T @ (g_c[:,None] * s_val[:,None] * z) - v * (g_c @ (z**2)) + L2 * v
                    grad_V[:, f, c] = z.T @ (g_c[:, None] * s_val[:, None] * z) \
                                      - v * (g_c @ (z * z)) \
                                      + L2 * v

            grads = [grad_w0, grad_W, grad_V]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup tables for inference
    linear_tab = np.zeros((card["input"]["bytes"], 256, C), np.float32)
    fsum_tab   = np.zeros((card["input"]["bytes"], 256, C, K), np.float32)
    fsq_tab    = np.zeros((card["input"]["bytes"], 256, C, K), np.float32)

    for j in range(card["input"]["bytes"]):
        off, dim = offsets[j], dims[j]
        W_col = W[off:off+dim, :]           # (dim, C)
        V_col = V[off:off+dim, :, :]        # (dim, K, C)
        E = enc[j]                          # (256, dim)
        for v in range(256):
            z = E[v]                        # (dim,)
            linear_tab[j, v] = z @ W_col
            for c in range(C):
                for f in range(K):
                    v_vec = V_col[:, f, c]  # (dim,)
                    fsum_tab[j, v, c, f] = z @ v_vec
                    fsq_tab[j, v, c, f] = (z * z) @ (v_vec * v_vec)

    # Write weights
    magic = 0x464D3100
    out = struct.pack("<4i", magic, card["input"]["bytes"], K, C)
    out += w0.astype("<f4").tobytes()
    out += linear_tab.astype("<f4").tobytes()
    out += fsum_tab.astype("<f4").tobytes()
    out += fsq_tab.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad params\n"); return 3; }

    long expected = 16L
                  + 4L * C
                  + 4L * d * 256 * C
                  + 4L * d * 256 * C * K * 2;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *linear = w0 + C;
    const float *fsum = linear + (long)d * 256 * C;
    const float *fsq  = fsum + (long)d * 256 * C * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logit = malloc(sizeof(float) * C);
    float *sf = malloc(sizeof(float) * C * K);
    float *ssq = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            logit[c] = w0[c];
            for (int f = 0; f < K; f++) {
                sf[c * K + f] = 0.0f;
                ssq[c * K + f] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_j = linear + ((long)j * 256 + v) * C;
            const float *fs_j = fsum + ((long)j * 256 + v) * C * K;
            const float *fsq_j = fsq + ((long)j * 256 + v) * C * K;

            for (int c = 0; c < C; c++) {
                logit[c] += lin_j[c];
                for (int f = 0; f < K; f++) {
                    int idx = c * K + f;
                    sf[idx] += fs_j[idx];
                    ssq[idx] += fsq_j[idx];
                }
            }
        }

        for (int c = 0; c < C; c++) {
            for (int f = 0; f < K; f++) {
                float s = sf[c * K + f];
                logit[c] += 0.5f * (s * s - ssq[c * K + f]);
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logit[c] > logit[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```