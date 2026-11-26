```hypothesis
Factorization Machine (FM) with 8 factors on the encoded features captures pairwise interactions
that a linear model misses. The adult dataset has known feature interactions (e.g., education × occupation).
With k=8 the model adds only ~140KB weights and O(d·k) inference cost, so it should beat label-linear
(0.8524) while staying faster than label-mlp16. If dev accuracy ≤0.8524 or latency ≥label-mlp16, the
hypothesis fails.
```

```python
"""Factorization Machine family for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Encodes columns as in label-linear, then learns w0, w[D], V[D, k] with Adam on logistic loss.
Exports per-byte lookup tables: magic 0x464D3231, d, k, C; then per column 256 entries of
(lin_contrib, fac_vec[k], fac_sq); finally bias w0."""
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

    # Build per-column encoding matrices (256 x E_j) and offsets
    enc_mats = []
    offsets = []
    off = 0
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)   # 256 x 1
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
        enc_mats.append(E)
        offsets.append(off)
        off += E.shape[1]
    D = off

    # Initialize parameters
    scale = 1.0 / math.sqrt(D)
    w0 = np.array(0.0, np.float32)
    w = (rng.uniform(-1, 1, D) * scale).astype(np.float32)
    V = (rng.uniform(-1, 1, (D, K)) * scale).astype(np.float32)
    params = [w0[None], w, V.ravel()]  # flat for Adam
    shapes = [(1,), (D,), (D, K)]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute encoded design matrix Z (n x D) for training
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc_mats)], axis=1).astype(np.float32)
    n = Z.shape[0]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # FM forward
            lin = z @ w + w0
            f = z @ V                      # batch x K
            f_sq = (z * z) @ (V * V)       # batch x K, but z is 0/1 for cat, so z*z = z
            inter = 0.5 * np.sum(f * f - f_sq, axis=1)
            logit = lin + inter
            p = sigmoid(logit)
            grad_logit = (p - yy).astype(np.float32) / len(yy)   # dL/dlogit

            # Gradients
            grad_w0 = grad_logit.sum()
            grad_w = z.T @ grad_logit + L2 * w
            # dL/dV = z.T @ (grad_logit[:,None] * (f - V[z_idx]?))
            # Vectorized: for each sample, dL/dV = grad_logit_i * (f_i * 1_K - V * z_i[:,:,None])
            # Since z is sparse-ish but we have dense, use: dL/dV = z.T @ (grad_logit[:,None] * f) - (z*z).T @ (grad_logit[:,None] * V) + L2*V
            g_f = grad_logit[:, None] * f
            g_V = z.T @ g_f
            g_V_sq = (z * z).T @ (grad_logit[:, None] * V)
            grad_V = (g_V - g_V_sq + L2 * V).ravel()

            grads = [np.array([grad_w0], np.float32), grad_w, grad_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Unpack trained params
    w0 = params[0][0]
    w = params[1]
    V = params[2].reshape(D, K)

    # Precompute per-column lookup tables
    lin_tables = []
    fac_tables = []
    sq_tables = []
    for j, E in enumerate(enc_mats):
        o = offsets[j]
        E_j = E.shape[1]
        w_j = w[o:o+E_j]
        V_j = V[o:o+E_j]
        lin_tables.append(E @ w_j)                     # 256
        fac_tables.append(E @ V_j)                     # 256 x K
        sq_tables.append(np.sum((E @ V_j) ** 2, axis=1))  # 256

    # Write weights.bin
    magic = 0x464D3231
    d = card["input"]["bytes"]
    out = struct.pack("<4i", magic, d, K, C)
    for j in range(d):
        for b in range(256):
            out += struct.pack("<f", lin_tables[j][b])
            out += struct.pack(f"<{K}f", *fac_tables[j][b])
            out += struct.pack("<f", sq_tables[j][b])
    out += struct.pack("<f", w0)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output one byte class (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }

    size_t per_col = 256 * (1 + K + 1) * sizeof(float);
    size_t expected = 16 + (size_t)d * per_col + sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    float *bias = (float *)(wb + nw - sizeof(float));
    float *tables = (float *)(hd + 4);   // [d][256][1 + K + 1]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *fac_sum = calloc(K, sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = *bias;
        memset(fac_sum, 0, K * sizeof(float));
        float fac_sq_sum = 0.0f;

        for (int j = 0; j < d; j++) {
            float *tbl = tables + (size_t)j * 256 * (K + 2) + xi[j] * (K + 2);
            lin_sum += tbl[0];
            for (int f = 0; f < K; f++) fac_sum[f] += tbl[1 + f];
            fac_sq_sum += tbl[1 + K];
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += fac_sum[f] * fac_sum[f];
        inter = 0.5f * (inter - fac_sq_sum);

        float score = lin_sum + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```