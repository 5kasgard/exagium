```hypothesis
Family: label-fm (Factorization Machine with binary output). 
Why: Adult tabular data has strong pairwise interactions (e.g., education×occupation, marital-status×relationship) that a linear model misses. A low-rank FM (fm_dim=4) captures these with O(d·256·fm_dim) parameters, adding only ~4 vector adds per feature at inference. 
Failure signal: dev accuracy ≤ label-linear (0.8524) or instructions/item > 2× label-linear.
```

```python
"""label-fm: Factorization Machine for binary tabular classification.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x464D424C, d, fm_dim, C(=2); float32 linear[d][256], factor[d][256][fm_dim], bias."""
import json, math, struct, sys

import numpy as np

FM_DIM = 4
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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    rng = np.random.default_rng(SEED)

    # Build encoding matrices per feature (same as label-linear)
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])          # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)                                               # (256, n_cat)

    # Flattened feature dimension
    k = sum(e.shape[1] for e in enc)
    # Offsets for each feature in the flattened space
    off = [0]
    for e in enc:
        off.append(off[-1] + e.shape[1])

    # Parameters: linear (k), factor (k, FM_DIM), bias (1)
    lin = np.zeros(k, np.float32)
    fac = (rng.uniform(-1, 1, (k, FM_DIM)) * math.sqrt(6.0 / k)).astype(np.float32)
    bias = np.float32(0.0)
    params = [lin, fac, bias[None]]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute per-feature per-value contributions for fast batch forward
    # lin_table[d, 256], fac_table[d, 256, FM_DIM]
    lin_table = np.zeros((d, 256), np.float32)
    fac_table = np.zeros((d, 256, FM_DIM), np.float32)
    for j in range(d):
        E = enc[j]                                  # (256, k_j)
        lin_table[j] = (E @ lin[off[j]:off[j+1]]).astype(np.float32)
        fac_table[j] = (E @ fac[off[j]:off[j+1]]).astype(np.float32)

    t = 0
    n = X.shape[0]
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                    # (B, d)
            yb = y[idx]                    # (B,)

            # Forward
            B = xb.shape[0]
            lin_sum = np.full(B, bias, np.float32)
            sum_v = np.zeros((B, FM_DIM), np.float32)
            sum_v2 = np.zeros((B, FM_DIM), np.float32)
            for j in range(d):
                v = xb[:, j]
                lin_sum += lin_table[j, v]
                fv = fac_table[j, v]                   # (B, FM_DIM)
                sum_v += fv
                sum_v2 += fv * fv
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
            logit = lin_sum + inter
            p = 1.0 / (1.0 + np.exp(-logit))           # sigmoid
            grad = (p - yb).astype(np.float32)         # dL/dlogit

            # Gradients w.r.t. tables
            g_lin_table = np.zeros_like(lin_table)
            g_fac_table = np.zeros_like(fac_table)
            for j in range(d):
                v = xb[:, j]
                g_lin_table[j, v] += grad
                # d inter / d fv = grad[:,None] * (sum_v - fv)
                g_fac_table[j, v] += grad[:, None] * (sum_v - fac_table[j, v])

            # Back-propagate to raw parameters (lin, fac)
            g_lin = np.zeros_like(lin)
            g_fac = np.zeros_like(fac)
            for j in range(d):
                E = enc[j]                                      # (256, k_j)
                g_lin[off[j]:off[j+1]] += (E.T @ g_lin_table[j]).astype(np.float32)
                g_fac[off[j]:off[j+1]] += (E.T @ g_fac_table[j]).astype(np.float32)

            g_bias = np.array([grad.sum()], np.float32)

            # L2 regularization
            g_lin += L2 * lin
            g_fac += L2 * fac

            grads = [g_lin, g_fac, g_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

            # Update tables after parameter step
            for j in range(d):
                lin_table[j] = (enc[j] @ lin[off[j]:off[j+1]]).astype(np.float32)
                fac_table[j] = (enc[j] @ fac[off[j]:off[j+1]]).astype(np.float32)
            bias = params[2][0]

    # Export weights
    out = struct.pack("<4i", 0x464D424C, d, FM_DIM, 2)
    out += lin_table.astype("<f4").tobytes()
    out += fac_table.astype("<f4").tobytes()
    out += np.array([bias], dtype="<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte (class 0/1). */
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
    if (nw < 16 || hd[0] != 0x464D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], fm_dim = hd[2], C = hd[3];
    if (C != 2 || fm_dim <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 + (long)d * 256 * fm_dim + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *lin_table = (const float *)(hd + 4);
    const float *fac_table = lin_table + (long)d * 256;
    const float bias = fac_table[(long)d * 256 * fm_dim][0];

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = calloc(fm_dim, sizeof(float));
    float *sum_v2 = calloc(fm_dim, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = bias;
        for (int fm = 0; fm < fm_dim; fm++) { sum_v[fm] = 0.0f; sum_v2[fm] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin_sum += lin_table[(long)j * 256 + v];
            const float *fv = fac_table + ((long)j * 256 + v) * fm_dim;
            for (int fm = 0; fm < fm_dim; fm++) {
                float f = fv[fm];
                sum_v[fm] += f;
                sum_v2[fm] += f * f;
            }
        }
        float inter = 0.0f;
        for (int fm = 0; fm < fm_dim; fm++) inter += sum_v[fm] * sum_v[fm] - sum_v2[fm];
        inter *= 0.5f;
        float logit = lin_sum + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```