```hypothesis
Family: label-fm (Factorization Machine). 
Why: Adult tabular data has strong categorical-ordinal interactions (e.g., education × capital-gain). 
A rank-8 FM adds pairwise interactions with O(d·f) inference cost, cheaper than an MLP hidden layer. 
If dev accuracy ≤ label-linear (0.8524) or inference cost > 2× label-linear, the model does not justify its complexity.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
F = 8  # factor size

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

    # Build encoders per column
    enc = []
    feat_sizes = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float64)
            enc.append(E)
            feat_sizes.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            feat_sizes.append(len(vals))

    # Concatenate to get full feature matrix Z (n, k)
    k = sum(feat_sizes)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # Initialize parameters
    w0 = np.array([0.0], dtype=np.float32)
    w = np.zeros(k, dtype=np.float32)
    V = (rng.normal(0, 0.01, (k, F))).astype(np.float32)

    # Adam state
    params = [w0, w, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Training loop
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]
            bsz = len(yy)

            # Forward
            linear = z @ w + w0[0]  # (bsz,)
            factor_sum = z @ V      # (bsz, F)
            factor_sq_sum = z @ (V * V)  # (bsz, F)
            interaction = 0.5 * np.sum(factor_sum * factor_sum - factor_sq_sum, axis=1)
            logit = linear + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            diff = p - yy  # (bsz,)

            # Gradients
            grad_w0 = np.array([diff.mean()], dtype=np.float32)
            grad_w = (z.T @ diff) / bsz + L2 * w
            term1 = (z.T @ (diff[:, None] * factor_sum)) / bsz
            term2 = V * ((z.T @ diff)[:, None] / bsz)
            grad_V = term1 - term2 + L2 * V

            grads = [grad_w0, grad_w, grad_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Precompute per-column lookup tables
    linear_contrib = np.zeros((d, 256), dtype=np.float32)
    factor_contrib = np.zeros((d, 256, F), dtype=np.float32)
    factor_sq_contrib = np.zeros((d, 256, F), dtype=np.float32)

    offset = 0
    for j in range(d):
        E = enc[j]  # (256, feat_j)
        fj = feat_sizes[j]
        w_j = w[offset:offset + fj]
        V_j = V[offset:offset + fj]
        V_j_sq = V_j * V_j
        linear_contrib[j] = (E @ w_j).astype(np.float32)
        factor_contrib[j] = (E @ V_j).astype(np.float32)
        factor_sq_contrib[j] = (E @ V_j_sq).astype(np.float32)
        offset += fj

    # Write weights.bin
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, F, C)
    out = header
    out += linear_contrib.astype("<f4").tobytes()
    out += factor_contrib.astype("<f4").tobytes()
    out += factor_sq_contrib.astype("<f4").tobytes()
    out += w0.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (0 or 1). */
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
    int d = hd[1], F = hd[2], C = hd[3];
    if (C != 2 || F <= 0 || d <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16 + 4L * d * 256 * (1 + 2 * F) + 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear_contrib = (const float *)(hd + 4);
    const float *factor_contrib = linear_contrib + (long)d * 256;
    const float *factor_sq_contrib = factor_contrib + (long)d * 256 * F;
    const float *w0 = factor_sq_contrib + (long)d * 256 * F;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = w0[0];
        float factor_sum[F];
        float factor_sq_sum[F];
        for (int f = 0; f < F; f++) {
            factor_sum[f] = 0.0f;
            factor_sq_sum[f] = 0.0f;
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lc = linear_contrib + (long)j * 256 + v;
            lin_sum += *lc;
            const float *fc = factor_contrib + ((long)j * 256 + v) * F;
            const float *fsc = factor_sq_contrib + ((long)j * 256 + v) * F;
            for (int f = 0; f < F; f++) {
                factor_sum[f] += fc[f];
                factor_sq_sum[f] += fsc[f];
            }
        }

        float interaction = 0.0f;
        for (int f = 0; f < F; f++) {
            interaction += 0.5f * (factor_sum[f] * factor_sum[f] - factor_sq_sum[f]);
        }
        float logit = lin_sum + interaction;
        float prob = 1.0f / (1.0f + expf(-logit));
        out_byte = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```