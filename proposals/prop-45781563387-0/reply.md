```hypothesis
Family: label-fm (Factorization Machine with logistic loss). Adult has strong pairwise interactions (education×occupation, marital×sex) that linear models miss. FM captures all pairwise interactions via factorized vectors of dimension k, adding only O(d·k) parameters. With k=8 it should beat label-linear's 0.8524 while keeping inference fast (one dot product per feature + k sums). If dev accuracy ≤ 0.853 the factor dimension is too small or regularization too strong.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-scored single feature; categorical -> one-hot over training values.
FM: w0 + sum_i w_i x_i + 0.5 * sum_f ( (sum_i V_{i,f} x_i)^2 - sum_i V_{i,f}^2 x_i^2 )
Trained with Adam (lr=0.1, L2=1e-4) on logistic loss, batch 64, epochs min(100, ceil(600000/n)).
Exports binary: magic 0x464D5F31, d, k, C=2; float32 w0, T_linear[d][256], T_factor[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.1, 1e-4, 64, 600000, 100


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
    types = card["input"]["types"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Build feature mapping
    feat_of_col = []          # list of (col_type, info)
    # for ordinal: (mu, sigma, feat_idx)
    # for categorical: dict value->feat_idx
    feat_idx = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            feat_of_col.append(("ord", (mu, sd, feat_idx)))
            feat_idx += 1
        else:
            vals = np.unique(X[:, j])
            mapping = {int(v): i for i, v in enumerate(vals)}
            feat_of_col.append(("cat", (mapping, feat_idx, len(vals))))
            feat_idx += len(vals)
    m = feat_idx

    # Build design matrix Z (n, m) as float32
    Z = np.zeros((n, m), dtype=np.float32)
    for j, (typ, info) in enumerate(feat_of_col):
        if typ == "ord":
            mu, sd, fi = info
            Z[:, fi] = (X[:, j] - mu) / sd
        else:
            mapping, base, _ = info
            for v, fi in mapping.items():
                Z[X[:, j] == v, base + fi] = 1.0

    # FM parameters
    w0 = np.float32(0.0)
    w = np.zeros(m, dtype=np.float32)
    V = (rng.normal(0, 1, (m, K)) * math.sqrt(1.0 / m)).astype(np.float32)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]                 # (B, m)
            yy = y[idx]                # (B,)
            B = len(yy)

            # FM forward
            linear = z @ w + w0                     # (B,)
            zV = z @ V                              # (B, K)
            interaction = 0.5 * np.sum(zV * zV - (z * z) @ (V * V), axis=1)  # (B,)
            logits = linear + interaction           # (B,)

            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))       # (B,)
            grad = (p - yy).astype(np.float32)      # (B,)

            # Gradients
            g_w0 = grad.mean()
            g_w = (z.T @ grad) / B + L2 * w
            # V gradient: for each f, sum_i grad_i * (z_i,f * zV_i - z_i^2 * V_i,f)
            # vectorized: (grad[:,None] * zV) @ z? Let's derive properly.
            # dL/dV_{j,f} = sum_i grad_i * (z_{i,j} * zV_{i,f} - z_{i,j}^2 * V_{j,f})
            # = sum_i grad_i * z_{i,j} * zV_{i,f} - sum_i grad_i * z_{i,j}^2 * V_{j,f}
            # First term: z.T @ (grad[:,None] * zV)  -> (m, K)
            # Second term: V * ( (z*z).T @ grad )[:,None]  -> (m, K)
            zV_grad = grad[:, None] * zV                 # (B, K)
            term1 = z.T @ zV_grad                        # (m, K)
            zz = z * z
            term2 = V * (zz.T @ grad)[:, None]           # (m, K)
            g_V = (term1 - term2) / B + L2 * V

            t += 1
            # Adam updates
            beta1, beta2 = 0.9, 0.999
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V

            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)

            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + 1e-8)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + 1e-8)

    # Expand to per-byte lookup tables
    T_lin = np.zeros((d, 256), dtype=np.float32)
    T_fac = np.zeros((d, 256, K), dtype=np.float32)

    for j, (typ, info) in enumerate(feat_of_col):
        if typ == "ord":
            mu, sd, fi = info
            for v in range(256):
                z = (v - mu) / sd
                T_lin[j, v] = w[fi] * z
                T_fac[j, v, :] = V[fi] * z
        else:
            mapping, base, _ = info
            for v in range(256):
                if v in mapping:
                    fi = base + mapping[v]
                    T_lin[j, v] = w[fi]
                    T_fac[j, v, :] = V[fi]

    # Write weights.bin
    magic = 0x464D5F31
    C = 2
    header = struct.pack("<4i", magic, d, K, C)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(struct.pack("<f", w0))
        f.write(T_lin.astype("<f4").tobytes())
        f.write(T_fac.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (0 or 1).
 * Weights format: magic 0x464D5F31, int32 d, K, C; float32 w0;
 * float32 T_linear[d][256]; float32 T_factor[d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *T_lin = p + 1;
    const float *T_fac = T_lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *fac_sum = (float *)calloc(K, sizeof(float));
    float *fac_sq_sum = (float *)calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = w0;
        for (int f = 0; f < K; f++) { fac_sum[f] = 0.0f; fac_sq_sum[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_row = T_lin + (long)j * 256;
            const float *fac_row = T_fac + ((long)j * 256 + v) * K;
            lin_sum += lin_row[v];
            for (int f = 0; f < K; f++) {
                float val = fac_row[f];
                fac_sum[f] += val;
                fac_sq_sum[f] += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float s = fac_sum[f];
            inter += s * s - fac_sq_sum[f];
        }
        float score = lin_sum + 0.5f * inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```