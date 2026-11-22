```hypothesis
Family: label-fm (factorization machine with latent dim 8). FMs capture pairwise feature interactions crucial for tabular data like Adult, while staying lightweight. With 14 columns and one-hot/ordinal encoding, a rank-8 factorization adds only ~8k parameters. Trained with Adam/logistic loss, it should exceed label-linear's 0.8524 dev accuracy. If dev accuracy ≤ 0.8524, interactions don't help or rank is too small.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-scored float; categorical -> one-hot over training values (255=unseen -> zero vec).
Model: bias + linear weights + factor vectors (dim K=8) per feature value.
Output: magic 0x464D3130, d, K, C, bias, then per-column 256*(1+K) float32 (linear + factors)."""
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
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Build encoders and feature index mapping
    encoders = []
    feat_starts = []
    offset = 0
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            # one float feature per ordinal column
            encoders.append(("ord", mu, sd, offset))
            offset += 1
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.int32)
            E[vals, np.arange(len(vals))] = 1
            encoders.append(("cat", E, offset))
            offset += len(vals)
    nfeat = offset

    # Convert dataset to dense feature matrix (n x nfeat)
    Z = np.zeros((X.shape[0], nfeat), np.float32)
    for j, enc in enumerate(encoders):
        if enc[0] == "ord":
            _, mu, sd, off = enc
            Z[:, off] = ((X[:, j].astype(np.float32) - mu) / sd)
        else:
            _, E, off = enc
            Z[:, off:off+E.shape[1]] = E[X[:, j]]

    # FM parameters
    bias = np.float32(0.0)
    w = (rng.uniform(-0.01, 0.01, nfeat)).astype(np.float32)
    V = (rng.uniform(-0.01, 0.01, (nfeat, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    n = Z.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z, yy = Z[idx], y[idx]
            # FM forward
            linear = z @ w + bias
            zV = z @ V                      # (B, K)
            z2_V2 = (z**2) @ (V**2)         # (B, K)
            inter = 0.5 * np.sum(zV**2 - z2_V2, axis=1)
            logits = linear + inter         # (B,)
            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = p - yy.astype(np.float32)   # (B,)
            # Gradients
            g_bias = grad.mean()
            g_w = (z.T @ grad) / len(idx) + L2 * w
            # V gradient: for each feature f, sum over batch: grad_i * (z_i[f] * zV_i - z_i[f]**2 * V[f])
            g_V = np.zeros_like(V)
            for f in range(nfeat):
                zf = z[:, f]
                if np.any(zf != 0):
                    g_V[f] = (grad * (zf[:, None] * zV - zf[:, None]**2 * V[f])).mean(axis=0) + L2 * V[f]
            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= LR * (m_bias / (1 - beta1**t)) / (np.sqrt(v_bias / (1 - beta2**t)) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= LR * (m_w / (1 - beta1**t)) / (np.sqrt(v_w / (1 - beta2**t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= LR * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)

    # Export weights as per-byte lookup tables
    # For each original column j (0..d-1), for each byte value v (0..255):
    #   linear contribution = sum over features from this column of w[f] * encoding[f][v]
    #   factor contribution = sum over features from this column of V[f] * encoding[f][v]
    out = bytearray()
    out += struct.pack("<4i", 0x464D3130, d, K, C)
    out += struct.pack("<f", bias)
    for j, enc in enumerate(encoders):
        lin_col = np.zeros(256, np.float32)
        fac_col = np.zeros((256, K), np.float32)
        if enc[0] == "ord":
            _, mu, sd, off = enc
            # ordinal: encoding is single float (v - mu)/sd
            for v in range(256):
                val = (v - mu) / sd
                lin_col[v] = w[off] * val
                fac_col[v] = V[off] * val
        else:
            _, E, off = enc
            for v in range(256):
                if v < E.shape[0]:
                    e = E[v]
                    if np.any(e):
                        lin_col[v] = np.dot(e, w[off:off+E.shape[1]])
                        fac_col[v] = e @ V[off:off+E.shape[1]]
        out += lin_col.astype("<f4").tobytes()
        out += fac_col.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1).
 * Weights format: magic 0x464D3130, int32 d, K, C, float bias,
 *   then for each column j=0..d-1: float linear[256], float factors[256][K].
 * FM score = bias + sum linear + 0.5 * sum_k (sum_factors[k]^2 - sum_factors_sq[k]).
 * Class 1 if score > 0 else 0. */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x464D3130) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], K = hdr[2], C = hdr[3];
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t expected = 20 + (size_t)d * 256 * (1 + K) * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }
    const float *p = (const float *)(hdr + 4);
    float bias = *p++;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_fac = malloc(K * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float lin = bias;
        for (int f = 0; f < K; f++) sum_fac[f] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *col_lin = p;
            const float *col_fac = p + 256;
            int v = xi[j];
            lin += col_lin[v];
            const float *fac_v = col_fac + (size_t)v * K;
            for (int f = 0; f < K; f++) sum_fac[f] += fac_v[f];
            p += 256 * (1 + K);
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float s = sum_fac[f];
            inter += s * s;
        }
        // subtract sum of squares of factors per feature
        // We need to recompute sum of squares: sum over j of (factor[j][v][f]^2)
        // But we didn't accumulate squares. We can either store squares in weights or recompute.
        // Since we only have summed factors, we cannot recover sum of squares without extra storage.
        // We'll change the weights format to also store squared factors per byte? That doubles factor storage.
        // Alternative: compute sum of squares on the fly by iterating columns again.
        // But we already advanced p. Let's restructure: first pass to compute linear and sum_fac, second pass for sum_sq.
        // We'll reset p and do two passes. Simpler: store both factor and factor_sq in weights.
        // However, we already wrote the weights without squares. We must change train.py to store squares.
        // But we cannot change train.py now. We need to compute squares from the factors we have.
        // We can't because we summed them. We need to keep the factors per column in memory.
        // Since d=14 and K=8, we can store the factor vectors for the current record.
        // Let's redo: allocate array [d][K] for factors of this record.
        // We'll revert p to start of columns and do a single pass collecting factors.
        // This is simpler and fast enough.
        // We'll change the loop.
    }
    // The above loop is incomplete; we need a correct implementation.
    // Let's rewrite the inference loop properly.
    // We'll go back to the start of column data.
    p = (const float *)(hdr + 4) + 1; // after bias
    pos = 0;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        float lin = bias;
        float fac_sum[8] = {0};
        float fac_sq_sum[8] = {0};
        const float *col_ptr = p;
        for (int j = 0; j < d; j++) {
            const float *col_lin = col_ptr;
            const float *col_fac = col_ptr + 256;
            int v = xi[j];
            lin += col_lin[v];
            const float *fac_v = col_fac + (size_t)v * K;
            for (int f = 0; f < K; f++) {
                float val = fac_v[f];
                fac_sum[f] += val;
                fac_sq_sum[f] += val * val;
            }
            col_ptr += 256 * (1 + K);
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float s = fac_sum[f];
            inter += s * s - fac_sq_sum[f];
        }
        inter *= 0.5f;
        float score = lin + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```