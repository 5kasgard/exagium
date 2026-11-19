```hypothesis
Family: label-fm (Factorization Machine with rank-4 pairwise interactions).
Why: Adult has strong categorical feature interactions (e.g., education × occupation). A factorization machine captures all pairwise interactions with O(k·d) parameters and compute, avoiding the O(d^2) blowup of explicit crosses. Rank 4 keeps weights ~130 KB and inference ~14·4·3 ≈ 170 FLOPs/item, cheaper than label-mlp16 (2325 Ir/item) while adding nonlinearity missing from label-linear.
Failure sign: dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 2000.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encodes ordinal columns as z-scored scalars, categorical as one-hot.
Learns linear weights + rank-4 latent vectors per feature.
Exports per-byte lookup tables for fast inference: linear, v*x, v^2*x^2."""
import json, math, struct, sys

import numpy as np

K = 4
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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    y = (y * 2 - 1).astype(np.float32)          # -1 / +1
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    n = X.shape[0]

    # Build feature mapping: for each column j and byte value v -> (feature_index, feature_value)
    feat_idx = np.full((d, 256), -1, dtype=np.int32)
    feat_val = np.zeros((d, 256), dtype=np.float32)
    n_features = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            vals = X[:, j].astype(np.float64)
            mu, sd = vals.mean(), vals.std() or 1.0
            scaled = (np.arange(256, dtype=np.float64) - mu) / sd
            feat_idx[j, :] = n_features
            feat_val[j, :] = scaled.astype(np.float32)
            n_features += 1
        else:  # categorical
            seen = np.unique(X[:, j])
            seen = seen[seen != 255]            # 255 = unseen marker
            for v in seen:
                feat_idx[j, v] = n_features
                feat_val[j, v] = 1.0
                n_features += 1
            # unseen (255) and other values stay -1/0

    # Parameters
    rng = np.random.default_rng(SEED)
    lin = np.zeros(n_features, np.float32)
    lat = (rng.uniform(-1, 1, (n_features, K)) * math.sqrt(6.0 / n_features)).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_lat, v_lat = np.zeros_like(lat), np.zeros_like(lat)
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                          # (B, d)
            yb = y[idx]                          # (B,)

            # Gather feature indices and values for the batch
            # feat_idx_batch: (B, d), feat_val_batch: (B, d)
            feat_idx_batch = feat_idx[np.arange(d), xb.T].T
            feat_val_batch = feat_val[np.arange(d), xb.T].T

            # Mask for valid features (categorical unseen = -1)
            valid = feat_idx_batch >= 0

            # Linear part
            lin_part = np.sum(np.where(valid, lin[feat_idx_batch] * feat_val_batch, 0.0), axis=1) + bias

            # Interaction part: 0.5 * sum_f (s_f^2 - sq_f)
            # latent_gathered: (B, d, K)
            lat_gathered = lat[feat_idx_batch]                    # invalid rows get lat[0] but masked out
            z = np.where(valid[:, :, None], lat_gathered * feat_val_batch[:, :, None], 0.0)
            s_f = np.sum(z, axis=1)                               # (B, K)
            sq_f = np.sum(z * z, axis=1)                          # (B, K)
            inter = 0.5 * np.sum(s_f * s_f - sq_f, axis=1)        # (B,)

            logit = lin_part + inter
            # Logistic loss gradient w.r.t logit
            z = -yb * logit
            # stable sigmoid: 1/(1+exp(-z)) for z>=0 else exp(z)/(1+exp(z))
            sig = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
            d_logit = -yb * sig                                   # (B,)

            # Gradient for linear weights
            d_lin = d_logit[:, None] * feat_val_batch             # (B, d)
            # Gradient for latent vectors: d_logit * (s_f - z) * val
            d_lat = d_logit[:, None, None] * (s_f[:, None, :] - z) * feat_val_batch[:, :, None]  # (B, d, K)

            # Accumulate gradients per feature
            # We'll use np.add.at for unsorted segment sum
            flat_idx = feat_idx_batch.ravel()
            flat_dlin = d_lin.ravel()
            flat_dlat = d_lat.reshape(-1, K)
            valid_flat = valid.ravel()

            g_lin = np.zeros_like(lin)
            g_lat = np.zeros_like(lat)
            np.add.at(g_lin, flat_idx[valid_flat], flat_dlin[valid_flat])
            np.add.at(g_lat, flat_idx[valid_flat], flat_dlat[valid_flat])
            g_bias = np.sum(d_logit)

            # L2 regularization
            g_lin += L2 * lin
            g_lat += L2 * lat

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            lin -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # latent
            m_lat = beta1 * m_lat + (1 - beta1) * g_lat
            v_lat = beta2 * v_lat + (1 - beta2) * g_lat * g_lat
            m_hat = m_lat / (1 - beta1 ** t)
            v_hat = v_lat / (1 - beta2 ** t)
            lat -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (math.sqrt(v_hat) + eps)

    # Build inference lookup tables: for each column j and byte v -> linear, vx[K], vx2[K]
    lin_table = np.zeros((d, 256), np.float32)
    vx_table = np.zeros((d, 256, K), np.float32)
    vx2_table = np.zeros((d, 256, K), np.float32)
    for j in range(d):
        for v in range(256):
            fi = feat_idx[j, v]
            fv = feat_val[j, v]
            if fi >= 0:
                lin_table[j, v] = lin[fi] * fv
                lv = lat[fi]
                vx_table[j, v] = lv * fv
                vx2_table[j, v] = lv * lv * fv * fv
            # else leave zeros

    # Write weights.bin
    # Format: magic, d, K, C, bias, then lin_table (d*256), vx_table (d*256*K), vx2_table (d*256*K) all float32 LE
    magic = 0x464D5442  # 'FMTS'
    C = 2
    hdr = struct.pack("<5i", magic, d, K, C, 0)  # bias is float, but we pack as int placeholder; we'll overwrite
    # Actually pack bias as float after the 4 ints? Let's do: 4 ints, then float bias, then tables.
    # We'll write header as 4 ints + 1 float.
    hdr = struct.pack("<4if", magic, d, K, C, bias)
    body = lin_table.astype("<f4").tobytes() + vx_table.astype("<f4").tobytes() + vx2_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Factorization Machine with rank-K pairwise interactions.
 * Weights layout: int32 magic, d, K, C; float bias; float lin[d][256]; float vx[d][256][K]; float vx2[d][256][K].
 * Input record: d bytes. Output: 1 byte class (0/1). */
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
    int32_t *hdi = (int32_t *)wb;
    if (hdi[0] != 0x464D5442) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdi[1], K = hdi[2], C = hdi[3];
    float bias = *(float *)(hdi + 4);
    if (C != 2 || K < 1 || K > 16 || d < 1 || d > 256) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 20 + (size_t)d * 256 * (1 + 2 * K) * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *lin = (const float *)(wb + 20);
    const float *vx = lin + (size_t)d * 256;
    const float *vx2 = vx + (size_t)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *s_f = calloc(K, sizeof(float));
    float *sq_f = calloc(K, sizeof(float));
    unsigned char pred;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = bias;
        for (int f = 0; f < K; f++) { s_f[f] = 0.0f; sq_f[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            size_t base = (size_t)j * 256 + v;
            lin_sum += lin[base];
            for (int f = 0; f < K; f++) {
                float vxf = vx[base * K + f];
                s_f[f] += vxf;
                sq_f[f] += vx2[base * K + f];
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += 0.5f * (s_f[f] * s_f[f] - sq_f[f]);
        float score = lin_sum + inter;
        pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```