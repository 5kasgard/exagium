```hypothesis
Propose "label-fm": a Factorization Machine with embedding dim k=8. FMs capture pairwise feature interactions crucial for tabular data (e.g., education × occupation) while keeping inference O(d·k). The Adult dataset has 6 ordinal + 8 categorical columns; label-linear ignores interactions. We precompute per-byte contributions for fast inference. If dev accuracy ≤ 0.855 (not beating label-linear's 0.8524 by a clear margin), the hypothesis fails.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-scored continuous, categorical as one-hot over training values (255=unseen).
Trains FM with binary cross-entropy + L2 via Adam (lr=1e-2, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Exports: magic 0x464D3231, d, k, C=2, w0, then for each column j and byte v: linear_contrib, embed_vec[k], embed_sq_sum.
"""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    n, d = X_bytes.shape
    types = card["input"]["types"]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)

    # Build feature mapping per column
    col_feat_start = []
    col_feat_dim = []
    col_is_ordinal = []
    ordinal_mu = []
    ordinal_sd = []
    cat_vals = []  # list of unique values per cat column (excluding 255)
    total_feats = 0

    for j, t in enumerate(types):
        col_feat_start.append(total_feats)
        if t == "ordinal":
            col_is_ordinal.append(True)
            v = X_bytes[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ordinal_mu.append(mu)
            ordinal_sd.append(sd)
            col_feat_dim.append(1)
            total_feats += 1
            cat_vals.append(None)
        else:
            col_is_ordinal.append(False)
            ordinal_mu.append(0.0)
            ordinal_sd.append(1.0)
            vals = np.unique(X_bytes[:, j])
            # remove 255 if present (unseen marker), we'll add it as extra feature
            vals = vals[vals != 255]
            cat_vals.append(vals)
            dim = len(vals) + 1  # +1 for unseen (255)
            col_feat_dim.append(dim)
            total_feats += dim

    # Construct dense feature matrix (n, total_feats) float32
    X = np.zeros((n, total_feats), dtype=np.float32)
    for j in range(d):
        start = col_feat_start[j]
        dim = col_feat_dim[j]
        if col_is_ordinal[j]:
            v = (X_bytes[:, j].astype(np.float32) - ordinal_mu[j]) / ordinal_sd[j]
            X[:, start] = v
        else:
            vals = cat_vals[j]
            # map each byte to feature index
            idx = np.full(256, dim - 1, dtype=np.int32)  # default to unseen (last)
            idx[vals] = np.arange(len(vals))
            feat_idx = idx[X_bytes[:, j]]
            X[np.arange(n), start + feat_idx] = 1.0

    # Initialize FM parameters
    w0 = np.float32(0.0)
    w = np.zeros(total_feats, dtype=np.float32)
    V = (rng.normal(0, 0.01, (total_feats, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, f)
            yb = y[idx]          # (b,)
            bsz = len(idx)

            # Forward
            linear = w0 + xb @ w                          # (b,)
            S = xb @ V                                    # (b, k)
            sq_sum = (xb * xb) @ (V * V)                  # (b,)  sum_i x_i^2 * ||V_i||^2
            interaction = 0.5 * (np.sum(S * S, axis=1) - sq_sum)
            logit = linear + interaction
            prob = 1.0 / (1.0 + np.exp(-logit))
            # Binary cross-entropy gradient wrt logit
            dlogit = prob - yb.astype(np.float32)         # (b,)

            # Gradients
            dw0 = np.mean(dlogit)
            dw = (xb.T @ dlogit) / bsz + L2 * w
            # dV = xb.T @ (dlogit[:,None] * S) - V * ((xb^2).T @ dlogit)[:,None]
            dV = (xb.T @ (dlogit[:, None] * S)) / bsz - V * ((xb * xb).T @ dlogit)[:, None] / bsz
            dV += L2 * V

            # Adam updates
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * dw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (dw0 * dw0)
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * (dw * dw)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * (dV * dV)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Precompute per-column per-byte tables
    # For each column j, for each byte v in 0..255:
    #   linear_contrib[j][v] = sum_{feat in col j} w_i * x_i(v)
    #   embed_vec[j][v] = sum_{feat in col j} V_i * x_i(v)   (vector K)
    #   embed_sq_sum[j][v] = sum_{feat in col j} (V_i^2) * x_i(v)^2
    linear_table = np.zeros((d, 256), dtype=np.float32)
    embed_table = np.zeros((d, 256, K), dtype=np.float32)
    embed_sq_table = np.zeros((d, 256), dtype=np.float32)

    for j in range(d):
        start = col_feat_start[j]
        dim = col_feat_dim[j]
        if col_is_ordinal[j]:
            mu, sd = ordinal_mu[j], ordinal_sd[j]
            w_col = w[start]
            V_col = V[start]  # (K,)
            V_sq_col = V_col * V_col
            for v in range(256):
                x = (v - mu) / sd
                linear_table[j, v] = w_col * x
                embed_table[j, v] = V_col * x
                embed_sq_table[j, v] = np.sum(V_sq_col) * (x * x)
        else:
            vals = cat_vals[j]
            # mapping from byte to feature index within column
            idx_map = np.full(256, dim - 1, dtype=np.int32)
            idx_map[vals] = np.arange(len(vals))
            for v in range(256):
                fi = idx_map[v]
                linear_table[j, v] = w[start + fi]
                embed_table[j, v] = V[start + fi]
                embed_sq_table[j, v] = np.sum(V[start + fi] ** 2)

    # Write weights.bin
    # Header: magic, d, K, C, w0
    # Then for each j, each v: linear_contrib (float), embed_vec (K floats), embed_sq_sum (float)
    magic = 0x464D3231  # 'FM21'
    header = struct.pack("<5i f", magic, d, K, C, 0, w0)  # last int is placeholder for alignment? Actually 5 ints + float
    # Let's pack properly: 4 ints (magic, d, K, C) then float w0
    header = struct.pack("<4i f", magic, d, K, C, w0)
    out = bytearray(header)
    # Tables
    out += linear_table.astype("<f4").tobytes()
    out += embed_table.astype("<f4").tobytes()
    out += embed_sq_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1).
 * Weights format: magic 0x464D3231, int32 d, K, C, float w0,
 *   then d*256 floats linear_contrib,
 *   then d*256*K floats embed_vec,
 *   then d*256 floats embed_sq_sum.
 * All little-endian. */
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
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x464D3231) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int d = hdr[1], K = hdr[2], C = hdr[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad weights: invalid dims\n"); return 3; }
    float w0 = *(float *)(hdr + 4);
    size_t table_bytes = (size_t)d * 256;
    size_t expected = 20 + table_bytes * (2 + K) * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    const float *linear = (const float *)(wb + 20);
    const float *embed = linear + table_bytes;
    const float *embed_sq = embed + table_bytes * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = (float *)malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear_sum = w0;
        for (int u = 0; u < K; u++) S[u] = 0.0f;
        float sq_sum = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            size_t idx = (size_t)j * 256 + v;
            linear_sum += linear[idx];
            const float *ev = embed + idx * K;
            for (int u = 0; u < K; u++) S[u] += ev[u];
            sq_sum += embed_sq[idx];
        }
        float dot = 0.0f;
        for (int u = 0; u < K; u++) dot += S[u] * S[u];
        float interaction = 0.5f * (dot - sq_sum);
        float logit = linear_sum + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(S);
    return 0;
}
```