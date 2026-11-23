```hypothesis
Family: label-fm (Factorization Machine). Adult benefits from pairwise feature interactions (e.g., education × occupation) which linear models miss. A rank-4 FM adds only ~130KB weights and ~100 FLOPs/item, so it should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at lower inference cost. If dev accuracy ≤ 0.853, the interaction gain is negligible and the family is not worthwhile.
```

```python
"""Factorization Machine for tabular label tasks (k=4).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 10


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X_bytes.shape[1]
    types = card["input"]["types"]
    C = card["classes"]

    # Build feature mapping
    feat_of_col = []          # list of (kind, info)
    # kind: 'ordinal' -> (feat_idx, mean, std)
    # kind: 'categorical' -> (start_feat_idx, cat_map)
    n_feats = 0
    for j, t in enumerate(types):
        col = X_bytes[:, j]
        if t == "ordinal":
            mu = col.mean()
            sd = col.std()
            if sd == 0:
                sd = 1.0
            feat_of_col.append(('ordinal', (n_feats, float(mu), float(sd))))
            n_feats += 1
        else:
            uniq = np.unique(col)
            uniq = uniq[uniq != 255]          # 255 = unseen
            cat_map = {int(v): i for i, v in enumerate(uniq)}
            feat_of_col.append(('categorical', (n_feats, cat_map)))
            n_feats += len(uniq)

    # Dense feature matrix
    n = X_bytes.shape[0]
    X = np.zeros((n, n_feats), dtype=np.float32)
    for j, (kind, info) in enumerate(feat_of_col):
        col = X_bytes[:, j]
        if kind == 'ordinal':
            f_idx, mu, sd = info
            X[:, f_idx] = (col.astype(np.float32) - mu) / sd
        else:
            start, cat_map = info
            for v, idx in cat_map.items():
                mask = (col == v)
                X[mask, start + idx] = 1.0

    # FM parameters
    rng = np.random.default_rng(SEED)
    w0 = 0.0
    w = np.zeros(n_feats, np.float32)
    V = (rng.normal(0, 0.01, (n_feats, K))).astype(np.float32)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_w0, v_w0 = 0.0, 0.0
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]
            yb = y[idx]

            # Forward
            linear = Xb @ w + w0
            s_vec = Xb @ V                     # (batch, K)
            s2_vec = (Xb ** 2) @ (V ** 2)      # (batch, K)
            fm = 0.5 * np.sum(s_vec ** 2 - s2_vec, axis=1)
            logits = linear + fm

            # Logistic loss gradient
            probs = 1.0 / (1.0 + np.exp(-logits))
            g = probs - yb                     # (batch,)

            # Gradients
            gw0 = g.sum()
            gw = Xb.T @ g + L2 * w
            # dFM/dV = X^T (g * s) - (X^2)^T (g * V)
            g_exp = g[:, None]
            gV = Xb.T @ (g_exp * s_vec) - (Xb ** 2).T @ (g_exp * V) + L2 * V

            t += 1
            # Adam update
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Build lookup tables for inference: per column, per byte value
    # linear_term[d][256], latent_vec[d][256][K], latent_vec_sq[d][256][K]
    linear_term = np.zeros((d, 256), np.float32)
    latent_vec = np.zeros((d, 256, K), np.float32)
    latent_vec_sq = np.zeros((d, 256, K), np.float32)

    for j, (kind, info) in enumerate(feat_of_col):
        if kind == 'ordinal':
            f_idx, mu, sd = info
            for v in range(256):
                x = (v - mu) / sd
                linear_term[j, v] = w[f_idx] * x
                latent_vec[j, v] = V[f_idx] * x
                latent_vec_sq[j, v] = (V[f_idx] * x) ** 2
        else:
            start, cat_map = info
            for v in range(256):
                if v in cat_map:
                    f_idx = start + cat_map[v]
                    linear_term[j, v] = w[f_idx]
                    latent_vec[j, v] = V[f_idx]
                    latent_vec_sq[j, v] = V[f_idx] ** 2
                else:
                    linear_term[j, v] = 0.0
                    latent_vec[j, v] = 0.0
                    latent_vec_sq[j, v] = 0.0

    # Write weights.bin
    # Header: magic, d, K, n_feats (for validation)
    magic = 0x464D3132  # "FM12"
    header = struct.pack("<4i", magic, d, K, n_feats)
    # Total floats = 1 (w0) + d * 256 * (1 + 2*K)
    expected_floats = 1 + d * 256 * (1 + 2 * K)
    out = bytearray()
    out += header
    out += np.array([w0], dtype=np.float32).tobytes()
    out += linear_term.astype(np.float32).tobytes()
    out += latent_vec.astype(np.float32).tobytes()
    out += latent_vec_sq.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (d=14). Output: one byte class index. */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], n_feats = hd[3];
    if (d != 14 || K != 4) { fprintf(stderr, "unexpected d or K\n"); return 3; }

    /* expected file size: header(16) + floats * 4 */
    long expected_floats = 1 + d * 256 * (1 + 2 * K);
    if (nw != 16 + expected_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0]; p++;
    const float *linear_term = p; p += d * 256;
    const float *latent_vec = p; p += d * 256 * K;
    const float *latent_vec_sq = p; p += d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float s[4], s2[4];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int u = 0; u < K; u++) { s[u] = 0.0f; s2[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            int base = j * 256;
            linear += linear_term[base + v];
            const float *lv = latent_vec + (base + v) * K;
            const float *lvsq = latent_vec_sq + (base + v) * K;
            for (int u = 0; u < K; u++) {
                s[u] += lv[u];
                s2[u] += lvsq[u];
            }
        }

        float fm = 0.0f;
        for (int u = 0; u < K; u++) fm += s[u] * s[u] - s2[u];
        fm *= 0.5f;
        float score = linear + fm;

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```