```hypothesis
Factorization Machine (FM) with rank-8 interactions captures pairwise feature crosses (e.g., education×occupation) crucial for tabular data while keeping inference fast via precomputed per-byte tables. It should beat label-linear (0.8524) on Adult because the dataset is known to benefit from low-order interactions. If dev accuracy ≤ 0.8524 or inference cost exceeds label-linear by >2×, the family is not justified.
```

```python
"""Factorization Machine (rank 8) for tabular classification with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D3132, d, r, total_features; float32 w0; then per-column tables:
for each column j (0..d-1), each byte value b (0..255): linear[b], factor_sum[b][0..r-1], factor_sq_sum[b][0..r-1]."""
import json, math, struct, sys

import numpy as np

RANK = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100


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
    rng = np.random.default_rng(SEED)

    # Build encodings and feature slices
    enc = []
    feat_sizes = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        feat_sizes.append(E.shape[1])
    total_feat = sum(feat_sizes)

    # Concatenate to Z (n, total_feat)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape
    assert k == total_feat

    # FM parameters
    w0 = np.float32(0.0)
    w = np.zeros(k, dtype=np.float32)
    V = (rng.normal(0, 0.01, (k, RANK))).astype(np.float32)

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
            z, yy = Z[idx], y[idx]
            bsz = len(yy)

            # Forward
            linear = z @ w
            S = z @ V                     # (bsz, RANK)
            S2 = (z * z) @ (V * V)        # (bsz, RANK)
            interaction = 0.5 * np.sum(S * S - S2, axis=1)
            logit = w0 + linear + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            dlogit = (p - yy).astype(np.float32)  # (bsz,)

            # Gradients
            g_w0 = np.mean(dlogit)
            g_w = (z.T @ dlogit) / bsz + L2 * w
            # g_V = (z.T @ (dlogit[:,None] * (S - V[None,:,:]))) / bsz + L2 * V
            # Efficient: term1 = (z * dlogit[:,None]).T @ S; term2 = V * (z * dlogit[:,None]).sum(0)[:,None]
            zw = z * dlogit[:, None]
            term1 = zw.T @ S
            term2 = V * zw.sum(axis=0)[:, None]
            g_V = (term1 - term2) / bsz + L2 * V

            t += 1
            # Adam updates
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Precompute per-column tables
    tables = []
    off = 0
    for j in range(d):
        f = feat_sizes[j]
        w_j = w[off:off + f]
        V_j = V[off:off + f, :]
        V_j_sq = V_j * V_j
        E = enc[j]  # (256, f)
        linear_col = E @ w_j                    # (256,)
        factor_sum_col = E @ V_j                # (256, RANK)
        factor_sq_sum_col = E @ V_j_sq          # (256, RANK)
        # Interleave: for each b: linear, factor_sum[0..RANK-1], factor_sq_sum[0..RANK-1]
        col_table = np.empty((256, 1 + 2 * RANK), dtype=np.float32)
        col_table[:, 0] = linear_col
        col_table[:, 1:1 + RANK] = factor_sum_col
        col_table[:, 1 + RANK:] = factor_sq_sum_col
        tables.append(col_table)
        off += f

    # Write weights.bin
    magic = 0x464D3132
    header = struct.pack("<4if", magic, d, RANK, total_feat, w0)
    body = np.stack(tables, axis=0).astype("<f4").tobytes()  # (d, 256, 1+2*RANK)
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record = d bytes (d from header). Output = 1 byte class (0/1). */
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

    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)(wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24);
    int r = (int)(wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24);
    int total_feat = (int)(wb[12] | wb[13] << 8 | wb[14] << 16 | wb[15] << 24);
    float w0 = *(float*)(wb + 16);

    long table_bytes = (long)d * 256 * (1 + 2 * r) * 4;
    if (nw != 20 + table_bytes) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *table = (const float*)(wb + 20);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear_term = 0.0f;
        float *S = (float*)alloca(r * sizeof(float));
        float *S2 = (float*)alloca(r * sizeof(float));
        for (int f = 0; f < r; f++) { S[f] = 0.0f; S2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char b = xi[j];
            long base = ((long)j * 256 + b) * (1 + 2 * r);
            linear_term += table[base];
            for (int f = 0; f < r; f++) {
                S[f]  += table[base + 1 + f];
                S2[f] += table[base + 1 + r + f];
            }
        }
        float interaction = 0.0f;
        for (int f = 0; f < r; f++) interaction += 0.5f * (S[f] * S[f] - S2[f]);
        float logit = w0 + linear_term + interaction;
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```