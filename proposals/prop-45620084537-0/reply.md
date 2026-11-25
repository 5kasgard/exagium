```hypothesis
Family: label-fm (Factorization Machine). Adult benefits from feature interactions (e.g., education × occupation, capital-gain × hours). A rank-8 FM adds pairwise interactions with O(k·d) inference cost, beating linear while staying faster than MLP/GBDT. If dev accuracy ≤ 0.8524 or Ir/item > 1000, the approach fails.
```
```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal z-scored, categorical one-hot (unseen=255). Rank-8 FM with Adam.
Exported: int32 magic 0x464D3130, d, k, C; float32 bias[C], linear[d][C], factor[d][k][C]."""
import json, math, struct, sys

import numpy as np

RANK = 8
SEED, LR, L2, L2_F, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 1e-4, 64, 600000, 100


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

    # Encode features same as label-linear
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, d = Z.shape

    # FM parameters: bias[C], linear[d][C], factor[d][RANK][C]
    bias = np.zeros(C, np.float32)
    linear = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    factor = (rng.normal(0, 0.01, (d, RANK, C))).astype(np.float32)

    params = [bias, linear, factor]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            m = z.shape[0]

            # Linear term: z @ linear + bias
            lin = z @ linear + bias  # [m, C]

            # FM interaction term: 0.5 * sum_k ( (sum_i z_i v_ik)^2 - sum_i z_i^2 v_ik^2 )
            # z: [m, d], factor: [d, k, C] -> zv: [m, k, C]
            zv = np.einsum('md,dkc->mkc', z, factor)  # sum_i z_i v_ikc
            zv_sq = zv * zv
            z_sq = z * z  # [m, d]
            v_sq = factor * factor  # [d, k, C]
            z_sq_v_sq = np.einsum('md,dkc->mkc', z_sq, v_sq)  # sum_i z_i^2 v_ikc^2
            inter = 0.5 * (zv_sq - z_sq_v_sq).sum(axis=1)  # [m, C]

            logits = lin + inter
            g = softmax_grad(logits, yy)  # [m, C]

            # Gradients
            g_bias = g.sum(axis=0)  # [C]
            g_linear = z.T @ g + L2 * linear  # [d, C]

            # Factor gradient: d/dv_ikc = sum_m g_mc * z_mi * (zv_mkc - z_mi * v_ikc)
            # = z.T @ (g[:, None, :] * zv) - (z_sq.T @ (g[:, None, :] * factor))
            g_factor = np.einsum('md,mkc->dkc', z, g[:, None, :] * zv) \
                     - np.einsum('md,dkc->dkc', z_sq, g[:, None, :] * factor) \
                     + L2_F * factor

            grads = [g_bias, g_linear, g_factor]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_, v_ = state[i]
                m_ *= 0.9; m_ += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                p -= LR * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)

    # Export weights
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D3130, d, RANK, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += factor.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d raw bytes. Output: 1 byte class (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    long expected = 16L + 4L * (C + (long)d * C + (long)d * k * C);
    if (C < 1 || C > 256 || nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *factor = linear + (long)d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *zv = malloc(sizeof(float) * k * C);
    float *zv_sq = malloc(sizeof(float) * k * C);
    float *z_sq_v_sq = malloc(sizeof(float) * k * C);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        // Initialize scores with bias
        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // Linear term
        for (int j = 0; j < d; j++) {
            float z = (xi[j] == 255 && 0) ? 0.0f : 1.0f;  // placeholder, we need actual encoding
        }
        // We need to replicate the encoding from train.py in C.
        // Since the encoding depends on training statistics, we must export the encoding tables.
        // But the contract says weights.bin is the only artifact. We need to include encoding in weights.
        // Let me redesign: export per-byte lookup tables for linear and factor terms like label-linear does.
        fprintf(stderr, "Encoding not implemented in infer.c\n"); return 3;
    }
    return 0;
}
```

Wait, I made a mistake in the infer.c - I need to include the encoding tables in the weights file, just like label-linear does. Let me fix this properly.

The train.py should export per-byte lookup tables for the linear and factor terms, so inference just does table lookups. This is what label-linear does with its T tables.

Let me redo this properly.```hypothesis
Family: label-fm (Factorization Machine). Adult benefits from feature interactions (e.g., education × occupation, capital-gain × hours). A rank-8 FM adds pairwise interactions with O(k·d) inference cost, beating linear while staying faster than MLP/GBDT. If dev accuracy ≤ 0.8524 or Ir/item > 1000, the approach fails.
```
```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal z-scored, categorical one-hot (unseen=255). Rank-8 FM with Adam.
Exports per-byte lookup tables for linear and factor terms, like label-linear."""
import json, math, struct, sys

import numpy as np

RANK = 8
SEED, LR, L2, L2_F, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 1e-4, 64, 600000, 100


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

    # Encode features: build per-column encoding matrices E_j [256, dim_j]
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])  # [256, 1]
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)  # [256, n_categories]
    # Concatenated feature dimension
    d = sum(E.shape[1] for E in enc)

    # Build full design matrix Z [n, d]
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # FM parameters: bias[C], linear[d][C], factor[d][RANK][C]
    bias = np.zeros(C, np.float32)
    linear = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    factor = (rng.normal(0, 0.01, (d, RANK, C))).astype(np.float32)

    params = [bias, linear, factor]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            m = z.shape[0]

            # Linear term
            lin = z @ linear + bias  # [m, C]

            # FM interaction: 0.5 * sum_k ( (sum_i z_i v_ik)^2 - sum_i z_i^2 v_ik^2 )
            zv = np.einsum('md,dkc->mkc', z, factor)          # [m, k, C]
            zv_sq = zv * zv
            z_sq = z * z
            v_sq = factor * factor
            z_sq_v_sq = np.einsum('md,dkc->mkc', z_sq, v_sq)  # [m, k, C]
            inter = 0.5 * (zv_sq - z_sq_v_sq).sum(axis=1)     # [m, C]

            logits = lin + inter
            g = softmax_grad(logits, yy)  # [m, C]

            # Gradients
            g_bias = g.sum(axis=0)
            g_linear = z.T @ g + L2 * linear

            g_factor = np.einsum('md,mkc->dkc', z, g[:, None, :] * zv) \
                     - np.einsum('md,dkc->dkc', z_sq, g[:, None, :] * factor) \
                     + L2_F * factor

            grads = [g_bias, g_linear, g_factor]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_, v_ = state[i]
                m_ *= 0.9; m_ += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                p -= LR * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)

    # Export per-byte lookup tables for fast inference
    # For each input column j (0..13) and byte value b (0..255), we need:
    # - linear contribution: E_j[b] @ linear[offset:offset+dim_j]  -> [C]
    # - factor contribution: E_j[b] @ factor[offset:offset+dim_j]  -> [RANK, C]
    # where offset is the start column in the concatenated feature space.
    linear_tables = []  # list of [256, C]
    factor_tables = []  # list of [256, RANK, C]
    off = 0
    for E in enc:
        dim = E.shape[1]
        # Linear table: [256, C]
        Lt = (E.astype(np.float32) @ linear[off:off+dim]).astype(np.float32)
        linear_tables.append(Lt)
        # Factor table: [256, RANK, C]
        Ft = np.einsum('bi,drc->brc', E.astype(np.float32), factor[off:off+dim]).astype(np.float32)
        factor_tables.append(Ft)
        off += dim

    # Pack weights: magic, d_in=14, k, C; bias[C]; linear_tables[14][256][C]; factor_tables[14][256][k][C]
    out = struct.pack("<4i", 0x464D3130, 14, RANK, C)
    out += bias.astype("<f4").tobytes()
    for Lt in linear_tables:
        out += Lt.astype("<f4").tobytes()
    for Ft in factor_tables:
        out += Ft.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: 14 raw bytes. Output: 1 byte class (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d_in = hd[1], k = hd[2], C = hd[3];
    if (d_in != 14 || C < 1 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * (C + 14L * 256 * C + 14L * 256 * k * C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear_tables = bias + C;                              // [14][256][C]
    const float *factor_tables = linear_tables + 14L * 256 * C;         // [14][256][k][C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *scores = malloc(sizeof(float) * C);
    float *zv = malloc(sizeof(float) * k * C);
    float *zv_sq = malloc(sizeof(float) * k * C);
    float *z_sq_v_sq = malloc(sizeof(float) * k * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if (len != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, len); return 3; }

        // Initialize scores with bias
        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // Clear FM accumulators
        for (int kc = 0; kc < k * C; kc++) {
            zv[kc] = 0.0f;
            z_sq_v_sq[kc] = 0.0f;
        }

        // Accumulate linear and factor contributions per column
        for (int j = 0; j < 14; j++) {
            unsigned char b = xi[j];
            const float *Lt = linear_tables + ((long)j * 256 + b) * C;
            const float *Ft = factor_tables + ((long)j * 256 + b) * k * C;
            for (int c = 0; c < C; c++) scores[c] += Lt[c];
            for (int kc = 0; kc < k * C; kc++) zv[kc] += Ft[kc];
            // For z_sq_v_sq we need sum of (z^2 * v^2). Since z is 0/1 for one-hot and continuous for ordinal,
            // we need the squared encoding values. But our tables store E[b] @ factor, not (E[b]^2) @ (factor^2).
            // We must export separate tables for squared terms, or compute differently.
            // Actually, for one-hot: z is 0 or 1, so z^2 = z. For ordinal: z is continuous, z^2 != z.
            // We need to export squared factor tables too. Let me fix the export.
            fprintf(stderr, "Squared term not handled\n"); return 3;
        }
    }
    return 0;
}
```

I need to fix the infer.c to handle the squared term properly. The FM formula needs both `sum(z_i * v_i)` and `sum(z_i^2 * v_i^2)`. Since we're using per-byte lookup tables, we need to export tables for the squared terms as well, or we can compute z_i^2 on the fly if we also export the raw encoding values.

Actually, looking at label-linear, it only exports the linear projection tables. For FM, we need:
- For linear term: table_L[j][b][c] = E_j[b] @ linear
- For factor term: table_F[j][b][k][c] = E_j[b] @ factor
- For squared factor term: table_F2[j][b][k][c] = (E_j[b]^2) @ (factor^2)

But wait, z_i^2 * v_i^2 is element-wise. For one-hot categorical, z_i is 0 or 1, so z_i^2 = z_i. For ordinal, z_i is the z-scored value, so z_i^2 is its square.

We can either:
1. Export separate tables for the squared terms
2. Export the raw encoding values E_j[b] and compute squares at inference time

Option 2 is more memory efficient. Let's export the encoding tables (like label-linear's T) plus the linear and factor weights, and compute everything at inference.

Actually, label-linear exports T = E @ W1 which is the linear projection. For FM we need:
- Linear projection: E @ linear
- Factor projection: E @ factor
- Squared factor projection: (E^2) @ (factor^2)

We can export three tables per column, or export E and the weights separately. Since d=14 and 256 values, exporting E (14 * 256 * max_dim) might be large. But max_dim for categorical is number of categories, for ordinal is 1.

Let me redesign: Export the encoding matrices E (as float32) and the weights linear, factor. Then inference computes:
- For each column j, get E_j[b] (vector of dim_j)
- Linear: dot(E_j[b], linear[off:off+dim_j])
- Factor: dot(E_j[b], factor[off:off+dim_j]) -> [k, C]
- Squared: dot(E_j[b]^2, factor[off:off+dim_j]^2) -> [k, C]

This is cleaner and matches label-linear's approach of exporting T = E @ W1.

But label-linear exports T which is already the projection. For FM we need the raw E for the squared term. Let's export E for each column.

Actually, looking at the label-linear train.py:
```python
off, T = 0, []
for E in enc:
    T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32)); off += E.shape[1]
out = struct.pack("<4i", 0x54424C31, X.shape[1], H, C) + np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
```

So it exports T = E @ W1 per column, stacked. The infer.c then does:
```c
for (int j = 0; j < d; j++) {
    const float *t = T + ((long)j * 256 + xi[j]) * W;
    for (int w = 0; w < W; w++) acc[w] += t[w];
}
```

For FM, I should export per-column tables for:
1. Linear projection: E @ linear -> [256, C]
2. Factor projection: E @ factor -> [256, k, C]
3. Squared factor projection: (E^2) @ (factor^2) -> [256, k, C]

That's three tables per column. Let me rewrite train.py and infer.c properly.```hypothesis
Family: label-fm (Factorization Machine). Adult benefits from feature interactions (e.g., education × occupation, capital-gain × hours). A rank-8 FM adds pairwise interactions with O(k·d) inference cost, beating linear while staying faster than MLP/GBDT. If dev accuracy ≤ 0.8524 or Ir/item > 1000, the approach fails.
```
```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal z-scored, categorical one-hot (unseen=255). Rank-8 FM with Adam.
Exports per-byte lookup tables: linear[14][256][C], factor[14][256][k][C], factor2[14][256][k][C]."""
import json, math, struct, sys

import numpy as np

RANK = 8
SEED, LR, L2, L2_F, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 1e-4, 64, 600000, 100


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

    # Build encoding matrices E_j [256, dim_j]
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])  # [256, 1]
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)  # [256, n_categories]
    d = sum(E.shape[1] for E in enc)  # total feature dimension

    # Full design matrix
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # FM parameters
    bias = np.zeros(C, np.float32)
    linear = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    factor = (rng.normal(0, 0.01, (d, RANK, C))).astype(np.float32)

    params = [bias, linear, factor]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            m = z.shape[0]

            lin = z @ linear + bias
            zv = np.einsum('md,dkc->mkc', z, factor)
            zv_sq = zv * zv
            z_sq = z * z
            v_sq = factor * factor
            z_sq_v_sq = np.einsum('md,dkc->mkc', z_sq, v_sq)
            inter = 0.5 * (zv_sq - z_sq_v_sq).sum(axis=1)

            logits = lin + inter
            g = softmax_grad(logits, yy)

            g_bias = g.sum(axis=0)
            g_linear = z.T @ g + L2 * linear
            g_factor = np.einsum('md,mkc->dkc', z, g[:, None, :] * zv) \
                     - np.einsum('md,dkc->dkc', z_sq, g[:, None, :] * factor) \
                     + L2_F * factor

            grads = [g_bias, g_linear, g_factor]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_, v_ = state[i]
                m_ *= 0.9; m_ += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                p -= LR * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)

    # Export per-byte lookup tables
    linear_tables = []   # [14][256][C]
    factor_tables = []   # [14][256][k][C]
    factor2_tables = []  # [14][256][k][C]  for (E^2) @ (factor^2)
    off = 0
    for E in enc:
        dim = E.shape[1]
        E_f32 = E.astype(np.float32)
        E_sq = (E_f32 * E_f32)
        W_lin = linear[off:off+dim]
        W_fac = factor[off:off+dim]
        W_fac_sq = W_fac * W_fac
        # Linear: [256, C]
        linear_tables.append((E_f32 @ W_lin).astype(np.float32))
        # Factor: [256, k, C]
        factor_tables.append(np.einsum('bi,drc->brc', E_f32, W_fac).astype(np.float32))
        # Factor squared: [256, k, C]
        factor2_tables.append(np.einsum('bi,drc->brc', E_sq, W_fac_sq).astype(np.float32))
        off += dim

    # Pack: magic, d_in=14, k, C; bias[C]; linear_tables; factor_tables; factor2_tables
    out = struct.pack("<4i", 0x464D3130, 14, RANK, C)
    out += bias.astype("<f4").tobytes()
    for T in linear_tables:
        out += T.astype("<f4").tobytes()
    for T in factor_tables:
        out += T.astype("<f4").tobytes()
    for T in factor2_tables:
        out += T.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: 14 raw bytes. Output: 1 byte class (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d_in = hd[1], k = hd[2], C = hd[3];
    if (d_in != 14 || C < 1 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * (C + 3L * 14L * 256 * k * C);  // wait, linear is C not k*C
    // Actually: bias[C] + linear[14*256*C] + factor[14*256*k*C] + factor2[14*256*k*C]
    expected = 16L + 4L * (C + 14L * 256 * C + 2L * 14L * 256 * k * C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear_tables = bias + C;                                    // [14][256][C]
    const float *factor_tables = linear_tables + 14L * 256 * C;               // [14][256][k][C]
    const float *factor2_tables = factor_tables + 14L * 256 * k * C;          // [14][256][k][C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *scores = malloc(sizeof(float) * C);
    float *zv = malloc(sizeof(float) * k * C);
    float *z_sq_v_sq = malloc(sizeof(float) * k * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if (len != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, len); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int kc = 0; kc < k * C; kc++) {
            zv[kc] = 0.0f;
            z_sq_v_sq[kc] = 0.0f;
        }

        for (int j = 0; j < 14; j++) {
            unsigned char b = xi[j];
            const float *Lt = linear_tables + ((long)j * 256 + b) * C;
            const float *Ft = factor_tables + ((long)j * 256 + b) * k * C;
            const float *F2t = factor2_tables + ((long)j * 256 + b) * k * C;
            for (int c = 0; c < C; c++) scores[c] += Lt[c];
            for (int kc = 0; kc < k * C; kc++) {
                zv[kc] += Ft[kc];
                z_sq_v_sq[kc] += F2t[kc];
            }
        }

        // FM interaction: 0.5 * sum_k (zv^2 - z_sq_v_sq)
        for (int kc = 0; kc < k * C; kc++) {
            float diff = zv[kc] * zv[kc] - z_sq_v_sq[kc];
            scores[kc % C] += 0.5f * diff;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```