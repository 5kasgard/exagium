```hypothesis
Family: label-fm (factorization machine with rank 2). Captures pairwise feature interactions via low-rank embeddings, which are crucial for tabular data like Adult where feature combinations (e.g., education × occupation) strongly predict income. The model adds only ~2k extra parameters over linear and inference costs ~2× linear (still <1500 Ir/item). If dev accuracy ≤ 0.8524 or Ir/item > 1500, the hypothesis fails.
```

```python
"""Factorization Machine (rank 2) for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal as z-scored scalars, categorical as one-hot over training values.
Trains with Adam on logistic loss, L2 1e-4, batch 64, 50 epochs.
Exports per-byte lookup tables: int32 magic 0x464D4C31, d, r, C; float32 w0; float32 table[d][256][1+2r]."""
import json, math, struct, sys

import numpy as np

SEED = 1
RANK = 2
LR = 0.01
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Build encoding: for each column, list of feature indices and parameters
    feat_indices = []          # list of arrays of feature indices per column
    ordinal_mu = []            # for ordinal columns
    ordinal_sd = []
    cat_maps = []              # for categorical columns: value -> feature index
    k = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu = v.mean()
            sd = v.std()
            if sd == 0:
                sd = 1.0
            ordinal_mu.append(mu)
            ordinal_sd.append(sd)
            feat_indices.append(np.array([k], dtype=np.int32))
            k += 1
            cat_maps.append(None)
        else:
            vals = np.unique(X[:, j])
            mapping = {val: idx for idx, val in enumerate(vals)}
            cat_maps.append(mapping)
            indices = np.array([k + i for i in range(len(vals))], dtype=np.int32)
            feat_indices.append(indices)
            k += len(vals)
            ordinal_mu.append(0.0)
            ordinal_sd.append(1.0)

    # Build dense feature matrix Z (n, k)
    Z = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        indices = feat_indices[j]
        if types[j] == "ordinal":
            mu = ordinal_mu[j]
            sd = ordinal_sd[j]
            Z[:, indices[0]] = (X[:, j].astype(np.float32) - mu) / sd
        else:
            mapping = cat_maps[j]
            for val, idx in mapping.items():
                mask = (X[:, j] == val)
                Z[mask, indices[idx]] = 1.0

    # Initialize parameters
    w0 = 0.0
    w = np.zeros(k, dtype=np.float32)
    V = rng.normal(0, 1 / math.sqrt(k), size=(k, RANK)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Training loop
    for _ in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]
            b = len(idx)

            # Forward
            linear = w0 + z @ w
            s_vec = z @ V
            z_sq = z * z
            V_sq = V * V
            q_vec = z_sq @ V_sq
            interaction = 0.5 * np.sum(s_vec * s_vec - q_vec, axis=1)
            logits = linear + interaction

            # Logistic loss gradient (y in {0,1})
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yy).astype(np.float32)

            # Gradients
            grad_w0 = grad.mean()
            grad_w = (z.T @ grad) / b + L2 * w
            grad_V = np.zeros_like(V)
            for f in range(RANK):
                grad_V[:, f] = (z.T @ (grad * s_vec[:, f])) / b - V[:, f] * (z_sq.T @ grad) / b + L2 * V[:, f]

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * grad_w0 * grad_w0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (math.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Build lookup tables for inference
    # table[j, v] = [linear_contribution, vec_sum[0..r-1], vec_sq_sum[0..r-1]]
    table = np.zeros((d, 256, 1 + 2 * RANK), dtype=np.float32)
    for j in range(d):
        indices = feat_indices[j]
        if types[j] == "ordinal":
            mu = ordinal_mu[j]
            sd = ordinal_sd[j]
            for v in range(256):
                x_val = (v - mu) / sd
                i = indices[0]
                table[j, v, 0] = w[i] * x_val
                for f in range(RANK):
                    table[j, v, 1 + f] = V[i, f] * x_val
                    table[j, v, 1 + RANK + f] = V[i, f] * V[i, f] * x_val * x_val
        else:
            mapping = cat_maps[j]
            for v in range(256):
                if v in mapping:
                    idx = mapping[v]
                    i = indices[idx]
                    table[j, v, 0] = w[i]
                    for f in range(RANK):
                        table[j, v, 1 + f] = V[i, f]
                        table[j, v, 1 + RANK + f] = V[i, f] * V[i, f]

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, RANK, C)
    w0_bytes = struct.pack("<f", w0)
    table_bytes = table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + w0_bytes + table_bytes)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (rank 2) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], r = hd[2], C = hd[3];
    if (d <= 0 || r <= 0 || C != 2) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * (1 + 2 * r) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float w0 = *(float *)(wb + 16);
    float *table = (float *)(wb + 20);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_vec[8], sum_vec_sq[8]; /* max rank 8 */
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float sum_linear = w0;
        for (int f = 0; f < r; f++) { sum_vec[f] = 0.0f; sum_vec_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            uint8_t byte = xi[j];
            float *entry = table + ((long)j * 256 + byte) * (1 + 2 * r);
            sum_linear += entry[0];
            for (int f = 0; f < r; f++) {
                sum_vec[f] += entry[1 + f];
                sum_vec_sq[f] += entry[1 + r + f];
            }
        }

        float interaction = 0.0f;
        for (int f = 0; f < r; f++) {
            interaction += sum_vec[f] * sum_vec[f] - sum_vec_sq[f];
        }
        interaction *= 0.5f;
        float score = sum_linear + interaction;
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```