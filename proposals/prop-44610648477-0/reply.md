```hypothesis
We propose a Factorization Machine family "label-fm". FM models pairwise feature interactions via low-rank embeddings (size k=8), which are known to be effective on tabular data like Adult. It is more expressive than linear models but much faster than GBT or MLP at inference: only O(d·k) operations per item (d=14, k=8). We expect dev accuracy >0.86 (beating label-linear's 0.8524) at similar or lower CPU cost. If accuracy ≤0.8524 or inference cost exceeds label-linear, the hypothesis fails.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-scored single value; categorical -> one-hot over training values.
FM: w0 + z·w_lin + 0.5*sum_c((z·V_c)^2 - z·(V_c^2)) with k=8.
Adam lr=0.01, L2=1e-4, batch=256, epochs=min(100, ceil(600000/n)).
Exported as per-byte lookup tables: int32 magic 0x464D3132, d, k, C; float32 w0; float32 table[d][256][1+k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 600000, 100


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

    # Build encoding for each feature
    enc_dims = []
    zscore_tables = []
    cat_maps = []
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            vals = X[:, j].astype(np.float64)
            mu, sd = vals.mean(), vals.std() or 1.0
            zscore_tables.append(((np.arange(256) - mu) / sd).astype(np.float32))
            enc_dims.append(1)
            cat_maps.append(None)
        else:
            uniq = np.unique(X[:, j])
            mp = np.full(256, -1, dtype=np.int32)
            mp[uniq] = np.arange(len(uniq))
            cat_maps.append(mp)
            enc_dims.append(len(uniq))
            zscore_tables.append(None)

    D = sum(enc_dims)
    n = X.shape[0]

    # Build dense encoded matrix Z (n x D)
    Z = np.zeros((n, D), dtype=np.float32)
    off = 0
    for j in range(d):
        if enc_dims[j] == 1:
            Z[:, off] = (X[:, j].astype(np.float32) - zscore_tables[j][0]) / (zscore_tables[j][1] - zscore_tables[j][0]) if zscore_tables[j][1] != zscore_tables[j][0] else 0.0
            # Actually we have zscore_table for all 256 values
            Z[:, off] = zscore_tables[j][X[:, j]]
            off += 1
        else:
            mp = cat_maps[j]
            idx = mp[X[:, j]]
            rows = np.arange(n)
            valid = idx >= 0
            Z[rows[valid], off + idx[valid]] = 1.0
            off += enc_dims[j]

    # Initialize parameters
    w0 = 0.0
    w_lin = np.zeros(D, dtype=np.float32)
    V = (rng.normal(0, 0.01, size=(D, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w_lin = np.zeros_like(w_lin)
    v_w_lin = np.zeros_like(w_lin)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]
            bsz = len(idx)

            # Forward
            linear_part = z @ w_lin
            sum_v = z @ V
            sum_v2 = z @ (V * V)
            interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
            logits = w0 + linear_part + interaction

            # Gradient
            probs = 1.0 / (1.0 + np.exp(-logits))
            dlogits = probs - yy

            dw0 = dlogits.mean()
            dw_lin = (z.T @ dlogits) / bsz + L2 * w_lin
            dV = (z.T @ (dlogits[:, None] * sum_v)) / bsz - (z.T @ dlogits)[:, None] * V / bsz + L2 * V

            t += 1
            # Adam updates
            m_w0 = 0.9 * m_w0 + 0.1 * dw0
            v_w0 = 0.999 * v_w0 + 0.001 * dw0 * dw0
            w0 -= LR * (m_w0 / (1 - 0.9**t)) / (np.sqrt(v_w0 / (1 - 0.999**t)) + 1e-8)

            m_w_lin = 0.9 * m_w_lin + 0.1 * dw_lin
            v_w_lin = 0.999 * v_w_lin + 0.001 * dw_lin * dw_lin
            w_lin -= LR * (m_w_lin / (1 - 0.9**t)) / (np.sqrt(v_w_lin / (1 - 0.999**t)) + 1e-8)

            m_V = 0.9 * m_V + 0.1 * dV
            v_V = 0.999 * v_V + 0.001 * dV * dV
            V -= LR * (m_V / (1 - 0.9**t)) / (np.sqrt(v_V / (1 - 0.999**t)) + 1e-8)

    # Build lookup tables: table[j][b] = (linear_weight, embedding_vector[K])
    table = np.zeros((d, 256, 1 + K), dtype=np.float32)
    off = 0
    for j in range(d):
        if enc_dims[j] == 1:
            ztab = zscore_tables[j]
            table[j, :, 0] = ztab * w_lin[off]
            table[j, :, 1:] = ztab[:, None] * V[off]
            off += 1
        else:
            mp = cat_maps[j]
            for b in range(256):
                ci = mp[b]
                if ci >= 0:
                    table[j, b, 0] = w_lin[off + ci]
                    table[j, b, 1:] = V[off + ci]
            off += enc_dims[j]

    # Write weights.bin
    magic = 0x464D3132  # "FM12"
    header = struct.pack("<4if", magic, d, K, C, w0)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(table.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0 or 1).
 * Weights format: int32 magic(0x464D3132), d, k, C; float32 w0; float32 table[d][256][1+k]. */
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
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || K > 256 || d < 1 || d > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    float w0 = *(float *)(hd + 4);
    long expected = 20L + (long)d * 256 * (1 + K) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *table = (const float *)(wb + 20);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = calloc(K, sizeof(float));
    float *sum_v2 = calloc(K, sizeof(float));
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear_sum = w0;
        memset(sum_v, 0, K * sizeof(float));
        memset(sum_v2, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = table + ((long)j * 256 + xi[j]) * (1 + K);
            linear_sum += t[0];
            for (int c = 0; c < K; c++) {
                float v = t[1 + c];
                sum_v[c] += v;
                sum_v2[c] += v * v;
            }
        }
        float interaction = 0.0f;
        for (int c = 0; c < K; c++) interaction += sum_v[c] * sum_v[c] - sum_v2[c];
        interaction *= 0.5f;
        float logit = linear_sum + interaction;
        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```