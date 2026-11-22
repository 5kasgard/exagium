```hypothesis
Family: label-fm (degree-2 Factorization Machine with rank 4).
Why: The Adult dataset has known feature interactions (e.g., education × occupation, marital-status × relationship). A linear model misses these; an FM captures all pairwise interactions with O(d·k) parameters (d≈107, k=4 → ~500 params), staying small and fast. Training uses the same one-hot/z-score encoding and Adam schedule as label-linear, so it's deterministic and fits the 20 min/8 GiB budget.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-mlp16 (≈2300 Ir/item), the model doesn't justify its complexity.
```

```python
"""Factorization Machine (degree 2, rank 4) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-scored, categorical -> one-hot over training values (255 = unseen).
Weights format: magic 0x464D4C31, d, k, C, n_cols; per-column encoding; w0, w[d], V[d*k] (all float32)."""
import json, math, struct, sys

import numpy as np

K = 4
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
    C = card["classes"]
    n_samples, n_cols = X.shape
    types = card["input"]["types"]

    rng = np.random.default_rng(SEED)

    # Build encoding info and design matrix Z (n_samples x d)
    col_info = []  # list of dicts per column
    d = 0
    Z = np.zeros((n_samples, 0), dtype=np.float32)

    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            feat_idx = d
            d += 1
            col_info.append({"type": 0, "mu": float(mu), "sd": float(sd), "feat_idx": feat_idx})
            Z_col = ((v - mu) / sd).astype(np.float32).reshape(-1, 1)
            Z = np.hstack([Z, Z_col])
        else:  # categorical
            vals = np.unique(X[:, j])
            vals = vals[vals != 255]  # 255 = unseen, not a real category
            num_cat = len(vals)
            mapping = np.full(256, -1, dtype=np.int8)
            for idx, val in enumerate(vals):
                mapping[val] = idx
            feat_offset = d
            d += num_cat
            col_info.append({"type": 1, "num_cat": num_cat, "feat_offset": feat_offset, "mapping": mapping})
            Z_col = np.zeros((n_samples, num_cat), dtype=np.float32)
            for i in range(n_samples):
                local = mapping[X[i, j]]
                if local >= 0:
                    Z_col[i, local] = 1.0
            Z = np.hstack([Z, Z_col])

    # Initialize FM parameters
    w0 = 0.0
    w = np.zeros(d, dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, K))).astype(np.float32)

    # Adam state
    params = [w0, w, V]
    shapes = [(), (d,), (d, K)]
    m = [np.zeros(s, dtype=np.float32) for s in shapes]
    v = [np.zeros(s, dtype=np.float32) for s in shapes]
    t = 0

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(Z_batch):
        # Z_batch: (B, d)
        linear = w0 + Z_batch @ w
        sum_vx = Z_batch @ V               # (B, K)
        sum_v2_x2 = (Z_batch**2) @ (V**2)  # (B, K)
        interaction = 0.5 * np.sum(sum_vx**2 - sum_v2_x2, axis=1)
        return linear + interaction

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_samples))
    for _ in range(max_epochs):
        order = rng.permutation(n_samples)
        for s in range(0, n_samples, BATCH):
            idx = order[s:s + BATCH]
            Z_batch = Z[idx]
            y_batch = y[idx]
            B = len(idx)

            logits = forward(Z_batch)
            p = sigmoid(logits)
            grad_logits = (p - y_batch).astype(np.float32)  # (B,)

            # Gradient for w0
            g_w0 = grad_logits.mean()

            # Gradient for w
            g_w = (Z_batch.T @ grad_logits) / B + L2 * w

            # Gradient for V
            sum_vx = Z_batch @ V  # (B, K)
            g_V = np.zeros((d, K), dtype=np.float32)
            for f in range(K):
                # term = grad_logits[:, None] * (sum_vx[:, f:f+1] * Z_batch - V[:, f] * (Z_batch**2))
                term = grad_logits[:, None] * (sum_vx[:, f:f+1] * Z_batch - V[:, f] * (Z_batch**2))
                g_V[:, f] = term.sum(axis=0) / B + L2 * V[:, f]

            # Adam update
            t += 1
            grads = [np.array(g_w0, dtype=np.float32), g_w, g_V]
            for i, (p_arr, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p_arr -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, K, C, n_cols
    out = bytearray()
    out += struct.pack("<5i", 0x464D4C31, d, K, C, n_cols)
    for info in col_info:
        if info["type"] == 0:  # ordinal
            out += struct.pack("<b f f i", 0, info["mu"], info["sd"], info["feat_idx"])
        else:  # categorical
            out += struct.pack("<b i i", 1, info["num_cat"], info["feat_offset"])
            out += info["mapping"].tobytes()
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (degree 2, rank 4) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is 14 bytes; output is one byte (class index 0 or 1). */
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

typedef struct {
    int8_t type;          // 0 ordinal, 1 categorical
    float mu, sd;         // ordinal
    int32_t feat_idx;     // ordinal
    int32_t num_cat;      // categorical
    int32_t feat_offset;  // categorical
    int8_t mapping[256];  // categorical
} ColInfo;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *ptr = wb;

    if (nw < 20) { fprintf(stderr, "bad weights (too small)\n"); return 3; }
    int32_t magic, d, K, C, n_cols;
    memcpy(&magic, ptr, 4); ptr += 4;
    if (magic != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    memcpy(&d, ptr, 4); ptr += 4;
    memcpy(&K, ptr, 4); ptr += 4;
    memcpy(&C, ptr, 4); ptr += 4;
    memcpy(&n_cols, ptr, 4); ptr += 4;
    if (C != 2 || K != 4 || n_cols != 14 || d <= 0) { fprintf(stderr, "bad header values\n"); return 3; }

    ColInfo cols[14];
    for (int j = 0; j < n_cols; j++) {
        int8_t type = *ptr++; 
        cols[j].type = type;
        if (type == 0) { // ordinal
            memcpy(&cols[j].mu, ptr, 4); ptr += 4;
            memcpy(&cols[j].sd, ptr, 4); ptr += 4;
            memcpy(&cols[j].feat_idx, ptr, 4); ptr += 4;
        } else { // categorical
            memcpy(&cols[j].num_cat, ptr, 4); ptr += 4;
            memcpy(&cols[j].feat_offset, ptr, 4); ptr += 4;
            memcpy(cols[j].mapping, ptr, 256); ptr += 256;
        }
    }

    float w0;
    memcpy(&w0, ptr, 4); ptr += 4;
    float *w = (float *)ptr; ptr += d * 4;
    float *V = (float *)ptr; ptr += d * K * 4;

    // Validate total size
    if (ptr - wb != nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = (float *)malloc(d * sizeof(float));
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }

        memset(feat, 0, d * sizeof(float));
        for (int j = 0; j < 14; j++) {
            if (cols[j].type == 0) {
                float val = (xi[j] - cols[j].mu) / cols[j].sd;
                feat[cols[j].feat_idx] = val;
            } else {
                int8_t local = cols[j].mapping[xi[j]];
                if (local >= 0) {
                    feat[cols[j].feat_offset + local] = 1.0f;
                }
            }
        }

        float linear = w0;
        for (int idx = 0; idx < d; idx++) linear += w[idx] * feat[idx];

        float interaction = 0.0f;
        for (int f = 0; f < K; f++) {
            float sum_vx = 0.0f, sum_v2_x2 = 0.0f;
            for (int idx = 0; idx < d; idx++) {
                float v = V[idx * K + f];
                float xi_val = feat[idx];
                sum_vx += v * xi_val;
                sum_v2_x2 += v * v * xi_val * xi_val;
            }
            interaction += 0.5f * (sum_vx * sum_vx - sum_v2_x2);
        }

        float logit = linear + interaction;
        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }

    free(wb);
    if (x) free(x);
    free(feat);
    return 0;
}
```