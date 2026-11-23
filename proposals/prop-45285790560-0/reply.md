```hypothesis
Family: label-fm (Factorization Machine). Captures pairwise feature interactions via low-rank latent vectors (k=8), which linear models miss. Adult has known interactions (e.g., education×occupation). With ~100 expanded features, adds only ~800 params; inference cost similar to linear. If dev accuracy ≤ 0.8524 (label-linear), interactions don't help or optimization fails.
```
```python
"""Factorization Machine for tabular label tasks: ordinal z-scored, categorical one-hot over training values.
Latent dim k=8. Adam lr=0.01, L2=1e-4, minibatch=256, epochs=50. Deterministic seed=1.
Exports: magic 0x464D4C31, d, k, n_features, C; per-column encoding; float32 w0, w[n_features], V[n_features*k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 50


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
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # Build feature encoding
    col_info = []  # list of dicts per column
    feature_index = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            col_info.append({"type": "ordinal", "mean": mu, "std": sd, "fidx": feature_index})
            feature_index += 1
        else:
            vals = np.unique(X[:, j])
            # map value -> feature index, with extra for unseen
            mapping = np.full(256, feature_index + len(vals), dtype=np.int16)  # unseen index at end
            mapping[vals] = np.arange(feature_index, feature_index + len(vals))
            col_info.append({"type": "categorical", "mapping": mapping, "fidx": feature_index, "ncat": len(vals) + 1})
            feature_index += len(vals) + 1

    n_features = feature_index

    # Build dense design matrix Z (n x n_features)
    Z = np.zeros((n, n_features), dtype=np.float32)
    for j, info in enumerate(col_info):
        if info["type"] == "ordinal":
            idx = info["fidx"]
            Z[:, idx] = (X[:, j].astype(np.float32) - info["mean"]) / info["std"]
        else:
            idxs = info["mapping"][X[:, j]]
            Z[np.arange(n), idxs] = 1.0

    # Convert labels to float32 {0,1}
    yf = y.astype(np.float32)

    # Initialize parameters
    rng = np.random.default_rng(SEED)
    w0 = np.float32(0.0)
    w = np.zeros(n_features, dtype=np.float32)
    V = (rng.normal(0, 0.01, (n_features, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Training loop
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = Z[idx]          # (B, n_features)
            yy = yf[idx]        # (B,)
            B = x.shape[0]

            # Forward
            linear = w0 + x @ w                      # (B,)
            xV = x @ V                               # (B, K)
            xV2 = xV * xV                            # (B, K)
            x2 = x * x                               # (B, n_features)
            V2 = V * V                               # (n_features, K)
            sum_V2_x2 = x2 @ V2                      # (B, K)
            interaction = 0.5 * np.sum(xV2 - sum_V2_x2, axis=1)  # (B,)
            score = linear + interaction             # (B,)
            p = 1.0 / (1.0 + np.exp(-score))         # (B,)

            # Gradient d_score = (p - y) / B
            d_score = (p - yy) / B                   # (B,)

            # dw0
            dw0 = float(d_score.sum())
            # dw
            dw = x.T @ d_score + L2 * w
            # dV
            # term1 = X^T (d_score * xV)  -> (n_features, K)
            term1 = x.T @ (d_score[:, None] * xV)
            # term2 = V * ( (X^2)^T d_score )[:, None]
            sum_d_x2 = x2.T @ d_score                # (n_features,)
            term2 = V * sum_d_x2[:, None]
            dV = term1 - term2 + L2 * V

            # Adam updates
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * dw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (dw0 * dw0)
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (math.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * (dw * dw)
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * (dV * dV)
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Pack weights
    # Header: magic, d, K, n_features, C
    out = struct.pack("<5i", 0x464D4C31, d, K, n_features, C)
    # Per-column info
    for info in col_info:
        if info["type"] == "ordinal":
            out += struct.pack("<Bff", 0, info["mean"], info["std"])
        else:
            mapping = info["mapping"]
            ncat = info["ncat"]
            out += struct.pack("<Bi", 1, ncat)
            out += mapping.astype("<h").tobytes()
    # Parameters
    out += struct.pack("<f", w0)
    out += w.astype("<f").tobytes()
    out += V.astype("<f").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (raw columns). Output: 1 byte class (0/1). */
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
    uint8_t type;          // 0=ordinal, 1=categorical
    float mean, std;       // ordinal
    int16_t *map;          // categorical: 256 entries -> feature index
} ColInfo;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    const int32_t *hd = (const int32_t *)wb;
    if (nw < 20 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], n_features = hd[3], C = hd[4];
    if (C != 2 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }

    const unsigned char *ptr = wb + 20;
    ColInfo cols[14];
    for (int j = 0; j < d; j++) {
        uint8_t type = *ptr++;
        if (type == 0) { // ordinal
            cols[j].type = 0;
            cols[j].mean = *(const float *)ptr; ptr += 4;
            cols[j].std  = *(const float *)ptr; ptr += 4;
            cols[j].map = NULL;
        } else { // categorical
            cols[j].type = 1;
            int32_t ncat = *(const int32_t *)ptr; ptr += 4;
            cols[j].map = (int16_t *)ptr;
            ptr += 256 * sizeof(int16_t);
        }
    }
    // Expected size: header(20) + cols + 4 + 4*n_features + 4*n_features*K
    long expected = 20;
    for (int j = 0; j < d; j++) {
        if (cols[j].type == 0) expected += 1 + 4 + 4;
        else expected += 1 + 4 + 256*2;
    }
    expected += 4 + 4L * n_features + 4L * n_features * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float w0 = *(const float *)ptr; ptr += 4;
    const float *w = (const float *)ptr; ptr += 4L * n_features;
    const float *V = (const float *)ptr;

    float *xV = (float *)malloc(K * sizeof(float));
    float *sum_sq = (float *)malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Compute linear term and xV
        float linear = w0;
        for (int f = 0; f < K; f++) { xV[f] = 0.0f; sum_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            if (cols[j].type == 0) {
                float val = (xi[j] - cols[j].mean) / cols[j].std;
                int idx = j; // ordinal features are first d_ordinal? Wait, feature indices are sequential per column.
                // Actually, in train.py we assigned feature indices sequentially across columns.
                // But we didn't store the feature index per column in the weights! We only stored mean/std.
                // Need to store feature index for each column.
                // Fix: In train.py, store fidx for each column. In infer.c, read it.
                // Let's adjust: in train.py, for ordinal we stored only mean,std. We need fidx.
                // Since we cannot change train.py now, we must infer fidx from order.
                // In train.py, ordinal columns get one feature each in order of appearance. Categorical get blocks.
                // But we don't know which columns are ordinal/categorical without the card? We have col_info types.
                // We can compute fidx on the fly in infer.c by iterating columns in same order as train.py.
                // Train.py iterates j=0..13, assigns fidx sequentially.
                // So we can do the same here: keep a running feature index.
            }
        }
        // We need to recompute feature indices in infer.c exactly as train.py.
        // Let's restructure: in infer.c, we'll parse columns and compute fidx on the fly.
        // But we already read col_info. We can compute fidx by simulating the assignment.
        // Better: store fidx in weights. Since we can't change train.py, we must assume the assignment is deterministic.
        // In train.py, feature_index starts at 0. For each column in order:
        //   if ordinal: fidx = feature_index; feature_index += 1
        //   if categorical: fidx = feature_index; feature_index += ncat
        // So we can replicate this in infer.c using the col_info we parsed (which has ncat for categorical).
        // But we need ncat for categorical. We have it in col_info? We stored ncat in train.py? Yes, we wrote ncat.
        // In train.py: out += struct.pack("<Bi", 1, ncat) then mapping.
        // In infer.c we read ncat but didn't store it. Let's fix infer.c to store ncat.
        // We'll redo the parsing loop.
    }
    // The above is incomplete. Let's rewrite infer.c properly with correct parsing.

    free(xV); free(sum_sq);
    return 0;
}
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (raw columns). Output: 1 byte class (0/1). */
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
    uint8_t type;          // 0=ordinal, 1=categorical
    float mean, std;       // ordinal
    int32_t ncat;          // categorical: number of categories (including unseen)
    int16_t *map;          // categorical: 256 entries -> feature index
} ColInfo;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    const int32_t *hd = (const int32_t *)wb;
    if (nw < 20 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], n_features = hd[3], C = hd[4];
    if (C != 2 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }

    const unsigned char *ptr = wb + 20;
    ColInfo cols[14];
    for (int j = 0; j < d; j++) {
        uint8_t type = *ptr++;
        cols[j].type = type;
        if (type == 0) { // ordinal
            cols[j].mean = *(const float *)ptr; ptr += 4;
            cols[j].std  = *(const float *)ptr; ptr += 4;
            cols[j].ncat = 0;
            cols[j].map = NULL;
        } else { // categorical
            cols[j].ncat = *(const int32_t *)ptr; ptr += 4;
            cols[j].map = (int16_t *)ptr;
            ptr += 256 * sizeof(int16_t);
        }
    }

    // Expected size check
    long expected = 20;
    for (int j = 0; j < d; j++) {
        if (cols[j].type == 0) expected += 1 + 4 + 4;
        else expected += 1 + 4 + 256*2;
    }
    expected += 4 + 4L * n_features + 4L * n_features * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float w0 = *(const float *)ptr; ptr += 4;
    const float *w = (const float *)ptr; ptr += 4L * n_features;
    const float *V = (const float *)ptr;

    float *xV = (float *)malloc(K * sizeof(float));
    float *sum_sq = (float *)malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Compute feature indices on the fly (same order as train.py)
        int fidx = 0;
        float linear = w0;
        for (int f = 0; f < K; f++) { xV[f] = 0.0f; sum_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            if (cols[j].type == 0) {
                float val = (xi[j] - cols[j].mean) / cols[j].std;
                linear += w[fidx] * val;
                const float *Vrow = V + (long)fidx * K;
                for (int f = 0; f < K; f++) {
                    float vxf = Vrow[f] * val;
                    xV[f] += vxf;
                    sum_sq[f] += vxf * vxf;
                }
                fidx++;
            } else {
                int idx = cols[j].map[xi[j]];
                linear += w[idx];
                const float *Vrow = V + (long)idx * K;
                for (int f = 0; f < K; f++) {
                    float v = Vrow[f];
                    xV[f] += v;
                    sum_sq[f] += v * v;
                }
                fidx += cols[j].ncat;
            }
        }

        float interaction = 0.0f;
        for (int f = 0; f < K; f++) {
            interaction += xV[f] * xV[f] - sum_sq[f];
        }
        interaction *= 0.5f;
        float score = linear + interaction;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(xV); free(sum_sq);
    return 0;
}
```