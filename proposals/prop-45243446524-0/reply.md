```hypothesis
Family: label-poly2 (logistic regression with pairwise interactions on ordinal features).
Why: The adult dataset has 6 ordinal (quantile-binned) features; interactions capture non-linearities missed by linear models, while adding only 15 extra parameters per class. Training remains a single convex problem (fast, deterministic). Inference computes 6 z-scores and 15 products, still O(d) and very fast.
Failure mode: If interactions don't improve over linear (dev accuracy ≤ 0.8524) or inference cost exceeds label-linear by >2x.
```

```python
"""label-poly2: logistic regression with pairwise ordinal interactions.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x504F4C32, d, H=0, C, n_ord, ord_idx[n_ord];
per column: type (0 ord, 1 cat), if ord: mean, std, w_lin[C]; if cat: table[256*C];
then w_int[n_int*C] (n_int=n_ord*(n_ord-1)/2), bias[C]."""
import json, math, struct, sys

import numpy as np

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
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    # Identify ordinal columns
    ordinal_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    n_ord = len(ordinal_idx)
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]

    # Compute ordinal z-scoring parameters
    ord_mean = np.zeros(n_ord, np.float64)
    ord_std = np.zeros(n_ord, np.float64)
    for i, j in enumerate(ordinal_idx):
        v = X[:, j].astype(np.float64)
        ord_mean[i] = v.mean()
        ord_std[i] = v.std() or 1.0

    # Build base feature matrix Z_base and interaction matrix Z_int
    # Base: ordinal -> 1 column each; categorical -> one-hot per seen value
    base_cols = []
    cat_maps = []  # for each cat column: (vals, weight_segment_indices)
    col_to_base_start = {}
    base_ptr = 0
    for j in range(d):
        if j in ordinal_idx:
            col_to_base_start[j] = base_ptr
            base_ptr += 1
        else:
            vals = np.unique(X[:, j])
            cat_maps.append((j, vals))
            col_to_base_start[j] = base_ptr
            base_ptr += len(vals)
    k_base = base_ptr

    # Interaction columns: pairs of ordinal columns
    n_int = n_ord * (n_ord - 1) // 2
    k_all = k_base + n_int

    Z_all = np.zeros((X.shape[0], k_all), dtype=np.float32)
    # Fill base features
    for i, j in enumerate(ordinal_idx):
        z = (X[:, j].astype(np.float64) - ord_mean[i]) / ord_std[i]
        Z_all[:, col_to_base_start[j]] = z.astype(np.float32)
    for j, vals in cat_maps:
        start = col_to_base_start[j]
        for vi, v in enumerate(vals):
            Z_all[X[:, j] == v, start + vi] = 1.0
    # Fill interactions
    int_ptr = k_base
    for i in range(n_ord):
        zi = Z_all[:, col_to_base_start[ordinal_idx[i]]]
        for j in range(i + 1, n_ord):
            zj = Z_all[:, col_to_base_start[ordinal_idx[j]]]
            Z_all[:, int_ptr] = zi * zj
            int_ptr += 1

    n, k = Z_all.shape
    assert k == k_all

    # Train logistic regression with Adam
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_all[idx], y[idx]
            scores = z @ W + b
            g = softmax_grad(scores, yy)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Extract weights for export
    W_base = W[:k_base]
    W_int = W[k_base:]
    bias = b

    # Prepare per-column export data
    col_type = np.zeros(d, dtype=np.int32)
    ord_data = {}  # j -> (mean, std, w_lin[C])
    cat_tables = {}  # j -> flat array 256*C
    for j in range(d):
        if j in ordinal_idx:
            col_type[j] = 0
            i = ordinal_idx.index(j)
            w_lin = W_base[col_to_base_start[j]]  # shape (C,)
            ord_data[j] = (float(ord_mean[i]), float(ord_std[i]), w_lin.astype(np.float32))
        else:
            col_type[j] = 1
            start = col_to_base_start[j]
            vals = np.unique(X[:, j])
            table = np.zeros((256, C), dtype=np.float32)
            for vi, v in enumerate(vals):
                table[v] = W_base[start + vi]
            cat_tables[j] = table.astype(np.float32).ravel()

    # Interaction weights in order of ordinal_idx pairs
    w_int_flat = W_int.astype(np.float32).ravel()

    # Write weights.bin
    magic = 0x504F4C32  # "POL2"
    n_int = n_ord * (n_ord - 1) // 2
    with open(sys.argv[4], "wb") as f:
        # Header
        f.write(struct.pack("<5i", magic, d, 0, C, n_ord))
        f.write(struct.pack(f"<{n_ord}i", *ordinal_idx))
        # Per column
        for j in range(d):
            if col_type[j] == 0:
                mean, std, w_lin = ord_data[j]
                f.write(struct.pack("<i", 0))
                f.write(struct.pack("<ff", mean, std))
                f.write(struct.pack(f"<{C}f", *w_lin))
            else:
                f.write(struct.pack("<i", 1))
                f.write(struct.pack(f"<{256*C}f", *cat_tables[j]))
        # Interactions
        f.write(struct.pack(f"<{n_int*C}f", *w_int_flat))
        # Bias
        f.write(struct.pack(f"<{C}f", *bias.astype(np.float32)))


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index per record.
 * Weights format: magic 0x504F4C32, d, H=0, C, n_ord, ord_idx[n_ord];
 * per column: type (0 ord, 1 cat); if ord: mean, std, w_lin[C]; if cat: table[256*C];
 * then w_int[n_int*C] (n_int=n_ord*(n_ord-1)/2), bias[C]. */
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
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    const uint32_t *u32 = (const uint32_t *)wb;
    const float *f32 = (const float *)wb;
    if (nw < 20 || u32[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = (int)u32[1];
    int H = (int)u32[2];
    int C = (int)u32[3];
    int n_ord = (int)u32[4];
    if (H != 0 || C < 1 || C > 256 || n_ord < 0 || n_ord > d) { fprintf(stderr, "bad header\n"); return 3; }

    const int32_t *ord_idx = (const int32_t *)(u32 + 5);
    size_t offset = 5 + n_ord;  // in uint32_t units

    // Allocate per-column data
    float *ord_mean = malloc(n_ord * sizeof(float));
    float *ord_std = malloc(n_ord * sizeof(float));
    float *ord_wlin = malloc(n_ord * C * sizeof(float));
    float **cat_table = malloc(d * sizeof(float*));
    int *col_type = malloc(d * sizeof(int));
    memset(cat_table, 0, d * sizeof(float*));

    // Parse columns
    for (int j = 0; j < d; j++) {
        if (offset >= (size_t)nw / 4) { fprintf(stderr, "truncated weights\n"); return 3; }
        int type = (int)u32[offset++];
        col_type[j] = type;
        if (type == 0) { // ordinal
            // Find which ordinal index this column is
            int oi = -1;
            for (int i = 0; i < n_ord; i++) if (ord_idx[i] == j) { oi = i; break; }
            if (oi == -1) { fprintf(stderr, "ordinal column not in ord_idx\n"); return 3; }
            if (offset + 2 > (size_t)nw / 4) { fprintf(stderr, "truncated weights\n"); return 3; }
            ord_mean[oi] = f32[offset++];
            ord_std[oi] = f32[offset++];
            if (offset + C > (size_t)nw / 4) { fprintf(stderr, "truncated weights\n"); return 3; }
            for (int c = 0; c < C; c++) ord_wlin[oi * C + c] = f32[offset++];
        } else if (type == 1) { // categorical
            size_t need = 256 * C;
            if (offset + need > (size_t)nw / 4) { fprintf(stderr, "truncated weights\n"); return 3; }
            cat_table[j] = malloc(need * sizeof(float));
            for (size_t i = 0; i < need; i++) cat_table[j][i] = f32[offset++];
        } else {
            fprintf(stderr, "unknown column type %d\n", type); return 3;
        }
    }

    // Interactions
    int n_int = n_ord * (n_ord - 1) / 2;
    float *w_int = malloc(n_int * C * sizeof(float));
    if (offset + n_int * C > (size_t)nw / 4) { fprintf(stderr, "truncated weights\n"); return 3; }
    for (int i = 0; i < n_int * C; i++) w_int[i] = f32[offset++];

    // Bias
    float *bias = malloc(C * sizeof(float));
    if (offset + C > (size_t)nw / 4) { fprintf(stderr, "truncated weights\n"); return 3; }
    for (int c = 0; c < C; c++) bias[c] = f32[offset++];

    // Validate total size
    if (offset * 4 != (size_t)nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    // Inference
    float *acc = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize with bias
        for (int c = 0; c < C; c++) acc[c] = bias[c];

        // Ordinal and categorical contributions
        for (int j = 0; j < d; j++) {
            if (col_type[j] == 0) {
                // find ordinal index
                int oi = -1;
                for (int i = 0; i < n_ord; i++) if (ord_idx[i] == j) { oi = i; break; }
                float z = (xi[j] - ord_mean[oi]) / ord_std[oi];
                for (int c = 0; c < C; c++) acc[c] += z * ord_wlin[oi * C + c];
            } else {
                const float *tbl = cat_table[j];
                if (tbl) {
                    const float *row = tbl + xi[j] * C;
                    for (int c = 0; c < C; c++) acc[c] += row[c];
                }
            }
        }

        // Interaction contributions
        int pair = 0;
        for (int ii = 0; ii < n_ord; ii++) {
            int coli = ord_idx[ii];
            float zi = (xi[coli] - ord_mean[ii]) / ord_std[ii];
            for (int jj = ii + 1; jj < n_ord; jj++) {
                int colj = ord_idx[jj];
                float zj = (xi[colj] - ord_mean[jj]) / ord_std[jj];
                float prod = zi * zj;
                const float *w = w_int + pair * C;
                for (int c = 0; c < C; c++) acc[c] += prod * w[c];
                pair++;
            }
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```