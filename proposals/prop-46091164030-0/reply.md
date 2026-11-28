```hypothesis
Propose label-fm: a binary Factorization Machine with embedding size E=8. It models pairwise feature interactions via low-rank embeddings, which are crucial for this categorical-heavy dataset (8 categorical, 6 ordinal). The FM adds only ~900 parameters over linear but captures interactions efficiently. If dev accuracy does not exceed label-linear (0.8524), it indicates interactions are not beneficial or E=8 is too small.
```

```python
"""Factorization Machine for tabular label tasks (binary classification).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x464D3130, d, E, then per-feature params, then global bias."""
import json, math, struct, sys

import numpy as np

E = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)  # 0 or 1
    n, d = X.shape
    types = card["input"]["types"]
    C = card["classes"]
    assert C == 2, "FM only for binary"

    rng = np.random.default_rng(SEED)

    # Preprocess: build mappings and normalization
    ordinal_idx = []
    cat_idx = []
    cat_maps = []          # list of dict: byte value -> category index (0..V)
    cat_sizes = []         # V+1 (including unseen)
    ord_mean = []
    ord_std = []

    for j, t in enumerate(types):
        if t == "ordinal":
            ordinal_idx.append(j)
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ord_mean.append(mu)
            ord_std.append(sd)
        else:
            cat_idx.append(j)
            vals = np.unique(X[:, j])
            mapping = {int(v): i for i, v in enumerate(vals)}
            # unseen (255) gets index len(vals)
            cat_maps.append(mapping)
            cat_sizes.append(len(vals) + 1)

    n_ord = len(ordinal_idx)
    n_cat = len(cat_idx)

    # Parameters
    w0 = 0.0
    # Linear weights: for ordinal: one per feature; for categorical: one per category (including unseen)
    w_ord = np.zeros(n_ord, dtype=np.float32)
    w_cat = [np.zeros(sz, dtype=np.float32) for sz in cat_sizes]
    # Embeddings: for ordinal: one vector per feature; for categorical: one vector per category
    v_ord = rng.normal(0, 0.01, (n_ord, E)).astype(np.float32)
    v_cat = [rng.normal(0, 0.01, (sz, E)).astype(np.float32) for sz in cat_sizes]

    # Adam state
    params = [w0] + [w_ord] + w_cat + [v_ord] + v_cat
    # We'll keep state as list of (m, v) for each parameter array
    state = []
    for p in params:
        if np.isscalar(p):
            state.append((0.0, 0.0))
        else:
            state.append((np.zeros_like(p), np.zeros_like(p)))

    # Precompute normalized ordinal values and categorical indices for all samples
    X_ord = np.zeros((n, n_ord), dtype=np.float32)
    for i, j in enumerate(ordinal_idx):
        X_ord[:, i] = (X[:, j].astype(np.float32) - ord_mean[i]) / ord_std[i]

    X_cat = np.zeros((n, n_cat), dtype=np.int32)
    for i, j in enumerate(cat_idx):
        mapping = cat_maps[i]
        V = cat_sizes[i] - 1
        # map each byte value to index; unseen (255) -> V
        col = X[:, j]
        mapped = np.array([mapping.get(int(v), V) for v in col], dtype=np.int32)
        X_cat[:, i] = mapped

    def forward_batch(x_ord, x_cat):
        # x_ord: (batch, n_ord), x_cat: (batch, n_cat) int indices
        batch = x_ord.shape[0]
        # Linear part
        lin = w0 + x_ord @ w_ord
        for i in range(n_cat):
            lin += w_cat[i][x_cat[:, i]]
        # Interaction part
        # Compute sum_f v_f * x_f for each embedding dimension
        # For ordinal: v_ord[i] * x_ord[:,i]  -> (batch, E)
        # For categorical: v_cat[i][x_cat[:,i]] -> (batch, E)
        sum_vx = np.zeros((batch, E), dtype=np.float32)
        sum_vx_sq = np.zeros((batch, E), dtype=np.float32)
        for i in range(n_ord):
            vx = x_ord[:, i:i+1] * v_ord[i]  # (batch, E)
            sum_vx += vx
            sum_vx_sq += vx * vx
        for i in range(n_cat):
            vx = v_cat[i][x_cat[:, i]]  # (batch, E)
            sum_vx += vx
            sum_vx_sq += vx * vx
        inter = 0.5 * np.sum(sum_vx * sum_vx - sum_vx_sq, axis=1)
        score = lin + inter
        return score

    t = 0
    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_ord_b = X_ord[idx]
            x_cat_b = X_cat[idx]
            y_b = y[idx]

            score = forward_batch(x_ord_b, x_cat_b)
            p = sigmoid(score)
            grad_score = p - y_b  # (batch,)

            # Gradients
            g_w0 = grad_score.mean()
            g_w_ord = np.zeros_like(w_ord)
            g_w_cat = [np.zeros_like(w) for w in w_cat]
            g_v_ord = np.zeros_like(v_ord)
            g_v_cat = [np.zeros_like(v) for v in v_cat]

            # Linear gradients
            g_w_ord = (x_ord_b.T @ grad_score) / BATCH
            for i in range(n_cat):
                # accumulate gradient for each category
                np.add.at(g_w_cat[i], x_cat_b[:, i], grad_score)
            g_w_cat = [g / BATCH for g in g_w_cat]

            # Interaction gradients
            # Recompute sum_vx for this batch
            sum_vx = np.zeros((BATCH, E), dtype=np.float32)
            for i in range(n_ord):
                sum_vx += x_ord_b[:, i:i+1] * v_ord[i]
            for i in range(n_cat):
                sum_vx += v_cat[i][x_cat_b[:, i]]

            # dL/dv = grad_score * x * (sum_vx - v * x)
            for i in range(n_ord):
                x = x_ord_b[:, i:i+1]  # (batch, 1)
                factor = grad_score[:, None] * x * (sum_vx - v_ord[i] * x)  # (batch, E)
                g_v_ord[i] = factor.mean(axis=0)
            for i in range(n_cat):
                x = (x_cat_b[:, i] == np.arange(cat_sizes[i])[:, None]).T  # (batch, V) one-hot? Too slow.
                # Instead, loop over batch
                for b in range(BATCH):
                    cat = x_cat_b[b, i]
                    factor = grad_score[b] * (sum_vx[b] - v_cat[i][cat])
                    g_v_cat[i][cat] += factor
            g_v_cat = [g / BATCH for g in g_v_cat]

            # Add L2 regularization
            g_w0 += L2 * w0
            g_w_ord += L2 * w_ord
            for i in range(n_cat):
                g_w_cat[i] += L2 * w_cat[i]
            g_v_ord += L2 * v_ord
            for i in range(n_cat):
                g_v_cat[i] += L2 * v_cat[i]

            # Adam update
            grads = [g_w0] + [g_w_ord] + g_w_cat + [g_v_ord] + g_v_cat
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                if np.isscalar(p):
                    m = 0.9 * m + 0.1 * gr
                    v = 0.999 * v + 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    params[i] = p - LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                    state[i] = (m, v)
                else:
                    m = 0.9 * m + 0.1 * gr
                    v = 0.999 * v + 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            # Unpack updated params
            w0 = params[0]
            w_ord = params[1]
            w_cat = params[2:2+n_cat]
            v_ord = params[2+n_cat]
            v_cat = params[3+n_cat:]

    # Write weights.bin
    # Layout:
    # magic (uint32), d (uint32), E (uint32), n_ord (uint32), n_cat (uint32)
    # For each feature in original order (0..d-1):
    #   type (uint8): 0=ordinal, 1=categorical
    #   if ordinal:
    #       mean (float32), std (float32), w_linear (float32), v_embedding[E] (float32)
    #   if categorical:
    #       V (uint16)  // number of training categories (excluding unseen)
    #       mapping[256] (uint16)  // byte value -> category index (0..V, where V is unseen)
    #       for each category 0..V:
    #           w_linear (float32), v_embedding[E] (float32)
    # w0 (float32)

    out = bytearray()
    out += struct.pack("<5I", 0x464D3130, d, E, n_ord, n_cat)

    ord_ptr = 0
    cat_ptr = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            out += b'\x00'  # type ordinal
            out += struct.pack("<f", ord_mean[ord_ptr])
            out += struct.pack("<f", ord_std[ord_ptr])
            out += struct.pack("<f", w_ord[ord_ptr])
            out += v_ord[ord_ptr].astype("<f4").tobytes()
            ord_ptr += 1
        else:
            out += b'\x01'  # type categorical
            V = cat_sizes[cat_ptr] - 1
            out += struct.pack("<H", V)
            # mapping for 256 values
            mapping = cat_maps[cat_ptr]
            map_arr = np.full(256, V, dtype=np.uint16)
            for val, idx in mapping.items():
                map_arr[val] = idx
            out += map_arr.astype("<H").tobytes()
            # weights and embeddings for each category (0..V)
            for c in range(V + 1):
                out += struct.pack("<f", w_cat[cat_ptr][c])
                out += v_cat[cat_ptr][c].astype("<f4").tobytes()
            cat_ptr += 1

    out += struct.pack("<f", w0)

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for binary label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1). */
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

    const uint32_t *hdr = (const uint32_t *)wb;
    if (hdr[0] != 0x464D3130) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1];
    int E = hdr[2];
    int n_ord = hdr[3];
    int n_cat = hdr[4];
    if (d != 14) { fprintf(stderr, "unexpected d=%d\n", d); return 3; }

    const unsigned char *p = wb + 20;
    const unsigned char *end = wb + nw;

    // Storage for ordinal features
    float *ord_mean = malloc(n_ord * sizeof(float));
    float *ord_std = malloc(n_ord * sizeof(float));
    float *w_ord = malloc(n_ord * sizeof(float));
    float *v_ord = malloc(n_ord * E * sizeof(float));

    // Storage for categorical features
    int *cat_V = malloc(n_cat * sizeof(int));
    uint16_t **cat_map = malloc(n_cat * sizeof(uint16_t*));
    float **w_cat = malloc(n_cat * sizeof(float*));
    float **v_cat = malloc(n_cat * sizeof(float*));

    int ord_idx = 0, cat_idx = 0;
    for (int j = 0; j < d; j++) {
        if (p >= end) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint8_t type = *p++;
        if (type == 0) { // ordinal
            if (p + 4*3 + E*4 > end) { fprintf(stderr, "truncated ordinal\n"); return 3; }
            ord_mean[ord_idx] = *(const float*)p; p += 4;
            ord_std[ord_idx] = *(const float*)p; p += 4;
            w_ord[ord_idx] = *(const float*)p; p += 4;
            memcpy(&v_ord[ord_idx * E], p, E * 4); p += E * 4;
            ord_idx++;
        } else if (type == 1) { // categorical
            if (p + 2 > end) { fprintf(stderr, "truncated cat V\n"); return 3; }
            uint16_t V = *(const uint16_t*)p; p += 2;
            cat_V[cat_idx] = V;
            if (p + 256*2 > end) { fprintf(stderr, "truncated cat map\n"); return 3; }
            cat_map[cat_idx] = malloc(256 * sizeof(uint16_t));
            memcpy(cat_map[cat_idx], p, 256 * 2); p += 256 * 2;
            int total_cats = V + 1;
            w_cat[cat_idx] = malloc(total_cats * sizeof(float));
            v_cat[cat_idx] = malloc(total_cats * E * sizeof(float));
            for (int c = 0; c < total_cats; c++) {
                if (p + 4 + E*4 > end) { fprintf(stderr, "truncated cat params\n"); return 3; }
                w_cat[cat_idx][c] = *(const float*)p; p += 4;
                memcpy(&v_cat[cat_idx][c * E], p, E * 4); p += E * 4;
            }
            cat_idx++;
        } else {
            fprintf(stderr, "unknown feature type %d\n", type); return 3;
        }
    }
    if (p + 4 > end) { fprintf(stderr, "truncated w0\n"); return 3; }
    float w0 = *(const float*)p; p += 4;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_vx = malloc(E * sizeof(float));
    float *sum_vx_sq = malloc(E * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        float score = w0;
        // Ordinal
        for (int o = 0; o < n_ord; o++) {
            float xval = (xi[ord_idx] - ord_mean[o]) / ord_std[o]; // wait, we need original feature index
            // Actually we need to know which original column this ordinal corresponds to.
            // We lost the mapping from ordinal index to column index. We must store column indices.
            // Fix: In train.py, we wrote features in original order, so we can just iterate columns in order.
            // But here we are iterating features in order of storage, which is original order because we wrote them in original order.
            // However, we need to know the column index for each ordinal/categorical. Since we wrote them in original order,
            // we can just keep a running column index. Let's restructure: we'll parse weights while iterating columns.
            // But we already parsed all weights into arrays. We need to know which column each ordinal belongs to.
            // We'll change the storage: store column index for each ordinal and categorical.
            // However, we can avoid this by processing the input record in the same order as we wrote weights.
            // Since we wrote weights in original column order (0..d-1), we can just iterate columns 0..d-1 again,
            // but we need to know for each column whether it's ordinal or categorical and its parameters.
            // We didn't store column type per column in a way that we can iterate by column. We stored sequentially.
            // We'll redo: during weight loading, we should store parameters per column.
        }
        // This inference code is incomplete because we need to map column index to parameters.
        // Let's redesign the weight format to include column index or just store per column in order.
        // Since we wrote in column order, we can just keep arrays indexed by column.
        // We'll change the loading loop to store per column.
    }

    // Due to complexity, we'll provide a corrected infer.c that stores per-column parameters.
    // But the answer must be complete. We'll rewrite infer.c properly in the final block.
    return 0;
}
```

```c
/* Factorization Machine inference for binary label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1). */
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

    const uint32_t *hdr = (const uint32_t *)wb;
    if (hdr[0] != 0x464D3130) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1];
    int E = hdr[2];
    int n_ord = hdr[3];
    int n_cat = hdr[4];
    if (d != 14) { fprintf(stderr, "unexpected d=%d\n", d); return 3; }

    const unsigned char *p = wb + 20;
    const unsigned char *end = wb + nw;

    // Per-column parameters
    float *col_mean = malloc(d * sizeof(float));
    float *col_std = malloc(d * sizeof(float));
    float *col_w_lin = malloc(d * sizeof(float));
    float *col_v_emb = malloc(d * E * sizeof(float));
    uint16_t *col_map = malloc(d * 256 * sizeof(uint16_t)); // only for categorical
    int *col_V = malloc(d * sizeof(int)); // V for categorical, 0 for ordinal
    uint8_t *col_type = malloc(d * sizeof(uint8_t)); // 0 ordinal, 1 categorical

    for (int j = 0; j < d; j++) {
        if (p >= end) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint8_t type = *p++;
        col_type[j] = type;
        if (type == 0) { // ordinal
            if (p + 4*3 + E*4 > end) { fprintf(stderr, "truncated ordinal\n"); return 3; }
            col_mean[j] = *(const float*)p; p += 4;
            col_std[j] = *(const float*)p; p += 4;
            col_w_lin[j] = *(const float*)p; p += 4;
            memcpy(&col_v_emb[j * E], p, E * 4); p += E * 4;
            col_V[j] = 0;
        } else if (type == 1) { // categorical
            if (p + 2 > end) { fprintf(stderr, "truncated cat V\n"); return 3; }
            uint16_t V = *(const uint16_t*)p; p += 2;
            col_V[j] = V;
            if (p + 256*2 > end) { fprintf(stderr, "truncated cat map\n"); return 3; }
            memcpy(&col_map[j * 256], p, 256 * 2); p += 256 * 2;
            int total_cats = V + 1;
            // We need to store w_lin and v_emb for each category.
            // But we have only one w_lin and one v_emb per column in the above arrays.
            // We need separate storage for categorical: w_cat[j][c], v_cat[j][c][E].
            // Let's allocate dynamically.
        } else {
            fprintf(stderr, "unknown feature type %d\n", type); return 3;
        }
    }
    // The above approach is flawed because categorical features have multiple categories.
    // We need to store per-category parameters. Let's allocate arrays of pointers.
    // We'll redo the loading with proper dynamic allocation.

    free(col_mean); free(col_std); free(col_w_lin); free(col_v_emb);
    free(col_map); free(col_V); free(col_type);

    // Proper loading:
    p = wb + 20;
    float **w_cat = malloc(d * sizeof(float*));
    float **v_cat = malloc(d * sizeof(float*));
    uint16_t **cat_map = malloc(d * sizeof(uint16_t*));
    int *cat_V = malloc(d * sizeof(int));
    uint8_t *col_type2 = malloc(d * sizeof(uint8_t));
    float *ord_mean = malloc(d * sizeof(float));
    float *ord_std = malloc(d * sizeof(float));
    float *ord_w_lin = malloc(d * sizeof(float));
    float *ord_v_emb = malloc(d * E * sizeof(float));

    for (int j = 0; j < d; j++) {
        if (p >= end) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint8_t type = *p++;
        col_type2[j] = type;
        if (type == 0) { // ordinal
            if (p + 4*3 + E*4 > end) { fprintf(stderr, "truncated ordinal\n"); return 3; }
            ord_mean[j] = *(const float*)p; p += 4;
            ord_std[j] = *(const float*)p; p += 4;
            ord_w_lin[j] = *(const float*)p; p += 4;
            memcpy(&ord_v_emb[j * E], p, E * 4); p += E * 4;
            cat_V[j] = 0;
            w_cat[j] = NULL;
            v_cat[j] = NULL;
            cat_map[j] = NULL;
        } else if (type == 1) { // categorical
            if (p + 2 > end) { fprintf(stderr, "truncated cat V\n"); return 3; }
            uint16_t V = *(const uint16_t*)p; p += 2;
            cat_V[j] = V;
            if (p + 256*2 > end) { fprintf(stderr, "truncated cat map\n"); return 3; }
            cat_map[j] = malloc(256 * sizeof(uint16_t));
            memcpy(cat_map[j], p, 256 * 2); p += 256 * 2;
            int total_cats = V + 1;
            w_cat[j] = malloc(total_cats * sizeof(float));
           