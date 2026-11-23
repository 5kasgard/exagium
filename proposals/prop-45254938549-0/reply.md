```hypothesis
Factorization Machine (label-fm) with low-rank embeddings (k=4) for pairwise feature interactions.
Captures non-linear feature combinations beyond linear model while keeping inference cost near linear (O(d*k)).
If dev accuracy does not exceed label-linear (0.8524), it indicates pairwise interactions alone are insufficient for this dataset.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-scored float, categorical as category index (unseen=card).
Model: logit = w0 + Σ w_i x_i + 0.5*( (Σ v_i x_i)^2 - Σ (v_i x_i)^2 )
Binary classification (C=2) with cross-entropy, Adam optimizer.
Weights exported as per-column lookup tables."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    n, d = X_bytes.shape
    types = card["input"]["types"]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)

    # Build encodings and convert training data
    ord_cols = []
    cat_cols = []
    ord_zscore = []      # list of (256,) float32
    cat_map = []         # list of (256,) uint16 mapping byte->index (unseen=card)
    cat_cards = []       # list of cardinalities (seen categories)
    X_ord_list = []
    X_cat_list = []

    for j, t in enumerate(types):
        col = X_bytes[:, j]
        if t == "ordinal":
            # z-score on byte values (0..254)
            vals = col.astype(np.float64)
            mu, sd = vals.mean(), vals.std() or 1.0
            table = ((np.arange(256) - mu) / sd).astype(np.float32)
            ord_zscore.append(table)
            X_ord_list.append(table[col])
            ord_cols.append(j)
        else:
            # categorical: map seen categories to 0..card-1, unseen (255) -> card
            seen = np.unique(col)
            # ensure 255 not in seen (reserved)
            seen = seen[seen != 255]
            card = len(seen)
            mapping = np.full(256, card, dtype=np.uint16)
            mapping[seen] = np.arange(card, dtype=np.uint16)
            cat_map.append(mapping)
            cat_cards.append(card)
            X_cat_list.append(mapping[col])
            cat_cols.append(j)

    X_ord = np.column_stack(X_ord_list) if X_ord_list else np.zeros((n, 0), dtype=np.float32)
    X_cat = np.column_stack(X_cat_list) if X_cat_list else np.zeros((n, 0), dtype=np.uint16)
    n_ord, n_cat = X_ord.shape[1], X_cat.shape[1]

    # Initialize parameters
    # w0
    p = y.mean()
    w0 = np.array([math.log(p / (1 - p))], dtype=np.float32)
    # ordinal: w_lin (n_ord), v_emb (n_ord, K), zscore tables already stored
    w_ord = np.zeros(n_ord, dtype=np.float32)
    v_ord = (rng.normal(0, 0.01, (n_ord, K))).astype(np.float32)
    # categorical: w_lin per category (n_cat, card+1), v_emb per category (n_cat, card+1, K)
    w_cat = [np.zeros(card + 1, dtype=np.float32) for card in cat_cards]
    v_cat = [(rng.normal(0, 0.01, (card + 1, K))).astype(np.float32) for card in cat_cards]

    # Adam state
    def adam_init(params):
        return [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    params = [w0, w_ord, v_ord] + w_cat + v_cat
    state = adam_init(params)
    t = 0

    # Precompute ordinal zscore for all samples (already in X_ord)
    # For categorical, we have indices in X_cat

    # Training loop
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            bs = len(idx)
            x_ord = X_ord[idx]          # (bs, n_ord)
            x_cat = X_cat[idx]          # (bs, n_cat)
            yy = y[idx]                 # (bs,)

            # Forward pass
            # linear part
            linear = w0[0] + np.sum(w_ord * x_ord, axis=1)
            for j in range(n_cat):
                linear += w_cat[j][x_cat[:, j]]

            # interaction part
            sum_vx = np.zeros((bs, K), dtype=np.float32)
            sum_vx_sq = np.zeros((bs, K), dtype=np.float32)
            # ordinal
            for j in range(n_ord):
                x = x_ord[:, j:j+1]          # (bs,1)
                v = v_ord[j:j+1]             # (1,K)
                sum_vx += x * v
                sum_vx_sq += (x * x) * (v * v)
            # categorical
            for j in range(n_cat):
                indices = x_cat[:, j]        # (bs,)
                v = v_cat[j][indices]        # (bs,K)
                sum_vx += v
                sum_vx_sq += v * v

            interaction = 0.5 * np.sum(sum_vx * sum_vx - sum_vx_sq, axis=1)
            logit = linear + interaction

            # Binary cross-entropy gradient: g = sigmoid(logit) - y
            sig = 1.0 / (1.0 + np.exp(-logit))
            g = (sig - yy).astype(np.float32)  # (bs,)

            # Gradients
            grad_w0 = np.array([g.mean()], dtype=np.float32)
            grad_w_ord = np.zeros_like(w_ord)
            grad_v_ord = np.zeros_like(v_ord)
            grad_w_cat = [np.zeros_like(w) for w in w_cat]
            grad_v_cat = [np.zeros_like(v) for v in v_cat]

            # dlogit/dw0 = 1
            # dlogit/dw_ord = x_ord
            grad_w_ord = (g[:, None] * x_ord).mean(axis=0)
            # dlogit/dv_ord = x_ord * (sum_vx - x_ord * v_ord)
            for j in range(n_ord):
                x = x_ord[:, j:j+1]          # (bs,1)
                v = v_ord[j:j+1]             # (1,K)
                # sum_vx - x*v is (bs,K)
                diff = sum_vx - x * v
                grad_v_ord[j] = (g[:, None] * x * diff).mean(axis=0)

            # categorical
            for j in range(n_cat):
                indices = x_cat[:, j]
                # dlogit/dw_cat = 1 for active category
                # accumulate gradient per category
                for c in range(cat_cards[j] + 1):
                    mask = (indices == c)
                    if np.any(mask):
                        grad_w_cat[j][c] = g[mask].mean()
                # dlogit/dv_cat = sum_vx - v_cat (since x=1)
                v = v_cat[j][indices]        # (bs,K)
                diff = sum_vx - v
                # accumulate per category
                for c in range(cat_cards[j] + 1):
                    mask = (indices == c)
                    if np.any(mask):
                        grad_v_cat[j][c] = (g[mask, None] * diff[mask]).mean(axis=0)

            # L2 regularization
            grad_w_ord += L2 * w_ord
            grad_v_ord += L2 * v_ord
            for j in range(n_cat):
                grad_w_cat[j] += L2 * w_cat[j]
                grad_v_cat[j] += L2 * v_cat[j]

            grads = [grad_w0, grad_w_ord, grad_v_ord] + grad_w_cat + grad_v_cat

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # Magic "FML1" = 0x314C4D46
    magic = 0x314C4D46
    header = struct.pack("<4I", magic, d, K, C)
    # Column types and cardinalities
    col_info = bytearray()
    for j, t in enumerate(types):
        if t == "ordinal":
            col_info.append(0)
            col_info.extend(struct.pack("<I", 0))
        else:
            # find cardinality for this categorical column
            cat_idx = cat_cols.index(j)
            card = cat_cards[cat_idx]
            col_info.append(1)
            col_info.extend(struct.pack("<I", card))

    # Parameters
    param_bytes = bytearray()
    param_bytes.extend(w0.astype("<f4").tobytes())
    param_bytes.extend(w_ord.astype("<f4").tobytes())
    param_bytes.extend(v_ord.astype("<f4").tobytes())
    for j in range(n_cat):
        param_bytes.extend(w_cat[j].astype("<f4").tobytes())
        param_bytes.extend(v_cat[j].astype("<f4").tobytes())
    # Ordinal zscore tables
    for table in ord_zscore:
        param_bytes.extend(table.astype("<f4").tobytes())
    # Categorical maps (uint16)
    for mapping in cat_map:
        param_bytes.extend(mapping.astype("<H").tobytes())

    out = header + bytes(col_info) + bytes(param_bytes)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes. Output: one byte class (0 or 1). */
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

    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x314C4D46) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || K > 64 || d <= 0 || d > 256) { fprintf(stderr, "bad header\n"); return 3; }

    unsigned char *ptr = wb + 16;
    long remaining = nw - 16;

    // Read column info
    uint8_t *col_type = malloc(d);
    uint32_t *col_card = malloc(d * sizeof(uint32_t));
    for (int j = 0; j < d; j++) {
        if (remaining < 5) { fprintf(stderr, "truncated col info\n"); return 3; }
        col_type[j] = ptr[0];
        col_card[j] = ptr[1] | ptr[2] << 8 | ptr[3] << 16 | ptr[4] << 24;
        ptr += 5; remaining -= 5;
    }

    // Pointers to parameter sections
    float *w0 = (float *)ptr; ptr += 4; remaining -= 4;
    float *w_ord = (float *)ptr; 
    // We'll compute sizes later, but first we need to know number of ordinal/categorical columns
    int n_ord = 0, n_cat = 0;
    for (int j = 0; j < d; j++) if (col_type[j] == 0) n_ord++; else n_cat++;
    ptr += n_ord * 4; remaining -= n_ord * 4;
    float *v_ord = (float *)ptr; ptr += n_ord * K * 4; remaining -= n_ord * K * 4;

    // Categorical linear weights and embeddings: variable per column
    float **w_cat = malloc(n_cat * sizeof(float *));
    float **v_cat = malloc(n_cat * sizeof(float *));
    int cat_idx = 0;
    for (int j = 0; j < d; j++) {
        if (col_type[j] == 1) {
            uint32_t card = col_card[j];
            if (remaining < (card + 1) * 4) { fprintf(stderr, "truncated w_cat\n"); return 3; }
            w_cat[cat_idx] = (float *)ptr; ptr += (card + 1) * 4; remaining -= (card + 1) * 4;
            if (remaining < (card + 1) * K * 4) { fprintf(stderr, "truncated v_cat\n"); return 3; }
            v_cat[cat_idx] = (float *)ptr; ptr += (card + 1) * K * 4; remaining -= (card + 1) * K * 4;
            cat_idx++;
        }
    }

    // Ordinal zscore tables: 256 floats each
    float **zscore = malloc(n_ord * sizeof(float *));
    int ord_idx = 0;
    for (int j = 0; j < d; j++) {
        if (col_type[j] == 0) {
            if (remaining < 256 * 4) { fprintf(stderr, "truncated zscore\n"); return 3; }
            zscore[ord_idx] = (float *)ptr; ptr += 256 * 4; remaining -= 256 * 4;
            ord_idx++;
        }
    }

    // Categorical maps: uint16[256] each
    uint16_t **cat_map = malloc(n_cat * sizeof(uint16_t *));
    cat_idx = 0;
    for (int j = 0; j < d; j++) {
        if (col_type[j] == 1) {
            if (remaining < 256 * 2) { fprintf(stderr, "truncated cat_map\n"); return 3; }
            cat_map[cat_idx] = (uint16_t *)ptr; ptr += 256 * 2; remaining -= 256 * 2;
            cat_idx++;
        }
    }

    if (remaining != 0) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    // Allocate inference buffers
    float *sum_vx = malloc(K * sizeof(float));
    float *sum_vx_sq = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        float linear = w0[0];
        ord_idx = 0; cat_idx = 0;
        for (int j = 0; j < d; j++) {
            uint8_t byte = xi[j];
            if (col_type[j] == 0) { // ordinal
                float x_val = zscore[ord_idx][byte];
                linear += w_ord[ord_idx] * x_val;
                ord_idx++;
            } else { // categorical
                uint16_t idx = cat_map[cat_idx][byte];
                linear += w_cat[cat_idx][idx];
                cat_idx++;
            }
        }

        // Interaction part
        memset(sum_vx, 0, K * sizeof(float));
        memset(sum_vx_sq, 0, K * sizeof(float));
        ord_idx = 0; cat_idx = 0;
        for (int j = 0; j < d; j++) {
            uint8_t byte = xi[j];
            if (col_type[j] == 0) { // ordinal
                float x_val = zscore[ord_idx][byte];
                float *v = v_ord + ord_idx * K;
                for (int u = 0; u < K; u++) {
                    float vx = v[u] * x_val;
                    sum_vx[u] += vx;
                    sum_vx_sq[u] += vx * vx;
                }
                ord_idx++;
            } else { // categorical
                uint16_t idx = cat_map[cat_idx][byte];
                float *v = v_cat[cat_idx] + idx * K;
                for (int u = 0; u < K; u++) {
                    float vx = v[u];
                    sum_vx[u] += vx;
                    sum_vx_sq[u] += vx * vx;
                }
                cat_idx++;
            }
        }

        float interaction = 0.0f;
        for (int u = 0; u < K; u++) {
            float s = sum_vx[u];
            interaction += s * s - sum_vx_sq[u];
        }
        interaction *= 0.5f;

        float logit = linear + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    return 0;
}
```