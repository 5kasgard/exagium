```hypothesis
Family: label-fm (factorization machine with low-rank embeddings).
Why: Captures pairwise feature interactions via FM trick (O(d*k) inference), which helps on tabular data with mixed categorical/numeric features where interactions matter (e.g., education × occupation). Linear model ignores interactions; MLP adds overhead. With k=4, inference cost should stay near label-linear while improving accuracy.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ 2325 (label-mlp16), the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning (last 20% of train.in)
K_CHOICES = [2, 4, 8]
LR_CHOICES = [1e-3, 3e-3]
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000
SEED = 1

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def build_mappings(X, types):
    """Return per-column mappings for ordinal (mean,std) and categorical (value->index)."""
    d = X.shape[1]
    ord_means, ord_stds = [], []
    cat_maps, cat_sizes = [], []
    for j, t in enumerate(types):
        col = X[:, j]
        if t == "ordinal":
            vals = col.astype(np.float64)
            mu = vals.mean()
            sd = vals.std()
            if sd == 0: sd = 1.0
            ord_means.append(mu)
            ord_stds.append(sd)
            cat_maps.append(None)
            cat_sizes.append(0)
        else:
            unique_vals = np.unique(col)
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            cat_maps.append(val_to_idx)
            cat_sizes.append(len(unique_vals))
            ord_means.append(0.0)
            ord_stds.append(1.0)
    return ord_means, ord_stds, cat_maps, cat_sizes

def build_feat_idx_val(d, types, ord_means, ord_stds, cat_maps, cat_sizes):
    """Create (d,256) arrays: feature index and feature value for each byte value."""
    feat_idx = np.full((d, 256), -1, dtype=np.int32)
    feat_val = np.zeros((d, 256), dtype=np.float32)
    col_feature_start = []
    total_features = 0
    for j in range(d):
        col_feature_start.append(total_features)
        if types[j] == "ordinal":
            total_features += 1
        else:
            total_features += cat_sizes[j]
    for j in range(d):
        if types[j] == "ordinal":
            idx = col_feature_start[j]
            for val in range(256):
                normalized = (val - ord_means[j]) / ord_stds[j]
                feat_idx[j, val] = idx
                feat_val[j, val] = normalized
        else:
            for val in range(256):
                if val in cat_maps[j]:
                    idx = col_feature_start[j] + cat_maps[j][val]
                    feat_idx[j, val] = idx
                    feat_val[j, val] = 1.0
                else:
                    feat_idx[j, val] = -1
                    feat_val[j, val] = 0.0
    return feat_idx, feat_val, total_features, col_feature_start

def train_fm(X, y, types, k, lr, l2, batch, max_epochs, samples, rng, feat_idx, feat_val, total_features):
    n, d = X.shape
    # Parameters: global bias w0, linear weights w[total_features], embeddings v[total_features, k]
    w0 = np.float32(0.0)
    w = np.zeros(total_features, dtype=np.float32)
    v = (rng.uniform(-1, 1, (total_features, k)) * math.sqrt(6.0 / (k + 1))).astype(np.float32)
    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx_batch = order[s:s+batch]
            x_batch = X[idx_batch]
            y_batch = y[idx_batch]
            batch_size = len(idx_batch)
            linear_sum = np.zeros(batch_size, dtype=np.float32)
            emb_sum = np.zeros((batch_size, k), dtype=np.float32)
            sq_sum = np.zeros(batch_size, dtype=np.float32)
            # Forward pass
            for j in range(d):
                vals = x_batch[:, j]
                idxs = feat_idx[j, vals]
                vals_f = feat_val[j, vals]
                mask = idxs >= 0
                if not mask.any():
                    continue
                idxs_m = idxs[mask]
                vals_m = vals_f[mask]
                linear_sum[mask] += w[idxs_m] * vals_m
                emb_sum[mask] += v[idxs_m] * vals_m[:, None]
                v_sq = (v[idxs_m] ** 2).sum(axis=1)
                sq_sum[mask] += v_sq * (vals_m ** 2)
            interaction = 0.5 * ((emb_sum ** 2).sum(axis=1) - sq_sum)
            logit = w0 + linear_sum + interaction
            p = sigmoid(logit)
            grad_logit = p - y_batch  # (batch_size,)
            # Backward pass
            g_w0 = grad_logit.sum()
            g_w = np.zeros_like(w)
            g_v = np.zeros_like(v)
            for j in range(d):
                vals = x_batch[:, j]
                idxs = feat_idx[j, vals]
                vals_f = feat_val[j, vals]
                mask = idxs >= 0
                if not mask.any():
                    continue
                idxs_m = idxs[mask]
                vals_m = vals_f[mask]
                grad_m = grad_logit[mask]
                s_m = emb_sum[mask]
                # Linear weight gradients
                np.add.at(g_w, idxs_m, grad_m * vals_m)
                # Embedding gradients
                term = grad_m[:, None] * (vals_m[:, None] * s_m - (vals_m[:, None] ** 2) * v[idxs_m])
                np.add.at(g_v, idxs_m, term)
            # L2 regularization
            g_w += l2 * w
            g_v += l2 * v
            # Adam updates
            t += 1
            # w0
            m_w0 = 0.9 * m_w0 + 0.1 * g_w0
            v_w0 = 0.999 * v_w0 + 0.001 * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - 0.9 ** t)
            v_w0_hat = v_w0 / (1 - 0.999 ** t)
            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + 1e-8)
            # w
            m_w = 0.9 * m_w + 0.1 * g_w
            v_w = 0.999 * v_w + 0.001 * g_w * g_w
            m_w_hat = m_w / (1 - 0.9 ** t)
            v_w_hat = v_w / (1 - 0.999 ** t)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
            # v
            m_v = 0.9 * m_v + 0.1 * g_v
            v_v = 0.999 * v_v + 0.001 * g_v * g_v
            m_v_hat = m_v / (1 - 0.9 ** t)
            v_v_hat = v_v / (1 - 0.999 ** t)
            v -= lr * m_v_hat / (np.sqrt(v_v_hat) + 1e-8)
    return w0, w, v

def evaluate(X, y, types, w0, w, v, feat_idx, feat_val, k):
    n, d = X.shape
    linear_sum = np.zeros(n, dtype=np.float32)
    emb_sum = np.zeros((n, k), dtype=np.float32)
    sq_sum = np.zeros(n, dtype=np.float32)
    for j in range(d):
        vals = X[:, j]
        idxs = feat_idx[j, vals]
        vals_f = feat_val[j, vals]
        mask = idxs >= 0
        if not mask.any():
            continue
        idxs_m = idxs[mask]
        vals_m = vals_f[mask]
        linear_sum[mask] += w[idxs_m] * vals_m
        emb_sum[mask] += v[idxs_m] * vals_m[:, None]
        v_sq = (v[idxs_m] ** 2).sum(axis=1)
        sq_sum[mask] += v_sq * (vals_m ** 2)
    interaction = 0.5 * ((emb_sum ** 2).sum(axis=1) - sq_sum)
    logit = w0 + linear_sum + interaction
    p = sigmoid(logit)
    pred = (p > 0.5).astype(np.int64)
    return (pred == y).mean()

def build_lookup_tables(d, k, types, w0, w, v, feat_idx, feat_val, col_feature_start, cat_sizes, ord_means, ord_stds, cat_maps):
    """Produce flat arrays for inference: linear_contribution, embedding_vectors, sq_norm."""
    linear_contrib = np.zeros((d, 256), dtype=np.float32)
    embedding_vecs = np.zeros((d, 256, k), dtype=np.float32)
    sq_norm = np.zeros((d, 256), dtype=np.float32)
    for j in range(d):
        for val in range(256):
            idx = feat_idx[j, val]
            vf = feat_val[j, val]
            if idx >= 0:
                linear_contrib[j, val] = w[idx] * vf
                embedding_vecs[j, val] = v[idx] * vf
            else:
                linear_contrib[j, val] = 0.0
                embedding_vecs[j, val] = 0.0
            sq_norm[j, val] = (embedding_vecs[j, val] ** 2).sum()
    return linear_contrib, embedding_vecs, sq_norm

def write_weights(path, d, k, C, w0, linear_contrib, embedding_vecs, sq_norm):
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, k, C)
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.array([w0], dtype="<f4").tobytes())
        f.write(linear_contrib.astype("<f4").tobytes())
        f.write(embedding_vecs.astype("<f4").tobytes())
        f.write(sq_norm.astype("<f4").tobytes())

def main():
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]
    # Build mappings on full train (for final model) and on train split (for tuning)
    # We'll tune on train split, then retrain on full train.
    rng = np.random.default_rng(SEED)
    # Prepare mappings for train split
    ord_means, ord_stds, cat_maps, cat_sizes = build_mappings(X_train, types)
    feat_idx, feat_val, total_features, col_feature_start = build_feat_idx_val(d, types, ord_means, ord_stds, cat_maps, cat_sizes)
    best_acc = -1.0
    best_k, best_lr = 4, 3e-3
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            w0, w, v = train_fm(X_train, y_train, types, k, lr, L2, BATCH, MAX_EPOCHS, SAMPLES, rng, feat_idx, feat_val, total_features)
            acc = evaluate(X_tune, y_tune, types, w0, w, v, feat_idx, feat_val, k)
            if acc > best_acc:
                best_acc = acc
                best_k, best_lr = k, lr
    # Retrain on full train with best hyperparameters
    ord_means_f, ord_stds_f, cat_maps_f, cat_sizes_f = build_mappings(X, types)
    feat_idx_f, feat_val_f, total_features_f, col_feature_start_f = build_feat_idx_val(d, types, ord_means_f, ord_stds_f, cat_maps_f, cat_sizes_f)
    w0_f, w_f, v_f = train_fm(X, y, types, best_k, best_lr, L2, BATCH, MAX_EPOCHS, SAMPLES, rng, feat_idx_f, feat_val_f, total_features_f)
    # Build lookup tables
    linear_contrib, embedding_vecs, sq_norm = build_lookup_tables(d, best_k, types, w0_f, w_f, v_f, feat_idx_f, feat_val_f, col_feature_start_f, cat_sizes_f, ord_means_f, ord_stds_f, cat_maps_f)
    # Write weights
    write_weights(sys.argv[4], d, best_k, C, w0_f, linear_contrib, embedding_vecs, sq_norm)

if __name__ == "__main__":
    main()
```

```c
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
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k <= 0 || k > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 20 + 4UL * d * 256 * (2 + k);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias = (const float *)(hd + 4);
    const float *lin = (const float *)(bias + 1);
    const float *emb = lin + (size_t)d * 256;
    const float *sqn = emb + (size_t)d * 256 * k;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = (float *)malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float sum_lin = bias[0];
        memset(sum_emb, 0, sizeof(float) * k);
        float sum_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            sum_lin += lin[idx];
            const float *e = emb + (size_t)idx * k;
            for (int u = 0; u < k; u++) sum_emb[u] += e[u];
            sum_sq += sqn[idx];
        }
        float dot = 0.0f;
        for (int u = 0; u < k; u++) dot += sum_emb[u] * sum_emb[u];
        float inter = 0.5f * (dot - sum_sq);
        float logit = sum_lin + inter;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char cls = prob > 0.5f ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```