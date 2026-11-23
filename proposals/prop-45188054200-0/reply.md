```hypothesis
Propose label-fm: a factorization machine with low-rank embeddings for pairwise feature interactions.
Adult has many categorical fields where interactions (e.g., education×occupation) are predictive.
FM captures these with O(d·k) parameters and inference cost, staying fast.
If dev accuracy ≤ label-linear (0.8524) or instructions/item ≥ label-linear (719), the approach fails.
```

```python
"""Factorization Machine for tabular label tasks (label-fm).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Encodes each column as categorical (0-255), adds an "unseen" bucket for 255.
Model: bias + linear weights + k-dim embeddings per category.
Trains with Adam (lr=0.03, L2=1e-3, batch=64) on full data after tuning k∈{4,8,16} on last 20%.
Exports: magic 0x464D4C31, d, k, total_cats; int32 mapping[d*256]; float32 bias, linear[total_cats], emb[total_cats*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
EPOCHS_TUNE = 8
EPOCHS_FULL = 24
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_mappings(X, d):
    """Return (mapping_flat, cat_offsets, total_cats).
    mapping_flat[j*256 + v] = global category index."""
    mapping = np.full(d * 256, -1, dtype=np.int32)
    cat_offsets = np.zeros(d + 1, dtype=np.int32)
    total = 0
    for j in range(d):
        vals = np.unique(X[:, j])
        # ensure 255 (unseen) is present
        if 255 not in vals:
            vals = np.append(vals, 255)
        ncat = len(vals)
        cat_offsets[j] = total
        for idx, v in enumerate(vals):
            mapping[j * 256 + v] = total + idx
        total += ncat
    cat_offsets[d] = total
    return mapping, cat_offsets, total

def indices_from_mapping(X, mapping, d):
    n = X.shape[0]
    idx = np.empty((n, d), dtype=np.int32)
    for j in range(d):
        idx[:, j] = mapping[j * 256 + X[:, j]]
    return idx

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def forward(indices, bias, linear, emb, k):
    """indices: (batch, d). Returns logits (batch,)."""
    batch, d = indices.shape
    lin = bias + np.sum(linear[indices], axis=1)
    e = emb[indices]                     # (batch, d, k)
    sum_e = e.sum(axis=1)                # (batch, k)
    sum_e_sq = (e * e).sum(axis=1)       # (batch, k)
    inter = 0.5 * np.sum(sum_e * sum_e - sum_e_sq, axis=1)
    return lin + inter

def train_fm(X_idx, y, total_cats, k, lr, l2, epochs, batch_size, rng):
    d = X_idx.shape[1]
    bias = np.float32(0.0)
    linear = np.zeros(total_cats, dtype=np.float32)
    emb = (rng.normal(0, 1/math.sqrt(k), (total_cats, k))).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    n = X_idx.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            ind = X_idx[idx]
            yy = y[idx]
            batch = ind.shape[0]

            # forward
            lin_part = bias + np.sum(linear[ind], axis=1)
            e = emb[ind]                       # (batch, d, k)
            sum_e = e.sum(axis=1)
            sum_e_sq = (e * e).sum(axis=1)
            inter = 0.5 * np.sum(sum_e * sum_e - sum_e_sq, axis=1)
            logits = lin_part + inter

            # gradient of logistic loss
            p = sigmoid(logits)
            g = (p - yy).astype(np.float32)    # (batch,)

            # bias grad
            g_bias = g.mean()

            # linear grad
            g_lin = np.zeros_like(linear)
            np.add.at(g_lin, ind.ravel(), np.repeat(g, d))
            g_lin /= batch
            g_lin += l2 * linear

            # embedding grad
            g_emb = np.zeros_like(emb)
            # for each sample and feature: g * (sum_e - e)
            diff = (sum_e[:, None, :] - e) * g[:, None, None]   # (batch, d, k)
            np.add.at(g_emb, ind.ravel(), diff.reshape(-1, k))
            g_emb /= batch
            g_emb += l2 * emb

            t += 1
            # Adam updates
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= lr * (m_bias / (1 - beta1**t)) / (math.sqrt(v_bias / (1 - beta2**t)) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= lr * (m_lin / (1 - beta1**t)) / (np.sqrt(v_lin / (1 - beta2**t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= lr * (m_emb / (1 - beta1**t)) / (np.sqrt(v_emb / (1 - beta2**t)) + eps)

    return bias, linear, emb

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # build mappings from all training data (allowed, we see train.in)
    mapping, cat_offsets, total_cats = build_mappings(X, d)
    X_idx = indices_from_mapping(X, mapping, d)

    # split for tuning (last 20%)
    n = X.shape[0]
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)

    # hyperparameter grid
    best_acc = -1.0
    best_params = None
    for k in (4, 8, 16):
        for lr in (0.01, 0.03, 0.1):
            for l2 in (1e-4, 1e-3):
                bias, linear, emb = train_fm(X_idx[train_idx], y[train_idx], total_cats, k, lr, l2,
                                             EPOCHS_TUNE, BATCH, rng)
                logits_val = forward(X_idx[val_idx], bias, linear, emb, k)
                pred = (logits_val > 0).astype(np.int64)
                acc = (pred == y[val_idx]).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = (k, lr, l2)

    k, lr, l2 = best_params
    # retrain on full data
    bias, linear, emb = train_fm(X_idx, y, total_cats, k, lr, l2,
                                 EPOCHS_FULL, BATCH, rng)

    # pack weights
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4I", magic, d, k, total_cats)
    mapping_bytes = mapping.astype("<i4").tobytes()
    bias_bytes = np.array([bias], dtype="<f4").tobytes()
    linear_bytes = linear.astype("<f4").tobytes()
    emb_bytes = emb.astype("<f4").tobytes()
    out = header + mapping_bytes + bias_bytes + linear_bytes + emb_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label-fm: ./infer weights.bin in.rec N
 * Input record: d bytes (one per column). Output: one byte class (0/1).
 * Weights format: magic 0x464D4C31, d, k, total_cats; int32 mapping[d*256];
 * float32 bias, linear[total_cats], emb[total_cats*k]. */
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
    uint32_t *hd = (uint32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], total_cats = hd[3];
    if (d <= 0 || k <= 0 || k > 64 || total_cats <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long mapping_size = (long)d * 256;
    long expected = 16 + mapping_size * 4 + 4 + (long)total_cats * 4 + (long)total_cats * k * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    int32_t *mapping = (int32_t *)(wb + 16);
    float *bias = (float *)(wb + 16 + mapping_size * 4);
    float *linear = bias + 1;
    float *emb = linear + total_cats;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = (float *)malloc(k * sizeof(float));
    float *sum_e_sq = (float *)malloc(k * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float score = *bias;
        memset(sum_e, 0, k * sizeof(float));
        memset(sum_e_sq, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            int idx = mapping[j * 256 + xi[j]];
            if (idx < 0 || idx >= total_cats) { fprintf(stderr, "bad mapping index\n"); return 3; }
            score += linear[idx];
            float *e = emb + (long)idx * k;
            for (int u = 0; u < k; u++) {
                float v = e[u];
                sum_e[u] += v;
                sum_e_sq[u] += v * v;
            }
        }
        for (int u = 0; u < k; u++) score += 0.5f * (sum_e[u] * sum_e[u] - sum_e_sq[u]);
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```