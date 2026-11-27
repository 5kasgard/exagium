```hypothesis
We propose a Factorization Machine family "label-fm". FMs model all pairwise feature interactions via low-rank embeddings, which is ideal for tabular data with mixed categorical/ordinal columns. They typically surpass linear models on such tasks while keeping inference O(k*d) – much faster than GBDT. We expect dev accuracy >0.855 (beating label-linear's 0.8524) with CPU cost close to label-linear. If accuracy is ≤0.855 or instructions/item exceed 2× label-linear, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks (label-fm).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-scored continuous, categorical as 256-category fields.
Trains FM with Adam on logistic loss. Hyperparameters chosen on last 20% tuning split.
Exports: magic 0x464D3132, d, k, C, 14 type bytes, 14 (mean,std) floats, bias,
then per-feature: ordinal -> (w, v[k]); categorical -> 256×(w, v[k])."""
import json, math, struct, sys

import numpy as np

SEED = 1
GRID = [
    {"k": 4, "lr": 0.01, "epochs": 10},
    {"k": 8, "lr": 0.01, "epochs": 10},
    {"k": 16, "lr": 0.03, "epochs": 10},
]
FINAL_EPOCHS = 30
BATCH = 256
L2 = 1e-5


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_data(X, y, ratio=0.8):
    n = len(X)
    cut = int(n * ratio)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def normalize_ordinal(X_ord, mean, std):
    return (X_ord - mean) / std


class FMModel:
    def __init__(self, d, types, k, rng):
        self.d = d
        self.types = types  # list of 0/1
        self.k = k
        self.ord_idx = [i for i, t in enumerate(types) if t == 0]
        self.cat_idx = [i for i, t in enumerate(types) if t == 1]
        self.n_ord = len(self.ord_idx)
        self.n_cat = len(self.cat_idx)
        # total linear features: each ordinal=1, each categorical=256
        self.total_feat = self.n_ord + self.n_cat * 256
        # offsets: for ordinal features, single index; for cat, start of 256 block
        self.offset = np.zeros(d, dtype=np.int32)
        cur = 0
        for j in range(d):
            if types[j] == 0:
                self.offset[j] = cur
                cur += 1
            else:
                self.offset[j] = cur
                cur += 256
        assert cur == self.total_feat
        # parameters
        self.bias = np.float32(0.0)
        self.w = np.zeros(self.total_feat, dtype=np.float32)
        self.v = rng.uniform(-0.01, 0.01, (self.total_feat, k)).astype(np.float32)
        # Adam state
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_v = np.zeros_like(self.v)
        self.v_v = np.zeros_like(self.v)
        self.m_b = 0.0
        self.v_b = 0.0
        self.t = 0

    def forward_batch(self, X_ord, X_cat):
        """X_ord: (batch, n_ord) float32 normalized; X_cat: (batch, n_cat) uint8 indices"""
        batch = X_ord.shape[0]
        linear = np.full(batch, self.bias, dtype=np.float32)
        sum_v = np.zeros((batch, self.k), dtype=np.float32)
        sum_v2 = np.zeros((batch, self.k), dtype=np.float32)
        # ordinal features
        for idx, j in enumerate(self.ord_idx):
            val = X_ord[:, idx]  # (batch,)
            off = self.offset[j]
            linear += self.w[off] * val
            sum_v += self.v[off] * val[:, None]
            sum_v2 += (self.v[off] ** 2) * (val[:, None] ** 2)
        # categorical features
        for idx, j in enumerate(self.cat_idx):
            cats = X_cat[:, idx]  # (batch,) uint8
            off = self.offset[j]
            # vectorized gather
            w_cats = self.w[off + cats]  # (batch,)
            v_cats = self.v[off + cats]  # (batch, k)
            linear += w_cats
            sum_v += v_cats
            sum_v2 += v_cats ** 2
        interaction = 0.5 * np.sum(sum_v ** 2 - sum_v2, axis=1)
        logit = linear + interaction
        prob = 1.0 / (1.0 + np.exp(-logit))
        return logit, prob

    def train_step(self, X_ord, X_cat, y, lr):
        batch = X_ord.shape[0]
        logit, prob = self.forward_batch(X_ord, X_cat)
        # gradient of loss w.r.t logit: prob - y
        grad_logit = (prob - y).astype(np.float32)  # (batch,)
        # gradients for bias
        g_bias = grad_logit.mean()
        # gradients for linear weights and embeddings
        g_w = np.zeros_like(self.w)
        g_v = np.zeros_like(self.v)
        # ordinal
        for idx, j in enumerate(self.ord_idx):
            val = X_ord[:, idx]  # (batch,)
            off = self.offset[j]
            g_w[off] = np.mean(grad_logit * val)
            g_v[off] = np.mean(grad_logit[:, None] * val[:, None] * (self.v_sum(val) - self.v[off] * val[:, None]), axis=0)
        # categorical
        for idx, j in enumerate(self.cat_idx):
            cats = X_cat[:, idx]
            off = self.offset[j]
            # accumulate per category
            for c in range(256):
                mask = (cats == c)
                if not np.any(mask):
                    continue
                g = grad_logit[mask]
                g_w[off + c] = g.mean()
                # interaction gradient: sum_v * 1 - v_c * 1
                sum_v_batch = self.get_sum_v(X_ord[mask], X_cat[mask])  # (n_mask, k)
                g_v[off + c] = np.mean(g[:, None] * (sum_v_batch - self.v[off + c]), axis=0)
        # L2 regularization
        g_w += L2 * self.w
        g_v += L2 * self.v
        # Adam update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        # bias
        self.m_b = beta1 * self.m_b + (1 - beta1) * g_bias
        self.v_b = beta2 * self.v_b + (1 - beta2) * g_bias * g_bias
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)
        self.bias -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
        # weights
        self.m_w = beta1 * self.m_w + (1 - beta1) * g_w
        self.v_w = beta2 * self.v_w + (1 - beta2) * g_w * g_w
        m_w_hat = self.m_w / (1 - beta1 ** self.t)
        v_w_hat = self.v_w / (1 - beta2 ** self.t)
        self.w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
        # embeddings
        self.m_v = beta1 * self.m_v + (1 - beta1) * g_v
        self.v_v = beta2 * self.v_v + (1 - beta2) * g_v * g_v
        m_v_hat = self.m_v / (1 - beta1 ** self.t)
        v_v_hat = self.v_v / (1 - beta2 ** self.t)
        self.v -= lr * m_v_hat / (np.sqrt(v_v_hat) + eps)

    def get_sum_v(self, X_ord, X_cat):
        """Helper to compute sum_v for a batch (used in cat gradient)."""
        batch = X_ord.shape[0]
        sum_v = np.zeros((batch, self.k), dtype=np.float32)
        for idx, j in enumerate(self.ord_idx):
            val = X_ord[:, idx]
            off = self.offset[j]
            sum_v += self.v[off] * val[:, None]
        for idx, j in enumerate(self.cat_idx):
            cats = X_cat[:, idx]
            off = self.offset[j]
            sum_v += self.v[off + cats]
        return sum_v

    def evaluate(self, X_ord, X_cat, y):
        _, prob = self.forward_batch(X_ord, X_cat)
        pred = (prob > 0.5).astype(np.int64)
        return np.mean(pred == y)

    def pack_weights(self, mean_ord, std_ord):
        """Serialize to bytes as per infer.c format."""
        # header
        magic = 0x464D3132  # 'FM12'
        d = self.d
        k = self.k
        C = 2
        types_arr = np.array(self.types, dtype=np.uint8)
        # mean/std for all 14 features (0 for categorical)
        mean_all = np.zeros(d, dtype=np.float32)
        std_all = np.ones(d, dtype=np.float32)
        for idx, j in enumerate(self.ord_idx):
            mean_all[j] = mean_ord[idx]
            std_all[j] = std_ord[idx]
        # pack
        out = struct.pack("<4i", magic, d, k, C)
        out += types_arr.tobytes()
        out += mean_all.astype("<f4").tobytes()
        out += std_all.astype("<f4").tobytes()
        out += np.array([self.bias], dtype="<f4").tobytes()
        # per feature in order 0..13
        for j in range(d):
            if self.types[j] == 0:
                off = self.offset[j]
                out += np.array([self.w[off]], dtype="<f4").tobytes()
                out += self.v[off].astype("<f4").tobytes()
            else:
                off = self.offset[j]
                out += self.w[off:off+256].astype("<f4").tobytes()
                out += self.v[off:off+256].astype("<f4").tobytes()
        return out


def main():
    card = json.load(open(sys.argv[1]))
    types = [0 if t == "ordinal" else 1 for t in card["input"]["types"]]
    d = card["input"]["bytes"]
    assert d == 14
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X_bytes)
    # split ordinal/categorical columns
    ord_idx = [i for i, t in enumerate(types) if t == 0]
    cat_idx = [i for i, t in enumerate(types) if t == 1]
    X_ord = X_bytes[:, ord_idx].astype(np.float64)
    X_cat = X_bytes[:, cat_idx].astype(np.uint8)
    # normalization stats on full training data
    mean_ord = X_ord.mean(axis=0)
    std_ord = X_ord.std(axis=0)
    std_ord[std_ord == 0] = 1.0
    X_ord_norm = (X_ord - mean_ord) / std_ord
    # tuning split (last 20%)
    cut = int(n * 0.8)
    X_ord_tr, X_ord_val = X_ord_norm[:cut], X_ord_norm[cut:]
    X_cat_tr, X_cat_val = X_cat[:cut], X_cat[cut:]
    y_tr, y_val = y[:cut], y[cut:]
    # grid search on tuning split
    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_cfg = None
    best_model = None
    for cfg in GRID:
        model = FMModel(d, types, cfg["k"], rng)
        for epoch in range(cfg["epochs"]):
            order = rng.permutation(len(y_tr))
            for s in range(0, len(y_tr), BATCH):
                idx = order[s:s+BATCH]
                model.train_step(X_ord_tr[idx], X_cat_tr[idx], y_tr[idx], cfg["lr"])
        acc = model.evaluate(X_ord_val, X_cat_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_model = model
    # retrain on full data with best config
    rng = np.random.default_rng(SEED)
    model = FMModel(d, types, best_cfg["k"], rng)
    for epoch in range(FINAL_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            model.train_step(X_ord_norm[idx], X_cat[idx], y[idx], best_cfg["lr"])
    # write weights
    weights_bytes = model.pack_weights(mean_ord.astype(np.float32), std_ord.astype(np.float32))
    open(sys.argv[4], "wb").write(weights_bytes)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label-fm.
 * ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output record: 1 byte class (0/1). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 4) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | (uint32_t)wb[3]<<24;
    if (magic != 0x464D3132) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t *hdr = (int32_t *)(wb + 4);
    int d = hdr[0], k = hdr[1], C = hdr[2];
    if (d != 14 || C != 2 || k <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    unsigned char *types = wb + 4 + 12;
    float *mean = (float *)(wb + 4 + 12 + 14);
    float *std = (float *)(wb + 4 + 12 + 14 + 14*4);
    float bias = *((float *)(wb + 4 + 12 + 14 + 14*8));
    float *params = (float *)(wb + 4 + 12 + 14 + 14*8 + 4);
    // compute expected size
    long expected = 4 + 12 + 14 + 14*8 + 4; // header + bias
    for (int j = 0; j < d; j++) {
        if (types[j] == 0) expected += 4 + 4*k;          // ordinal: w + v[k]
        else expected += 256 * (4 + 4*k);                // categorical: 256*(w + v[k])
    }
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    // precompute offsets for each feature
    int offset[14];
    int cur = 0;
    for (int j = 0; j < d; j++) {
        offset[j] = cur;
        if (types[j] == 0) cur += 1;
        else cur += 256;
    }
    // linear weights and embeddings are interleaved per feature? Actually layout:
    // after bias, for each feature j:
    //   if ordinal: float w, float v[k]
    //   if cat: 256 * (float w, float v[k])
    // So we can compute pointers by iterating.
    float *w_ptr = params;
    float *v_ptr = NULL; // we'll compute per feature

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[16], sum_v2[16]; // k <= 16 in our grid
    unsigned char pred;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if (len != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float linear = bias;
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        // iterate features
        float *wp = params;
        for (int j = 0; j < d; j++) {
            if (types[j] == 0) { // ordinal
                float val = (xi[j] - mean[j]) / std[j];
                float w = wp[0];
                float *v = wp + 1;
                linear += w * val;
                for (int f = 0; f < k; f++) {
                    float vf = v[f];
                    sum_v[f] += vf * val;
                    sum_v2[f] += vf * vf * val * val;
                }
                wp += 1 + k;
            } else { // categorical
                int cat = xi[j];
                float w = wp[cat * (1 + k)];
                float *v = wp + cat * (1 + k) + 1;
                linear += w;
                for (int f = 0; f < k; f++) {
                    float vf = v[f];
                    sum_v[f] += vf;
                    sum_v2[f] += vf * vf;
                }
                wp += 256 * (1 + k);
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += sum_v[f]*sum_v[f] - sum_v2[f];
        inter *= 0.5f;
        float logit = linear + inter;
        pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```