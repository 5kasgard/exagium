```hypothesis
A single deterministic decision tree (CART) on the same one-hot/standardized encoding as label-linear.
It captures non-linear feature interactions that a linear model misses, while keeping inference extremely fast
(only a few dozen node evaluations per record). Hyperparameters (max_depth, min_samples_leaf) are chosen on the
tuning split (last 20% of train.in). If dev accuracy does not exceed label-linear (0.8524) by at least 0.005,
the approach is not worthwhile.
```
```python
"""Decision tree family for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys
from array import array

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
MAX_DEPTH_GRID = [3, 4, 5, 6, 7, 8]
MIN_SAMPLES_LEAF_GRID = [1, 5, 10, 20]
MIN_SAMPLES_SPLIT = 2  # fixed


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class DecisionTree:
    def __init__(self, max_depth, min_samples_leaf, min_samples_split, n_classes):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_classes = n_classes
        self.nodes = []  # each node: (feature_index, threshold, left, right, leaf_class)
        self.n_features = None

    def gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p * p)

    def best_split(self, X, y, feat_indices):
        n, k = X.shape
        best_gini = self.gini(y)
        best_feat = -1
        best_thresh = 0.0
        # For each feature, try thresholds at midpoints of sorted unique values
        for f in feat_indices:
            vals = X[:, f]
            # For binary features (0/1) only threshold 0.5 matters; for continuous we sort unique
            unique_vals = np.unique(vals)
            if len(unique_vals) == 1:
                continue
            # candidate thresholds: midpoints between consecutive unique values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                gini_left = self.gini(y[left_mask])
                gini_right = self.gini(y[right_mask])
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feat = f
                    best_thresh = thresh
        return best_feat, best_thresh, best_gini

    def build(self, X, y, depth=0, feat_indices=None):
        if feat_indices is None:
            feat_indices = np.arange(self.n_features)
        n = len(y)
        # Stopping criteria
        if depth >= self.max_depth or n < self.min_samples_split or len(np.unique(y)) == 1:
            leaf_class = np.bincount(y, minlength=self.n_classes).argmax()
            self.nodes.append((-1, 0.0, -1, -1, leaf_class))
            return len(self.nodes) - 1

        best_feat, best_thresh, _ = self.best_split(X, y, feat_indices)
        if best_feat == -1:
            leaf_class = np.bincount(y, minlength=self.n_classes).argmax()
            self.nodes.append((-1, 0.0, -1, -1, leaf_class))
            return len(self.nodes) - 1

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        left_idx = self.build(X[left_mask], y[left_mask], depth + 1, feat_indices)
        right_idx = self.build(X[right_mask], y[right_mask], depth + 1, feat_indices)
        node_idx = len(self.nodes)
        self.nodes.append((best_feat, best_thresh, left_idx, right_idx, -1))
        return node_idx

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.nodes = []
        self.build(X, y)
        return self


def encode_data(X_raw, types, fit_encoders=None):
    """Encode raw bytes (n, d) -> (Z, encoders).
    If fit_encoders is None, fit new encoders and return them.
    Otherwise use provided encoders (list per column)."""
    n, d = X_raw.shape
    encoders = []
    Z_parts = []
    feat_info = []  # for each column: (type, mean, std, feat_idx) or (type, cat_to_feat, feat_start)
    feat_idx = 0
    for j in range(d):
        col = X_raw[:, j].astype(np.float64)
        t = types[j]
        if t == "ordinal":
            if fit_encoders is None:
                mu = col.mean()
                sd = col.std()
                if sd == 0:
                    sd = 1.0
                encoders.append(('ordinal', mu, sd))
            else:
                mu, sd = fit_encoders[j][1], fit_encoders[j][2]
            z = (col - mu) / sd
            Z_parts.append(z.reshape(-1, 1).astype(np.float32))
            feat_info.append(('ordinal', mu, sd, feat_idx))
            feat_idx += 1
        else:
            if fit_encoders is None:
                vals = np.unique(X_raw[:, j])
                cat_to_feat = np.full(256, -1, dtype=np.int16)
                for v in vals:
                    cat_to_feat[v] = feat_idx
                    feat_idx += 1
                encoders.append(('categorical', cat_to_feat))
            else:
                cat_to_feat = fit_encoders[j][1]
            # Build one-hot for this column
            z_col = np.zeros((n, feat_idx - (feat_idx - len(np.unique(cat_to_feat[cat_to_feat != -1])))), dtype=np.float32)
            # Actually we need to know how many categories for this column
            cat_indices = cat_to_feat[X_raw[:, j]]
            valid = cat_indices != -1
            z_col[valid, cat_indices[valid] - (feat_idx - z_col.shape[1])] = 1.0
            Z_parts.append(z_col)
            feat_info.append(('categorical', cat_to_feat, feat_idx - z_col.shape[1]))
    Z = np.concatenate(Z_parts, axis=1)
    return Z, encoders, feat_info


def evaluate_tree(tree, X, y):
    n = len(y)
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        node_idx = 0
        while True:
            feat, thresh, left, right, leaf = tree.nodes[node_idx]
            if feat == -1:
                preds[i] = leaf
                break
            if X[i, feat] <= thresh:
                node_idx = left
            else:
                node_idx = right
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    C = card["classes"]

    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)

    # Split: last 20% for tuning
    split = int(n * 0.8)
    X_train, y_train = X_raw[:split], y[:split]
    X_tune, y_tune = X_raw[split:], y[split:]

    # Encode training split to get encoders
    Z_train, encoders, feat_info = encode_data(X_train, types, fit_encoders=None)
    # Encode tuning split with same encoders
    Z_tune, _, _ = encode_data(X_tune, types, fit_encoders=encoders)

    # Grid search
    best_score = -1
    best_params = None
    for max_depth in MAX_DEPTH_GRID:
        for min_samples_leaf in MIN_SAMPLES_LEAF_GRID:
            tree = DecisionTree(max_depth, min_samples_leaf, MIN_SAMPLES_SPLIT, C)
            tree.fit(Z_train, y_train)
            score = evaluate_tree(tree, Z_tune, y_tune)
            if score > best_score:
                best_score = score
                best_params = (max_depth, min_samples_leaf)
                print(f"Tuning: depth={max_depth}, min_leaf={min_samples_leaf} -> acc={score:.4f}")

    print(f"Best params: depth={best_params[0]}, min_leaf={best_params[1]}, tune_acc={best_score:.4f}")

    # Retrain on full data
    Z_full, encoders_full, feat_info_full = encode_data(X_raw, types, fit_encoders=None)
    tree = DecisionTree(best_params[0], best_params[1], MIN_SAMPLES_SPLIT, C)
    tree.fit(Z_full, y_full)

    # Write weights.bin
    # Header: magic, d, k, n_nodes, n_classes
    k = Z_full.shape[1]
    n_nodes = len(tree.nodes)
    magic = 0x45455254  # "TREE"
    out = struct.pack("<IIIII", magic, d, k, n_nodes, C)

    # Column encoders
    for j in range(d):
        info = feat_info_full[j]
        if info[0] == 'ordinal':
            _, mu, sd, feat_idx = info
            out += struct.pack("<BffI", 0, mu, sd, feat_idx)
        else:
            _, cat_to_feat, feat_start = info
            out += struct.pack("<B", 1)
            # cat_to_feat is int16 array of length 256
            out += array('h', cat_to_feat).tobytes()

    # Tree nodes
    for node in tree.nodes:
        feat, thresh, left, right, leaf = node
        out += struct.pack("<ifiii", feat, thresh, left, right, leaf)

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Decision tree inference for tabular label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (original features). Output: one byte class index. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    uint8_t type;        // 0 ordinal, 1 categorical
    float mean, std;     // ordinal
    uint32_t feat_idx;   // ordinal: single feature index
    int16_t *cat_to_feat; // categorical: 256 int16_t, -1 for unseen
    uint32_t feat_start; // categorical: first feature index for this column
} ColumnEnc;

typedef struct {
    int32_t feature;
    float threshold;
    int32_t left;
    int32_t right;
    int32_t leaf_class;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)w;
    if (hd[0] != 0x45455254) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2], n_nodes = hd[3], C = hd[4];
    if (C != 2 || d != 14) { fprintf(stderr, "unexpected d or C\n"); return 3; }

    const unsigned char *ptr = w + 20;
    const unsigned char *end = w + nw;

    ColumnEnc *cols = calloc(d, sizeof(ColumnEnc));
    for (int j = 0; j < d; j++) {
        if (ptr + 1 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        cols[j].type = *ptr++;
        if (cols[j].type == 0) { // ordinal
            if (ptr + 12 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
            cols[j].mean = *(float *)ptr; ptr += 4;
            cols[j].std = *(float *)ptr; ptr += 4;
            cols[j].feat_idx = *(uint32_t *)ptr; ptr += 4;
        } else { // categorical
            if (ptr + 512 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
            cols[j].cat_to_feat = (int16_t *)ptr;
            ptr += 512;
            // feat_start will be set later
        }
    }

    // Compute feat_start for categorical columns
    for (int j = 0; j < d; j++) {
        if (cols[j].type == 1) {
            // find minimum feature index in cat_to_feat (excluding -1)
            int16_t min_idx = 32767;
            for (int v = 0; v < 256; v++) {
                if (cols[j].cat_to_feat[v] != -1 && cols[j].cat_to_feat[v] < min_idx)
                    min_idx = cols[j].cat_to_feat[v];
            }
            cols[j].feat_start = (min_idx == 32767) ? 0 : min_idx;
        }
    }

    if (ptr + (long)n_nodes * 20 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
    Node *nodes = (Node *)ptr;

    float *feat = calloc(k, sizeof(float));
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k_rec; const unsigned char *xi = next_rec(x, nx, &pos, &k_rec);
        if ((int)k_rec != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k_rec, d); return 3; }

        memset(feat, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            uint8_t val = xi[j];
            if (cols[j].type == 0) {
                feat[cols[j].feat_idx] = (val - cols[j].mean) / cols[j].std;
            } else {
                int16_t idx = cols[j].cat_to_feat[val];
                if (idx != -1) feat[idx] = 1.0f;
            }
        }

        int node_idx = 0;
        while (1) {
            Node *nd = &nodes[node_idx];
            if (nd->feature == -1) {
                out_byte = (unsigned char)nd->leaf_class;
                break;
            }
            if (feat[nd->feature] <= nd->threshold)
                node_idx = nd->left;
            else
                node_idx = nd->right;
        }
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```