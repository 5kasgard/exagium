```hypothesis
Family: label-rf (random forest). Random forests handle non-linear interactions in the 8x8 pixel space better than a linear model. With 10-30 shallow trees (max depth 8-12) inference visits only a few hundred nodes per item, keeping CPU cost below label-linear. Hyperparameters (n_estimators, max_depth, min_samples_leaf) are chosen on the last 20% tuning split. Failure: dev accuracy ≤ 0.9613 or instructions/item ≥ 5058.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def find_best_split(X_col, y, idx, C):
    vals = X_col[idx]
    order = np.argsort(vals)
    sorted_vals = vals[order]
    sorted_y = y[idx][order]
    unique_vals, first_idx = np.unique(sorted_vals, return_index=True)
    n_samples = len(idx)
    if len(unique_vals) <= 1:
        return 1.0, None, None, None
    class_counts = np.bincount(sorted_y, minlength=C)
    best_gini = 1.0
    best_thresh = None
    best_left_idx = None
    best_right_idx = None
    left_counts = np.zeros(C, dtype=np.int32)
    for i in range(len(unique_vals) - 1):
        start = first_idx[i]
        end = first_idx[i + 1]
        left_counts += np.bincount(sorted_y[start:end], minlength=C)
        right_counts = class_counts - left_counts
        n_left = end
        n_right = n_samples - n_left
        if n_left == 0 or n_right == 0:
            continue
        gini_left = 1.0 - np.sum((left_counts / n_left) ** 2)
        gini_right = 1.0 - np.sum((right_counts / n_right) ** 2)
        gini = (n_left * gini_left + n_right * gini_right) / n_samples
        if gini < best_gini:
            best_gini = gini
            best_thresh = unique_vals[i]
            best_left_idx = idx[order[:end]]
            best_right_idx = idx[order[end:]]
    return best_gini, best_thresh, best_left_idx, best_right_idx

def build_tree(X, y, idx, depth, max_depth, min_samples_leaf, n_features, C, rng):
    if len(idx) < min_samples_leaf or depth == max_depth:
        leaf_class = int(np.argmax(np.bincount(y[idx], minlength=C)))
        return [(-1, -1, -1, -1, leaf_class)]
    d = X.shape[1]
    feature_indices = rng.choice(d, size=n_features, replace=False)
    best_gini = 1.0
    best_feature = -1
    best_thresh = -1
    best_left_idx = None
    best_right_idx = None
    for f in feature_indices:
        gini, thresh, left_idx, right_idx = find_best_split(X[:, f], y, idx, C)
        if gini < best_gini:
            best_gini = gini
            best_feature = f
            best_thresh = thresh
            best_left_idx = left_idx
            best_right_idx = right_idx
    if best_feature == -1 or best_left_idx is None or len(best_left_idx) < min_samples_leaf or len(best_right_idx) < min_samples_leaf:
        leaf_class = int(np.argmax(np.bincount(y[idx], minlength=C)))
        return [(-1, -1, -1, -1, leaf_class)]
    left_nodes = build_tree(X, y, best_left_idx, depth + 1, max_depth, min_samples_leaf, n_features, C, rng)
    right_nodes = build_tree(X, y, best_right_idx, depth + 1, max_depth, min_samples_leaf, n_features, C, rng)
    left_child = 1
    right_child = 1 + len(left_nodes)
    node = [(left_child, right_child, int(best_feature), int(best_thresh), -1)]
    return node + left_nodes + right_nodes

def train_forest(X, y, n_estimators, max_depth, min_samples_leaf, d, C, rng):
    n_train = X.shape[0]
    n_features = int(math.sqrt(d))
    trees = []
    for _ in range(n_estimators):
        boot_idx = rng.choice(n_train, size=n_train, replace=True)
        tree = build_tree(X, y, boot_idx, 0, max_depth, min_samples_leaf, n_features, C, rng)
        trees.append(tree)
    return trees

def predict_forest(trees, X):
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        votes = np.zeros(10, dtype=np.int32)
        for tree in trees:
            node_idx = 0
            while True:
                left, right, feat, thresh, leaf = tree[node_idx]
                if leaf != -1:
                    votes[leaf] += 1
                    break
                if X[i, feat] <= thresh:
                    node_idx = left
                else:
                    node_idx = right
        preds[i] = np.argmax(votes)
    return preds

def serialize_forest(trees, d, C, path):
    n_trees = len(trees)
    total_nodes = sum(len(t) for t in trees)
    tree_starts = []
    offset = 0
    for t in trees:
        tree_starts.append(offset)
        offset += len(t)
    magic = 0x52465354
    header = struct.pack("<IIIII", magic, n_trees, d, C, total_nodes)
    tree_starts_bytes = struct.pack("<" + "I" * n_trees, *tree_starts)
    nodes_bytes = b""
    for t in trees:
        for left, right, feat, thresh, leaf in t:
            nodes_bytes += struct.pack("<iiiii", left, right, feat, thresh, leaf)
    with open(path, "wb") as f:
        f.write(header)
        f.write(tree_starts_bytes)
        f.write(nodes_bytes)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    n_tune = int(0.2 * n)
    train_idx = np.arange(n - n_tune)
    tune_idx = np.arange(n - n_tune, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]
    param_grid = {
        'n_estimators': [10, 20, 30],
        'max_depth': [8, 10, 12],
        'min_samples_leaf': [1, 2, 5]
    }
    best_score = -1.0
    best_params = None
    best_forest = None
    rng = np.random.default_rng(SEED)
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_leaf in param_grid['min_samples_leaf']:
                forest = train_forest(X_train, y_train, n_est, max_d, min_leaf, d, C, rng)
                preds = predict_forest(forest, X_tune)
                acc = (preds == y_tune).mean()
                if acc > best_score:
                    best_score = acc
                    best_params = (n_est, max_d, min_leaf)
                    best_forest = forest
    final_forest = train_forest(X, y, *best_params, d, C, rng)
    serialize_forest(final_forest, d, C, sys.argv[4])

if __name__ == "__main__":
    main()
```

```c
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
    int32_t left;
    int32_t right;
    int32_t feature;
    int32_t threshold;
    int32_t leaf_class;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x52465354) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t n_trees = hdr[1];
    uint32_t n_features = hdr[2];
    uint32_t n_classes = hdr[3];
    uint32_t total_nodes = hdr[4];
    if (n_classes < 1 || n_classes > 256) { fprintf(stderr, "bad n_classes\n"); return 3; }
    size_t expected = 20 + (size_t)n_trees * 4 + (size_t)total_nodes * 20;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    uint32_t *tree_starts = (uint32_t *)(wb + 20);
    Node *nodes = (Node *)(wb + 20 + (size_t)n_trees * 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *votes = malloc(n_classes);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, n_features); return 3; }
        memset(votes, 0, n_classes);
        for (uint32_t t = 0; t < n_trees; t++) {
            uint32_t node_idx = tree_starts[t];
            while (1) {
                Node *nd = &nodes[node_idx];
                if (nd->leaf_class != -1) {
                    votes[nd->leaf_class]++;
                    break;
                }
                if (xi[nd->feature] <= (uint32_t)nd->threshold) node_idx += nd->left;
                else node_idx += nd->right;
            }
        }
        unsigned char best = 0;
        for (uint32_t c = 1; c < n_classes; c++) if (votes[c] > votes[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```