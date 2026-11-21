```hypothesis
Family: label-rf (Random Forest with shallow trees). Random Forests excel on tabular ordinal data like optdigits, often surpassing linear models. A small ensemble (10-30 trees, max depth 10) should achieve >0.97 dev accuracy with low inference cost (a few hundred instructions/item). If dev accuracy ≤ 0.96 or inference cost exceeds label-linear by >5×, the hypothesis fails.
```

```python
"""Random Forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, uses only stdlib and numpy.
Chooses n_trees and max_depth on the tuning split (last 20% of train.in),
then retrains on all data. Exports a compact binary format."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
N_TREES_GRID = [10, 20, 30]
MAX_DEPTH_GRID = [5, 10, 15]
MIN_SAMPLES_LEAF = 5
FEATURE_SUBSET_RATIO = 0.33  # sqrt(64) ≈ 8


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(y):
    """Gini impurity of integer labels y."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)


def best_split(X, y, feature_indices, rng):
    """Find best split among given features. Returns (feature, threshold, left_mask)."""
    best_gini = 1.0
    best_feat = -1
    best_thr = -1
    best_mask = None
    n_samples = len(y)
    parent_gini = gini(y)
    for feat in feature_indices:
        vals = X[:, feat]
        unique_vals = np.unique(vals)
        # Try thresholds between unique values
        for thr in unique_vals[:-1]:
            mask = vals <= thr
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            gini_left = gini(y[mask])
            gini_right = gini(y[~mask])
            weighted_gini = (mask.sum() * gini_left + (~mask).sum() * gini_right) / n_samples
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feat = feat
                best_thr = thr
                best_mask = mask
    if best_feat == -1:
        return None, None, None
    return best_feat, best_thr, best_mask


def build_tree(X, y, max_depth, min_samples_leaf, feature_subset_size, depth, rng):
    """Recursively build a decision tree. Returns flat arrays."""
    n_samples = len(y)
    # Stopping criteria
    if depth >= max_depth or n_samples < 2 * min_samples_leaf or gini(y) == 0.0:
        # Leaf: majority class (smallest class on tie)
        classes, counts = np.unique(y, return_counts=True)
        leaf_class = classes[np.argmax(counts)]
        return {
            'features': np.array([], dtype=np.uint8),
            'thresholds': np.array([], dtype=np.uint8),
            'left': np.array([], dtype=np.int32),
            'right': np.array([], dtype=np.int32),
            'leaf_class': np.array([leaf_class], dtype=np.uint8)
        }
    # Random feature subset
    n_features = X.shape[1]
    feat_indices = rng.choice(n_features, size=feature_subset_size, replace=False)
    feat, thr, mask = best_split(X, y, feat_indices, rng)
    if feat is None:
        classes, counts = np.unique(y, return_counts=True)
        leaf_class = classes[np.argmax(counts)]
        return {
            'features': np.array([], dtype=np.uint8),
            'thresholds': np.array([], dtype=np.uint8),
            'left': np.array([], dtype=np.int32),
            'right': np.array([], dtype=np.int32),
            'leaf_class': np.array([leaf_class], dtype=np.uint8)
        }
    # Recurse
    left_tree = build_tree(X[mask], y[mask], max_depth, min_samples_leaf, feature_subset_size, depth + 1, rng)
    right_tree = build_tree(X[~mask], y[~mask], max_depth, min_samples_leaf, feature_subset_size, depth + 1, rng)
    # Combine: current node + left subtree + right subtree
    left_nodes = len(left_tree['leaf_class'])
    right_nodes = len(right_tree['leaf_class'])
    features = np.concatenate(([feat], left_tree['features'], right_tree['features']))
    thresholds = np.concatenate(([thr], left_tree['thresholds'], right_tree['thresholds']))
    # left child index = 1 (next node), right child index = 1 + left_nodes
    left_arr = np.concatenate(([1], left_tree['left'] + 1, right_tree['left'] + 1 + left_nodes))
    right_arr = np.concatenate(([1 + left_nodes], left_tree['right'] + 1, right_tree['right'] + 1 + left_nodes))
    leaf_class = np.concatenate(([255], left_tree['leaf_class'], right_tree['leaf_class']))
    return {
        'features': features.astype(np.uint8),
        'thresholds': thresholds.astype(np.uint8),
        'left': left_arr.astype(np.int32),
        'right': right_arr.astype(np.int32),
        'leaf_class': leaf_class.astype(np.uint8)
    }


def build_forest(X, y, n_trees, max_depth, min_samples_leaf, feature_subset_size, rng):
    """Build a random forest (list of trees)."""
    n_samples = X.shape[0]
    trees = []
    for _ in range(n_trees):
        # Bootstrap sample
        idx = rng.integers(0, n_samples, size=n_samples)
        X_boot, y_boot = X[idx], y[idx]
        tree = build_tree(X_boot, y_boot, max_depth, min_samples_leaf, feature_subset_size, 0, rng)
        trees.append(tree)
    return trees


def evaluate_forest(trees, X, y):
    """Compute accuracy of forest on (X, y)."""
    n_samples = X.shape[0]
    votes = np.zeros((n_samples, 10), dtype=np.int32)
    for tree in trees:
        features = tree['features']
        thresholds = tree['thresholds']
        left = tree['left']
        right = tree['right']
        leaf_class = tree['leaf_class']
        # Traverse all samples simultaneously? For simplicity, loop samples.
        for i in range(n_samples):
            node = 0
            while leaf_class[node] == 255:
                feat = features[node]
                thr = thresholds[node]
                if X[i, feat] <= thr:
                    node = left[node]
                else:
                    node = right[node]
            votes[i, leaf_class[node]] += 1
    preds = np.argmax(votes, axis=1)
    return np.mean(preds == y)


def serialize_forest(trees, n_features, n_classes):
    """Serialize forest to bytes: magic, n_trees, then per tree: n_nodes, features, thresholds, left, right, leaf_class."""
    MAGIC = 0x52465354  # 'RFST'
    out = struct.pack("<II", MAGIC, len(trees))
    for tree in trees:
        n_nodes = len(tree['leaf_class'])
        out += struct.pack("<I", n_nodes)
        out += tree['features'].tobytes()
        out += tree['thresholds'].tobytes()
        out += tree['left'].tobytes()
        out += tree['right'].tobytes()
        out += tree['leaf_class'].tobytes()
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_samples, n_features = X.shape
    n_classes = card["classes"]

    rng = np.random.default_rng(SEED)

    # Split into train and tuning (last 20% in file order)
    tune_size = n_samples // 5
    train_size = n_samples - tune_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_tune, y_tune = X[train_size:], y[train_size:]

    feature_subset_size = max(1, int(n_features * FEATURE_SUBSET_RATIO))

    best_acc = -1.0
    best_params = None
    best_forest = None

    # Grid search on tuning split
    for n_trees in N_TREES_GRID:
        for max_depth in MAX_DEPTH_GRID:
            forest = build_forest(X_train, y_train, n_trees, max_depth, MIN_SAMPLES_LEAF, feature_subset_size, rng)
            acc = evaluate_forest(forest, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (n_trees, max_depth)
                best_forest = forest

    # Retrain on full data with best params
    n_trees, max_depth = best_params
    rng = np.random.default_rng(SEED)  # reset for reproducibility
    final_forest = build_forest(X, y, n_trees, max_depth, MIN_SAMPLES_LEAF, feature_subset_size, rng)

    # Serialize and write
    weights = serialize_forest(final_forest, n_features, n_classes)
    open(sys.argv[4], "wb").write(weights)


if __name__ == "__main__":
    main()
```

```c
/* Random Forest inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the majority-vote class (first max wins). */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 8) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x52465354) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    uint32_t n_trees = wb[4] | wb[5] << 8 | wb[6] << 16 | (uint32_t)wb[7] << 24;
    if (n_trees == 0 || n_trees > 1000) { fprintf(stderr, "bad weights: n_trees out of range\n"); return 3; }

    unsigned char *p = wb + 8;
    unsigned char *end = wb + nw;

    // We'll store tree data in arrays of pointers for each tree.
    // Since we don't know total nodes upfront, we parse sequentially and keep offsets.
    // For simplicity, we parse each tree and store its arrays in a struct.
    typedef struct {
        uint32_t n_nodes;
        const uint8_t *features;
        const uint8_t *thresholds;
        const int32_t *left;
        const int32_t *right;
        const uint8_t *leaf_class;
    } Tree;
    Tree *trees = calloc(n_trees, sizeof(Tree));
    if (!trees) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (uint32_t t = 0; t < n_trees; t++) {
        if (p + 4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t n_nodes = p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24;
        p += 4;
        if (n_nodes == 0 || n_nodes > 100000) { fprintf(stderr, "bad weights: n_nodes out of range\n"); return 3; }
        size_t need = (size_t)n_nodes * (1 + 1 + 4 + 4 + 1); // feature, threshold, left, right, leaf_class
        if (p + need > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t].n_nodes = n_nodes;
        trees[t].features = p; p += n_nodes;
        trees[t].thresholds = p; p += n_nodes;
        trees[t].left = (const int32_t *)p; p += n_nodes * 4;
        trees[t].right = (const int32_t *)p; p += n_nodes * 4;
        trees[t].leaf_class = p; p += n_nodes;
    }
    if (p != end) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    // We assume 10 classes (fixed by task). Use local array for votes.
    int votes[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        // We don't know d here, but we can check against first tree's expected features? Not stored.
        // We'll trust the data.
        memset(votes, 0, sizeof(votes));
        for (uint32_t t = 0; t < n_trees; t++) {
            const Tree *tr = &trees[t];
            uint32_t node = 0;
            while (node < tr->n_nodes && tr->leaf_class[node] == 255) {
                uint8_t feat = tr->features[node];
                uint8_t thr = tr->thresholds[node];
                if (xi[feat] <= thr) {
                    node = tr->left[node];
                } else {
                    node = tr->right[node];
                }
            }
            if (node >= tr->n_nodes) { fprintf(stderr, "tree traversal error\n"); return 3; }
            uint8_t cls = tr->leaf_class[node];
            if (cls < 10) votes[cls]++;
        }
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (votes[c] > votes[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(wb);
    free(trees);
    if (x) free(x);
    return 0;
}
```