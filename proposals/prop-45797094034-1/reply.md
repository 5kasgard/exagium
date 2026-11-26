```hypothesis
Propose "label-rf": a random forest of shallow decision trees. Optdigits' 64 ordinal pixel features (0–16) are well-suited to axis-aligned splits; forests often beat linear models on such tabular data while remaining fast at inference (tree traversal only). Hyperparameters (n_trees, max_depth, min_samples_leaf) are chosen on the tuning split (last 20% of train). If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds ~10k Ir/item, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning
N_TREES_GRID = [10, 20, 30]
MAX_DEPTH_GRID = [5, 10, 15]
MIN_SAMPLES_LEAF_GRID = [1, 5, 10]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class DecisionTree:
    def __init__(self, max_depth, min_samples_leaf, rng):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rng = rng
        # Node arrays (will be lists during building, converted to arrays at end)
        self.left = []
        self.right = []
        self.feature = []
        self.threshold = []
        self.leaf_class = []  # valid only for leaves
        self.is_leaf = []

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p * p)

    def _best_split(self, X, y, feature_indices):
        n_samples, n_features = X.shape
        best_gini = 1.0
        best_feat = -1
        best_thresh = -1
        parent_gini = self._gini(y)
        # For each feature, try thresholds from 0 to 254 (since values 0..255)
        # But we can optimize by only trying thresholds that exist in the data.
        for feat in feature_indices:
            vals = X[:, feat]
            unique_vals = np.unique(vals)
            if len(unique_vals) <= 1:
                continue
            # Try midpoints between consecutive unique values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) // 2
            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (left_mask.sum() * gini_left + right_mask.sum() * gini_right) / n_samples
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feat = feat
                    best_thresh = thresh
        if best_feat == -1 or best_gini >= parent_gini:
            return -1, -1, -1
        return best_feat, best_thresh, best_gini

    def _build(self, X, y, depth, feature_indices):
        node_idx = len(self.is_leaf)
        self.left.append(-1)
        self.right.append(-1)
        self.feature.append(-1)
        self.threshold.append(0)
        self.leaf_class.append(0)
        self.is_leaf.append(0)

        # Check stopping criteria
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or self._gini(y) == 0.0:
            self.is_leaf[node_idx] = 1
            # majority class (smallest class on tie)
            vals, counts = np.unique(y, return_counts=True)
            self.leaf_class[node_idx] = vals[np.argmax(counts)]
            return node_idx

        best_feat, best_thresh, _ = self._best_split(X, y, feature_indices)
        if best_feat == -1:
            self.is_leaf[node_idx] = 1
            vals, counts = np.unique(y, return_counts=True)
            self.leaf_class[node_idx] = vals[np.argmax(counts)]
            return node_idx

        self.feature[node_idx] = best_feat
        self.threshold[node_idx] = best_thresh
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        self.is_leaf[node_idx] = 0
        left_child = self._build(X[left_mask], y[left_mask], depth + 1, feature_indices)
        right_child = self._build(X[right_mask], y[right_mask], depth + 1, feature_indices)
        self.left[node_idx] = left_child
        self.right[node_idx] = right_child
        return node_idx

    def fit(self, X, y, feature_indices):
        self._build(X, y, 0, feature_indices)
        # Convert lists to numpy arrays for serialization
        self.left = np.array(self.left, dtype=np.int32)
        self.right = np.array(self.right, dtype=np.int32)
        self.feature = np.array(self.feature, dtype=np.int16)
        self.threshold = np.array(self.threshold, dtype=np.uint8)
        self.leaf_class = np.array(self.leaf_class, dtype=np.uint8)
        self.is_leaf = np.array(self.is_leaf, dtype=np.uint8)

    def predict_one(self, x):
        node = 0
        while not self.is_leaf[node]:
            feat = self.feature[node]
            thresh = self.threshold[node]
            if x[feat] <= thresh:
                node = self.left[node]
            else:
                node = self.right[node]
        return self.leaf_class[node]

    def serialize(self):
        # Returns bytes for this tree
        n_nodes = len(self.is_leaf)
        out = struct.pack("<i", n_nodes)
        out += self.left.tobytes()
        out += self.right.tobytes()
        out += self.feature.tobytes()
        out += self.threshold.tobytes()
        out += self.leaf_class.tobytes()
        out += self.is_leaf.tobytes()
        return out

    @staticmethod
    def deserialize(buf, offset):
        tree = DecisionTree(0, 0, None)  # dummy, we'll fill arrays
        n_nodes = struct.unpack_from("<i", buf, offset)[0]
        offset += 4
        tree.left = np.frombuffer(buf, dtype=np.int32, count=n_nodes, offset=offset).copy()
        offset += n_nodes * 4
        tree.right = np.frombuffer(buf, dtype=np.int32, count=n_nodes, offset=offset).copy()
        offset += n_nodes * 4
        tree.feature = np.frombuffer(buf, dtype=np.int16, count=n_nodes, offset=offset).copy()
        offset += n_nodes * 2
        tree.threshold = np.frombuffer(buf, dtype=np.uint8, count=n_nodes, offset=offset).copy()
        offset += n_nodes * 1
        tree.leaf_class = np.frombuffer(buf, dtype=np.uint8, count=n_nodes, offset=offset).copy()
        offset += n_nodes * 1
        tree.is_leaf = np.frombuffer(buf, dtype=np.uint8, count=n_nodes, offset=offset).copy()
        offset += n_nodes * 1
        return tree, offset

class RandomForest:
    def __init__(self, n_trees, max_depth, min_samples_leaf, rng):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rng = rng
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        feature_indices = np.arange(n_features)
        self.trees = []
        for t in range(self.n_trees):
            # Bootstrap sample
            idx = self.rng.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[idx], y[idx]
            # Random subset of features for each tree (sqrt(n_features))
            n_sub = max(1, int(math.sqrt(n_features)))
            feat_sub = self.rng.choice(feature_indices, n_sub, replace=False)
            tree = DecisionTree(self.max_depth, self.min_samples_leaf, self.rng)
            tree.fit(X_boot, y_boot, feat_sub)
            self.trees.append(tree)

    def predict_one(self, x):
        votes = np.zeros(10, dtype=np.int32)
        for tree in self.trees:
            c = tree.predict_one(x)
            votes[c] += 1
        return np.argmax(votes)  # ties broken by smallest class

    def serialize(self):
        out = struct.pack("<iii", 0x52464F52, self.n_trees, self.max_depth)  # magic 'RFOR'
        out += struct.pack("<i", self.min_samples_leaf)
        for tree in self.trees:
            out += tree.serialize()
        return out

    @staticmethod
    def deserialize(buf):
        magic, n_trees, max_depth = struct.unpack_from("<iii", buf, 0)
        if magic != 0x52464F52:
            raise ValueError("Bad magic")
        offset = 12
        min_samples_leaf = struct.unpack_from("<i", buf, offset)[0]
        offset += 4
        rf = RandomForest(n_trees, max_depth, min_samples_leaf, None)
        rf.trees = []
        for _ in range(n_trees):
            tree, offset = DecisionTree.deserialize(buf, offset)
            rf.trees.append(tree)
        return rf

def evaluate(rf, X, y):
    correct = 0
    for i in range(len(X)):
        if rf.predict_one(X[i]) == y[i]:
            correct += 1
    return correct / len(X)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n_total = len(X)
    # Tuning split: last 20%
    n_tune = int(0.2 * n_total)
    n_train = n_total - n_tune
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]

    best_acc = -1.0
    best_params = None
    rng_base = np.random.default_rng(SEED)

    for n_trees in N_TREES_GRID:
        for max_depth in MAX_DEPTH_GRID:
            for min_samples_leaf in MIN_SAMPLES_LEAF_GRID:
                # Use a deterministic seed for each config
                seed = rng_base.integers(2**31)
                rng = np.random.default_rng(seed)
                rf = RandomForest(n_trees, max_depth, min_samples_leaf, rng)
                rf.fit(X_train, y_train)
                acc = evaluate(rf, X_tune, y_tune)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_trees, max_depth, min_samples_leaf, seed)

    print(f"Best params: n_trees={best_params[0]}, max_depth={best_params[1]}, min_samples_leaf={best_params[2]}, seed={best_params[3]}, acc={best_acc:.4f}", file=sys.stderr)

    # Train on all data with best params
    final_rng = np.random.default_rng(best_params[3])
    final_rf = RandomForest(best_params[0], best_params[1], best_params[2], final_rng)
    final_rf.fit(X, y)

    # Serialize
    out = final_rf.serialize()
    open(sys.argv[4], "wb").write(out)

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
    int32_t *left;
    int32_t *right;
    int16_t *feature;
    uint8_t *threshold;
    uint8_t *leaf_class;
    uint8_t *is_leaf;
    int32_t n_nodes;
} Tree;

typedef struct {
    int32_t n_trees;
    int32_t max_depth;
    int32_t min_samples_leaf;
    Tree *trees;
} Forest;

static Forest load_forest(const unsigned char *buf, long nw) {
    if (nw < 16) { fprintf(stderr, "weights too small\n"); exit(3); }
    int32_t magic = buf[0] | buf[1] << 8 | buf[2] << 16 | (int32_t)buf[3] << 24;
    if (magic != 0x52464F52) { fprintf(stderr, "bad magic\n"); exit(3); }
    int32_t n_trees = *(int32_t*)(buf + 4);
    int32_t max_depth = *(int32_t*)(buf + 8);
    int32_t min_samples_leaf = *(int32_t*)(buf + 12);
    size_t offset = 16;
    Forest f;
    f.n_trees = n_trees;
    f.max_depth = max_depth;
    f.min_samples_leaf = min_samples_leaf;
    f.trees = calloc(n_trees, sizeof(Tree));
    for (int t = 0; t < n_trees; t++) {
        if (offset + 4 > (size_t)nw) { fprintf(stderr, "truncated tree header\n"); exit(3); }
        int32_t n_nodes = *(int32_t*)(buf + offset);
        offset += 4;
        f.trees[t].n_nodes = n_nodes;
        size_t need = n_nodes * (4 + 4 + 2 + 1 + 1 + 1);
        if (offset + need > (size_t)nw) { fprintf(stderr, "truncated tree data\n"); exit(3); }
        f.trees[t].left = (int32_t*)(buf + offset); offset += n_nodes * 4;
        f.trees[t].right = (int32_t*)(buf + offset); offset += n_nodes * 4;
        f.trees[t].feature = (int16_t*)(buf + offset); offset += n_nodes * 2;
        f.trees[t].threshold = (uint8_t*)(buf + offset); offset += n_nodes * 1;
        f.trees[t].leaf_class = (uint8_t*)(buf + offset); offset += n_nodes * 1;
        f.trees[t].is_leaf = (uint8_t*)(buf + offset); offset += n_nodes * 1;
    }
    return f;
}

static uint8_t predict_tree(const Tree *tree, const unsigned char *x) {
    int32_t node = 0;
    while (!tree->is_leaf[node]) {
        int16_t feat = tree->feature[node];
        uint8_t thresh = tree->threshold[node];
        if (x[feat] <= thresh)
            node = tree->left[node];
        else
            node = tree->right[node];
    }
    return tree->leaf_class[node];
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    Forest forest = load_forest(wbuf, nw);
    unsigned char *xbuf = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(xbuf, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int votes[10] = {0};
        for (int t = 0; t < forest.n_trees; t++) {
            uint8_t c = predict_tree(&forest.trees[t], xi);
            votes[c]++;
        }
        uint8_t best = 0;
        for (int c = 1; c < 10; c++)
            if (votes[c] > votes[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```