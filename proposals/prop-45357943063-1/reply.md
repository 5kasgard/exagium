```hypothesis
Family: label-rf (random forest). Random forests handle non-linear interactions in pixel data well and are robust to noise. With 10 trees of depth 10, they should surpass label-linear (0.9613) and approach label-mlp16 (0.9770) at similar or lower CPU cost. If dev accuracy is below 0.965, the model is insufficiently expressive for this task.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
N_TREES_GRID = [5, 10, 20]
MAX_DEPTH_GRID = [5, 10, 15]
MAX_FEATURES = 8  # sqrt(64)
MIN_SAMPLES_SPLIT = 2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class DecisionTree:
    __slots__ = ("feature", "threshold", "left", "right", "leaf_class", "nodes")
    def __init__(self, max_depth, max_features, min_samples_split, rng):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.rng = rng
        self.nodes = []  # list of dicts, will be converted to arrays

    def fit(self, X, y):
        # X: (n, d) uint8, y: (n,) uint8
        self.nodes = []
        self._build(0, np.arange(len(y)), 0)
        # Convert to flat arrays
        n_nodes = len(self.nodes)
        feature = np.zeros(n_nodes, dtype=np.uint8)
        threshold = np.zeros(n_nodes, dtype=np.uint8)
        left = np.zeros(n_nodes, dtype=np.int32)
        right = np.zeros(n_nodes, dtype=np.int32)
        leaf_class = np.zeros(n_nodes, dtype=np.uint8)
        for i, node in enumerate(self.nodes):
            feature[i] = node["feature"]
            threshold[i] = node["threshold"]
            left[i] = node["left"]
            right[i] = node["right"]
            leaf_class[i] = node["leaf_class"]
        return feature, threshold, left, right, leaf_class

    def _build(self, node_idx, indices, depth):
        # Determine if leaf
        unique_classes, counts = np.unique(y[indices], return_counts=True)
        majority = unique_classes[np.argmax(counts)]
        if depth >= self.max_depth or len(indices) < self.min_samples_split or len(unique_classes) == 1:
            self.nodes.append({
                "feature": 255, "threshold": 255, "left": -1, "right": -1, "leaf_class": majority
            })
            return

        # Choose random subset of features
        n_features = X.shape[1]
        feat_subset = self.rng.choice(n_features, size=min(self.max_features, n_features), replace=False)

        best_gini = 1.0
        best_feat = -1
        best_thresh = -1
        best_left = None
        best_right = None

        for feat in feat_subset:
            values = X[indices, feat]
            # Possible thresholds: midpoints between unique sorted values
            uniq = np.unique(values)
            if len(uniq) <= 1:
                continue
            threshes = (uniq[:-1] + uniq[1:]) / 2
            for thresh in threshes:
                left_mask = values <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                # Gini impurity
                n = len(indices)
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                _, counts_left = np.unique(y[indices[left_mask]], return_counts=True)
                _, counts_right = np.unique(y[indices[right_mask]], return_counts=True)
                gini_left = 1.0 - np.sum((counts_left / n_left) ** 2)
                gini_right = 1.0 - np.sum((counts_right / n_right) ** 2)
                gini = (n_left * gini_left + n_right * gini_right) / n
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat
                    best_thresh = thresh
                    best_left = indices[left_mask]
                    best_right = indices[right_mask]

        if best_feat == -1:
            self.nodes.append({
                "feature": 255, "threshold": 255, "left": -1, "right": -1, "leaf_class": majority
            })
            return

        # Create internal node
        left_idx = len(self.nodes) + 1  # temporary, will be updated after left subtree built
        # We need to know the index of left child before building it? We'll build left first, then right.
        # Instead, we can build recursively and use the current length.
        # We'll push a placeholder and update later.
        node_dict = {"feature": best_feat, "threshold": int(best_thresh), "left": -1, "right": -1, "leaf_class": 255}
        self.nodes.append(node_dict)
        cur_idx = len(self.nodes) - 1
        # Build left
        left_child_idx = len(self.nodes)
        self._build(left_child_idx, best_left, depth + 1)
        # Build right
        right_child_idx = len(self.nodes)
        self._build(right_child_idx, best_right, depth + 1)
        # Update current node
        self.nodes[cur_idx]["left"] = left_child_idx
        self.nodes[cur_idx]["right"] = right_child_idx

class RandomForest:
    def __init__(self, n_trees, max_depth, max_features, min_samples_split, seed):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees = []  # each tree: (feature, threshold, left, right, leaf_class)

    def fit(self, X, y):
        self.trees = []
        for t in range(self.n_trees):
            rng = np.random.default_rng(self.seed + t * 1000)
            # Bootstrap sample
            n = len(y)
            idx = rng.choice(n, size=n, replace=True)
            X_boot, y_boot = X[idx], y[idx]
            tree = DecisionTree(self.max_depth, self.max_features, self.min_samples_split, rng)
            tree_data = tree.fit(X_boot, y_boot)
            self.trees.append(tree_data)

    def predict(self, X):
        # X: (n, d)
        n = X.shape[0]
        votes = np.zeros((n, 10), dtype=np.int32)
        for tree in self.trees:
            feature, threshold, left, right, leaf_class = tree
            # Vectorized traversal for all samples
            node_idx = np.zeros(n, dtype=np.int32)
            while True:
                # Check if leaf
                is_leaf = (left[node_idx] == -1)
                if np.all(is_leaf):
                    for i in range(n):
                        votes[i, leaf_class[node_idx[i]]] += 1
                    break
                # For non-leaf nodes, split
                feat = feature[node_idx]
                thresh = threshold[node_idx]
                # Since features differ per sample, we must loop? But we can vectorize by grouping by node.
                # Given small n (max 2867) and small trees, we can loop over samples for simplicity.
                # However, we need speed for tuning. We'll implement a simple loop per sample.
                pass
        # We'll implement a per-sample prediction for simplicity in training (tuning).
        # For final inference, C code will do per-sample.
        preds = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            votes_i = np.zeros(10, dtype=np.int32)
            for tree in self.trees:
                feature, threshold, left, right, leaf_class = tree
                node = 0
                while left[node] != -1:
                    if X[i, feature[node]] <= threshold[node]:
                        node = left[node]
                    else:
                        node = right[node]
                votes_i[leaf_class[node]] += 1
            preds[i] = np.argmax(votes_i)
        return preds

def evaluate(y_true, y_pred):
    return np.mean(y_true == y_pred)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = len(y)
    # Split: last 20% for tuning
    n_train = int(0.8 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]

    best_acc = -1
    best_params = None
    for n_trees in N_TREES_GRID:
        for max_depth in MAX_DEPTH_GRID:
            rf = RandomForest(n_trees, max_depth, MAX_FEATURES, MIN_SAMPLES_SPLIT, SEED)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_tune)
            acc = evaluate(y_tune, preds)
            if acc > best_acc:
                best_acc = acc
                best_params = (n_trees, max_depth)
    print(f"Best params: n_trees={best_params[0]}, max_depth={best_params[1]}, tune_acc={best_acc:.4f}")

    # Train final on all data
    rf_final = RandomForest(best_params[0], best_params[1], MAX_FEATURES, MIN_SAMPLES_SPLIT, SEED)
    rf_final.fit(X, y)

    # Serialize weights.bin
    # Header: magic (0x52464C42), n_trees, max_depth, n_features, n_classes
    magic = 0x52464C42  # "RFBL"
    n_trees, max_depth = best_params
    n_features = X.shape[1]
    n_classes = card["classes"]
    header = struct.pack("<5i", magic, n_trees, max_depth, n_features, n_classes)
    out = bytearray(header)
    for tree in rf_final.trees:
        feature, threshold, left, right, leaf_class = tree
        n_nodes = len(feature)
        out += struct.pack("<i", n_nodes)
        # Each node: feature (uint8), threshold (uint8), left (int32), right (int32), leaf_class (uint8), pad (3 bytes)
        for i in range(n_nodes):
            out += struct.pack("<BBiiiB3x", feature[i], threshold[i], left[i], right[i], leaf_class[i])
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
    uint8_t feature;
    uint8_t threshold;
    int32_t left;
    int32_t right;
    uint8_t leaf_class;
} Node;

typedef struct {
    int32_t n_nodes;
    Node *nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x52464C42) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t n_trees = hd[1];
    int32_t max_depth = hd[2];
    int32_t n_features = hd[3];
    int32_t n_classes = hd[4];
    if (n_features != 64 || n_classes != 10 || n_trees <= 0 || max_depth <= 0) {
        fprintf(stderr, "bad weights header\n"); return 3;
    }
    unsigned char *p = wb + 20;
    long remaining = nw - 20;
    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (int t = 0; t < n_trees; t++) {
        if (remaining < 4) { fprintf(stderr, "truncated weights\n"); return 3; }
        int32_t n_nodes = *(int32_t *)p;
        p += 4; remaining -= 4;
        if (n_nodes <= 0) { fprintf(stderr, "bad n_nodes\n"); return 3; }
        size_t node_size = 16; // sizeof(Node) with padding
        if (remaining < (long)n_nodes * node_size) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t].n_nodes = n_nodes;
        trees[t].nodes = (Node *)p;
        p += n_nodes * node_size;
        remaining -= n_nodes * node_size;
    }
    if (remaining != 0) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)n_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_features); return 3; }
        int votes[10] = {0};
        for (int t = 0; t < n_trees; t++) {
            Node *nodes = trees[t].nodes;
            int32_t node_idx = 0;
            while (1) {
                Node *nd = &nodes[node_idx];
                if (nd->left == -1) {
                    votes[nd->leaf_class]++;
                    break;
                }
                if (xi[nd->feature] <= nd->threshold) node_idx = nd->left;
                else node_idx = nd->right;
            }
        }
        int best = 0;
        for (int c = 1; c < 10; c++) if (votes[c] > votes[best]) best = c;
        out_byte = (unsigned char)best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```