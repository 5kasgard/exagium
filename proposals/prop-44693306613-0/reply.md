```hypothesis
Family: label-rf (random forest with equality splits on all 14 byte features).  
Reason: The Adult dataset mixes ordinal (quantile-binned) and categorical columns; a small forest of shallow trees can capture non-linear feature interactions that a linear model misses, while inference stays fast (e.g., 10 trees of depth ≤5 → ~50 node visits per item).  
Failure: If dev accuracy ≤ 0.8524 (label-linear) or inference instructions per item exceed label-linear by a large margin.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
N_TREES_OPTIONS = [5, 10, 20]
MAX_DEPTH_OPTIONS = [3, 4, 5]
MIN_SAMPLES_LEAF_OPTIONS = [5, 10, 20]

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

class TreeNode:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'leaf_class')
    def __init__(self, feature=-1, threshold=-1, left=None, right=None, leaf_class=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_class = leaf_class
    def is_leaf(self):
        return self.left is None and self.right is None

def gini_impurity(y, idx):
    if len(idx) == 0:
        return 0.0
    counts = np.bincount(y[idx])
    p = counts / len(idx)
    return 1.0 - np.sum(p * p)

def build_tree_rec(X, y, indices, depth, max_depth, min_samples_leaf, rng):
    n = len(indices)
    if n < min_samples_leaf or depth == max_depth:
        leaf_class = np.bincount(y[indices]).argmax()
        return TreeNode(leaf_class=leaf_class)
    if np.all(y[indices] == y[indices[0]]):
        return TreeNode(leaf_class=y[indices[0]])

    d = X.shape[1]
    n_features = max(1, int(math.sqrt(d)))
    features = rng.choice(d, size=n_features, replace=False)

    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    best_left = None
    best_right = None

    for feat in features:
        vals = np.unique(X[indices, feat])
        for val in vals:
            left_mask = X[indices, feat] == val
            left_idx = indices[left_mask]
            right_idx = indices[~left_mask]
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue
            gini_left = gini_impurity(y, left_idx)
            gini_right = gini_impurity(y, right_idx)
            weighted_gini = (len(left_idx) * gini_left + len(right_idx) * gini_right) / n
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feat = feat
                best_thresh = val
                best_left = left_idx
                best_right = right_idx

    if best_feat == -1:
        leaf_class = np.bincount(y[indices]).argmax()
        return TreeNode(leaf_class=leaf_class)

    left_child = build_tree_rec(X, y, best_left, depth+1, max_depth, min_samples_leaf, rng)
    right_child = build_tree_rec(X, y, best_right, depth+1, max_depth, min_samples_leaf, rng)
    return TreeNode(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

def tree_to_arrays(root):
    """Flatten tree in post-order: children before parent. Returns (node_list, root_index)."""
    arr = []
    def flatten(node):
        if node.is_leaf():
            arr.append((node.feature, node.threshold, -1, -1, node.leaf_class))
            return len(arr) - 1
        left_idx = flatten(node.left)
        right_idx = flatten(node.right)
        arr.append((node.feature, node.threshold, left_idx, right_idx, 0))
        return len(arr) - 1
    root_idx = flatten(root)
    return arr, root_idx

def train_forest(X, y, n_trees, max_depth, min_samples_leaf, rng):
    n_samples = X.shape[0]
    trees = []
    tree_roots = []
    for _ in range(n_trees):
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        tree_root = build_tree_rec(X, y, boot_idx, 0, max_depth, min_samples_leaf, rng)
        tree_arr, root_idx = tree_to_arrays(tree_root)
        trees.append(tree_arr)
        tree_roots.append(root_idx)
    return trees, tree_roots

def predict_forest(trees, tree_roots, X):
    n = X.shape[0]
    votes = np.zeros((n, 2), dtype=np.int32)
    for tree_arr, root_idx in zip(trees, tree_roots):
        for i in range(n):
            idx = root_idx
            while True:
                feat, thresh, left, right, leaf_class = tree_arr[idx]
                if left == -1:
                    votes[i, leaf_class] += 1
                    break
                if X[i, feat] == thresh:
                    idx = left
                else:
                    idx = right
    return np.argmax(votes, axis=1)

def serialize_forest(trees, tree_roots, num_features, num_classes, output_path):
    all_nodes = []
    tree_info = []  # (global_root_idx, num_nodes)
    for tree_arr, root_idx in zip(trees, tree_roots):
        global_offset = len(all_nodes)
        global_root = global_offset + root_idx
        all_nodes.extend(tree_arr)
        tree_info.append((global_root, len(tree_arr)))

    num_trees = len(trees)
    magic = 0x464F5245  # "FORE" in little-endian
    header = struct.pack("<5I", magic, num_trees, 0, num_features, num_classes)
    tree_info_bytes = b''.join(struct.pack("<2I", root, n_nodes) for root, n_nodes in tree_info)

    node_bytes = bytearray()
    for feat, thresh, left, right, leaf_class in all_nodes:
        node_bytes.append(feat & 0xFF)
        node_bytes.append(thresh & 0xFF)
        node_bytes.extend(b'\x00\x00')          # padding to align left/right
        node_bytes.extend(struct.pack("<i", left))
        node_bytes.extend(struct.pack("<i", right))
        node_bytes.append(leaf_class & 0xFF)
        node_bytes.extend(b'\x00\x00\x00')      # padding to 16 bytes

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(tree_info_bytes)
        f.write(node_bytes)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning (validation)
    n = X.shape[0]
    split = int(0.8 * n)
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    best_acc = -1.0
    best_params = None
    best_forest = None

    for n_trees in N_TREES_OPTIONS:
        for max_depth in MAX_DEPTH_OPTIONS:
            for min_samples_leaf in MIN_SAMPLES_LEAF_OPTIONS:
                trees, tree_roots = train_forest(X_train, y_train, n_trees, max_depth, min_samples_leaf, rng)
                preds = predict_forest(trees, tree_roots, X_val)
                acc = np.mean(preds == y_val)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_trees, max_depth, min_samples_leaf)
                    best_forest = (trees, tree_roots)

    # Retrain on full dataset with best hyperparameters
    n_trees, max_depth, min_samples_leaf = best_params
    trees, tree_roots = train_forest(X, y, n_trees, max_depth, min_samples_leaf, rng)
    serialize_forest(trees, tree_roots, d, C, sys.argv[4])

if __name__ == "__main__":
    main()
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma pack(push, 1)
typedef struct {
    uint8_t feature;
    uint8_t threshold;
    uint16_t pad0;
    int32_t left;
    int32_t right;
    uint8_t leaf_class;
    uint8_t pad1[3];
} Node;
#pragma pack(pop)

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
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }

    uint32_t *hdr = (uint32_t *)w;
    if (hdr[0] != 0x464F5245) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t num_trees = hdr[1];
    uint32_t max_depth = hdr[2]; (void)max_depth;
    uint32_t num_features = hdr[3];
    uint32_t num_classes = hdr[4];
    if (num_features != 14 || num_classes != 2) { fprintf(stderr, "unexpected feature/class count\n"); return 3; }

    uint32_t *tree_info = (uint32_t *)(w + 20);
    long expected_nodes = 0;
    for (uint32_t t = 0; t < num_trees; t++) {
        expected_nodes += tree_info[2*t + 1];
    }
    long expected_size = 20 + (long)num_trees * 8 + expected_nodes * 16;
    if (nw != expected_size) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    Node *nodes = (Node *)(w + 20 + num_trees * 8);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != num_features) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, num_features); return 3; }

        int votes[2] = {0, 0};
        for (uint32_t t = 0; t < num_trees; t++) {
            uint32_t root_idx = tree_info[2*t];
            uint32_t idx = root_idx;
            while (1) {
                Node *nd = &nodes[idx];
                if (nd->left == -1) {
                    votes[nd->leaf_class]++;
                    break;
                }
                if (xi[nd->feature] == nd->threshold)
                    idx = nd->left;
                else
                    idx = nd->right;
            }
        }
        unsigned char best = (votes[1] > votes[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```