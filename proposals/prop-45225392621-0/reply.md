```hypothesis
We propose a "label-tree" family: a single CART decision tree trained directly on the 14 raw byte features (0–255) with binary splits, Gini impurity, max depth 5, and min_samples_leaf 10. The tree captures non-linear feature interactions that a linear model misses, while inference requires only a handful of integer comparisons per record – far fewer instructions than label-linear's dense dot products. Hyperparameters (depth, min_samples_leaf) are chosen on the last 20% tuning split, then the tree is retrained on all training data. If dev accuracy does not exceed label-linear's 0.8524 or if inference cost (Ir/item) exceeds label-linear's 719, the hypothesis fails.
```

```python
"""label-tree: single CART decision tree on raw byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports a flat binary tree: magic 0x54524545, n_features, max_depth, n_nodes,
then n_nodes * 16-byte records (is_leaf, feature_idx, threshold, left, right, class_label)."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
DEPTH_GRID = [3, 4, 5, 6]
MIN_LEAF_GRID = [5, 10, 20, 50]
MIN_GAIN = 1e-7


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class TreeNode:
    __slots__ = ("is_leaf", "class_label", "feature_idx", "threshold", "left", "right", "idx")
    def __init__(self, is_leaf, class_label=None, feature_idx=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.class_label = class_label
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.idx = -1


def gini_impurity(counts):
    # counts: array of shape (2,)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - (p * p).sum()


def best_split(X, y, indices, min_leaf):
    """Return (feature_idx, threshold, gain) or (None, None, -1) if no valid split."""
    n = len(indices)
    if n < 2 * min_leaf:
        return None, None, -1.0
    parent_counts = np.bincount(y[indices], minlength=2)
    parent_gini = gini_impurity(parent_counts)
    best_gain = -1.0
    best_feat = None
    best_thresh = None

    for feat in range(X.shape[1]):
        # Histogram of labels per feature value (0..255)
        hist = np.zeros((256, 2), dtype=np.int32)
        np.add.at(hist, (X[indices, feat], y[indices]), 1)
        cum = hist.cumsum(axis=0)
        total = cum[-1]
        left_counts = cum[:-1]          # (255, 2)
        right_counts = total - left_counts
        left_sum = left_counts.sum(axis=1)
        right_sum = right_counts.sum(axis=1)
        # Valid splits: both sides have at least min_leaf samples
        valid = (left_sum >= min_leaf) & (right_sum >= min_leaf)
        if not valid.any():
            continue
        left_p = left_counts / left_sum[:, None]
        right_p = right_counts / right_sum[:, None]
        left_gini = 1.0 - (left_p * left_p).sum(axis=1)
        right_gini = 1.0 - (right_p * right_p).sum(axis=1)
        weighted_gini = (left_sum * left_gini + right_sum * right_gini) / (left_sum + right_sum)
        gain = parent_gini - weighted_gini
        gain[~valid] = -1.0
        local_best = np.argmax(gain)
        if gain[local_best] > best_gain:
            best_gain = gain[local_best]
            best_feat = feat
            best_thresh = local_best  # threshold is the feature value (0..254)
    if best_gain < MIN_GAIN:
        return None, None, -1.0
    return best_feat, best_thresh, best_gain


def build_tree(X, y, indices, depth, max_depth, min_leaf):
    # Stopping conditions
    if len(indices) < min_leaf or depth == max_depth:
        label = int(np.bincount(y[indices], minlength=2).argmax())
        return TreeNode(True, class_label=label)
    if np.all(y[indices] == y[indices[0]]):
        return TreeNode(True, class_label=int(y[indices[0]]))

    feat, thresh, gain = best_split(X, y, indices, min_leaf)
    if feat is None or gain < MIN_GAIN:
        label = int(np.bincount(y[indices], minlength=2).argmax())
        return TreeNode(True, class_label=label)

    left_mask = X[indices, feat] <= thresh
    left_idx = indices[left_mask]
    right_idx = indices[~left_mask]
    if len(left_idx) < min_leaf or len(right_idx) < min_leaf:
        label = int(np.bincount(y[indices], minlength=2).argmax())
        return TreeNode(True, class_label=label)

    left_node = build_tree(X, y, left_idx, depth + 1, max_depth, min_leaf)
    right_node = build_tree(X, y, right_idx, depth + 1, max_depth, min_leaf)
    return TreeNode(False, feature_idx=feat, threshold=thresh, left=left_node, right=right_node)


def flatten_tree(root):
    """Pre-order traversal to assign indices and produce flat list."""
    nodes = []
    def assign(node):
        node.idx = len(nodes)
        nodes.append(node)
        if not node.is_leaf:
            assign(node.left)
            assign(node.right)
    assign(root)
    return nodes


def evaluate_tree(root, X, y, indices):
    correct = 0
    for i in indices:
        node = root
        while not node.is_leaf:
            if X[i, node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        if node.class_label == y[i]:
            correct += 1
    return correct / len(indices)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20% in file order
    tune_start = int(0.8 * n)
    train_idx = np.arange(tune_start)
    tune_idx = np.arange(tune_start, n)

    best_acc = -1.0
    best_params = None
    best_tree = None

    for max_depth in DEPTH_GRID:
        for min_leaf in MIN_LEAF_GRID:
            tree = build_tree(X, y, train_idx, 0, max_depth, min_leaf)
            acc = evaluate_tree(tree, X, y, tune_idx)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_tree = tree

    # Retrain on full data with best params
    max_depth, min_leaf = best_params
    final_tree = build_tree(X, y, np.arange(n), 0, max_depth, min_leaf)
    flat_nodes = flatten_tree(final_tree)

    # Write weights.bin
    # Header: magic(4), n_features(1), max_depth(1), n_nodes(2), reserved(4) = 12? Wait, we need 16 bytes for alignment.
    # Let's do: magic(4), n_features(1), max_depth(1), n_nodes(2), reserved(4) = 12, but we'll pad to 16 with extra 4 bytes.
    # Actually the example uses 16-byte header (4 ints). We'll follow: 4 int32.
    magic = 0x54524545  # 'TREE'
    n_features = X.shape[1]
    n_nodes = len(flat_nodes)
    header = struct.pack("<4i", magic, n_features, max_depth, n_nodes)
    # Node: 16 bytes = is_leaf(1), feature_idx(1), threshold(1), pad(1), left(4), right(4), class_label(1), pad(3)
    node_fmt = "<BBBBiiB3x"
    node_bytes = bytearray()
    for node in flat_nodes:
        if node.is_leaf:
            node_bytes += struct.pack(node_fmt, 1, 0, 0, 0, 0, 0, node.class_label)
        else:
            node_bytes += struct.pack(node_fmt, 0, node.feature_idx, node.threshold, 0, node.left.idx, node.right.idx, 0)
    open(sys.argv[4], "wb").write(header + node_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (raw features). Output: 1 byte class label. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
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

typedef struct __attribute__((packed)) {
    uint32_t magic;
    int32_t n_features;
    int32_t max_depth;
    int32_t n_nodes;
} Header;

typedef struct __attribute__((packed)) {
    uint8_t is_leaf;
    uint8_t feature_idx;
    uint8_t threshold;
    uint8_t pad1;
    int32_t left;
    int32_t right;
    uint8_t class_label;
    uint8_t pad2[3];
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < (long)sizeof(Header)) { fprintf(stderr, "weights too small\n"); return 3; }
    Header *hd = (Header *)wb;
    if (hd->magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    if (hd->n_features != 14) { fprintf(stderr, "expected 14 features\n"); return 3; }
    if (hd->n_nodes <= 0) { fprintf(stderr, "no nodes\n"); return 3; }
    long expected = sizeof(Header) + (long)hd->n_nodes * (long)sizeof(Node);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + sizeof(Header));
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)hd->n_features) { fprintf(stderr, "item %ld has %u bytes, expected %d\n", i, k, hd->n_features); return 3; }
        int node_idx = 0;
        while (1) {
            Node *nd = &nodes[node_idx];
            if (nd->is_leaf) {
                put_rec(stdout, &nd->class_label, 1);
                break;
            }
            uint8_t val = xi[nd->feature_idx];
            if (val <= nd->threshold) node_idx = nd->left;
            else node_idx = nd->right;
            if (node_idx < 0 || node_idx >= hd->n_nodes) { fprintf(stderr, "invalid child index\n"); return 3; }
        }
    }
    return 0;
}
```