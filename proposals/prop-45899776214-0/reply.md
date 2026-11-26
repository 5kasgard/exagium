```hypothesis
Propose label-tree: a single decision tree (max depth 5) trained with Gini impurity on the Adult dataset.
Decision trees naturally handle mixed ordinal/categorical features and capture interactions that linear models miss.
A shallow tree should exceed label-linear's 0.8524 dev accuracy while being extremely fast at inference (only a few branches per item).
Failure: if dev accuracy ≤ 0.8524 or inference cost (Ir/item) exceeds label-linear's 719.
```

```python
"""label-tree: single decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Hyperparameters (max_depth, min_samples_leaf) chosen on last 20% of train.in (tuning split).
Weights format: magic 0x54524545, num_nodes (uint32), then nodes in pre-order.
Each node: 44 bytes = uint8 is_leaf, uint8 feature, uint8 split_type, uint8 threshold,
           uint8 bitmask[32], int32 left, int32 right.
Leaf: is_leaf=1, class in feature, rest zero.
Internal: is_leaf=0, split_type 0=ordinal (<=threshold), 1=categorical (bitmask)."""
import json, math, struct, sys
from array import array

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
MAX_DEPTH_OPTS = [3, 4, 5, 6]
MIN_LEAF_OPTS = [10, 20, 50, 100]

MAGIC = 0x54524545  # "TREE" in little-endian
NODE_SIZE = 44  # bytes per node


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)


def find_best_split(X, y, indices, feature_types, min_leaf):
    """Return (feature, split_type, threshold, bitmask, gain) or None."""
    best_gain = -1.0
    best = None
    parent_gini = gini(y[indices])
    n = len(indices)
    for j, ftype in enumerate(feature_types):
        vals = X[indices, j]
        if ftype == "ordinal":
            # try thresholds between unique values
            uniq = np.unique(vals)
            if len(uniq) <= 1:
                continue
            for th in uniq[:-1]:
                left_mask = vals <= th
                right_mask = ~left_mask
                nl, nr = left_mask.sum(), right_mask.sum()
                if nl < min_leaf or nr < min_leaf:
                    continue
                gain = parent_gini - (nl / n) * gini(y[indices[left_mask]]) - (nr / n) * gini(y[indices[right_mask]])
                if gain > best_gain:
                    best_gain = gain
                    best = (j, 0, int(th), None)
        else:  # categorical
            # compute class 1 probability per category
            cat_probs = {}
            for cat in range(256):
                mask = (vals == cat)
                if np.any(mask):
                    cat_probs[cat] = np.mean(y[indices[mask]])
            if len(cat_probs) <= 1:
                continue
            # sort categories by prob
            sorted_cats = sorted(cat_probs.keys(), key=lambda c: cat_probs[c])
            # try splits between adjacent sorted categories
            left_cats = set()
            for i in range(len(sorted_cats) - 1):
                left_cats.add(sorted_cats[i])
                left_mask = np.isin(vals, list(left_cats))
                right_mask = ~left_mask
                nl, nr = left_mask.sum(), right_mask.sum()
                if nl < min_leaf or nr < min_leaf:
                    continue
                gain = parent_gini - (nl / n) * gini(y[indices[left_mask]]) - (nr / n) * gini(y[indices[right_mask]])
                if gain > best_gain:
                    best_gain = gain
                    # build bitmask
                    bitmask = bytearray(32)
                    for c in left_cats:
                        bitmask[c >> 3] |= 1 << (c & 7)
                    best = (j, 1, 0, bytes(bitmask))
    if best_gain <= 0:
        return None
    return best + (best_gain,)


class Node:
    __slots__ = ("is_leaf", "class_label", "feature", "split_type", "threshold", "bitmask", "left", "right")

    def __init__(self, is_leaf, class_label=None, feature=None, split_type=None, threshold=None, bitmask=None):
        self.is_leaf = is_leaf
        self.class_label = class_label
        self.feature = feature
        self.split_type = split_type
        self.threshold = threshold
        self.bitmask = bitmask
        self.left = None
        self.right = None

    def write(self, buf, nodes_list):
        """Write this node and its children in pre-order. Return index of this node in nodes_list."""
        idx = len(nodes_list)
        nodes_list.append(self)
        if not self.is_leaf:
            self.left.write(buf, nodes_list)
            self.right.write(buf, nodes_list)
        return idx

    def to_bytes(self, left_idx, right_idx):
        """Serialize node to 44 bytes."""
        arr = bytearray(NODE_SIZE)
        arr[0] = 1 if self.is_leaf else 0
        if self.is_leaf:
            arr[1] = self.class_label
        else:
            arr[1] = self.feature
            arr[2] = self.split_type
            arr[3] = self.threshold
            if self.split_type == 1:
                arr[4:36] = self.bitmask
        struct.pack_into("<ii", arr, 36, left_idx, right_idx)
        return bytes(arr)


def build_tree(X, y, indices, feature_types, max_depth, min_leaf, depth=0):
    # stopping criteria
    if depth >= max_depth or len(indices) < 2 * min_leaf:
        return make_leaf(y, indices)
    unique_classes = np.unique(y[indices])
    if len(unique_classes) == 1:
        return make_leaf(y, indices)
    split = find_best_split(X, y, indices, feature_types, min_leaf)
    if split is None:
        return make_leaf(y, indices)
    feat, stype, thresh, bitmask, _ = split
    vals = X[indices, feat]
    if stype == 0:
        left_mask = vals <= thresh
    else:
        left_mask = np.isin(vals, np.where(np.frombuffer(bitmask, dtype=np.uint8) > 0)[0])  # simplified
    right_mask = ~left_mask
    left_idx = indices[left_mask]
    right_idx = indices[right_mask]
    node = Node(False, feature=feat, split_type=stype, threshold=thresh, bitmask=bitmask)
    node.left = build_tree(X, y, left_idx, feature_types, max_depth, min_leaf, depth + 1)
    node.right = build_tree(X, y, right_idx, feature_types, max_depth, min_leaf, depth + 1)
    return node


def make_leaf(y, indices):
    # majority class
    vals, counts = np.unique(y[indices], return_counts=True)
    cls = vals[np.argmax(counts)]
    return Node(True, class_label=int(cls))


def evaluate_tree(root, X, y, indices):
    correct = 0
    for idx in indices:
        node = root
        while not node.is_leaf:
            v = X[idx, node.feature]
            if node.split_type == 0:
                if v <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                if (node.bitmask[v >> 3] >> (v & 7)) & 1:
                    node = node.left
                else:
                    node = node.right
        if node.class_label == y[idx]:
            correct += 1
    return correct / len(indices)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    feature_types = card["input"]["types"]
    n = len(X)
    split_idx = int(0.8 * n)
    train_idx = np.arange(split_idx)
    tune_idx = np.arange(split_idx, n)

    rng = np.random.default_rng(SEED)
    # Shuffle training indices for any internal randomness (though our tree is deterministic)
    rng.shuffle(train_idx)

    best_acc = -1.0
    best_params = None
    best_tree = None

    for max_depth in MAX_DEPTH_OPTS:
        for min_leaf in MIN_LEAF_OPTS:
            tree = build_tree(X, y, train_idx, feature_types, max_depth, min_leaf)
            acc = evaluate_tree(tree, X, y, tune_idx)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_tree = tree

    # Retrain on full dataset with best params
    max_depth, min_leaf = best_params
    full_idx = np.arange(n)
    rng.shuffle(full_idx)
    final_tree = build_tree(X, y, full_idx, feature_types, max_depth, min_leaf)

    # Serialize tree
    nodes_list = []
    final_tree.write(None, nodes_list)  # populate nodes_list in pre-order
    # Now assign left/right indices
    def assign_indices(node, idx):
        if node.is_leaf:
            return idx + 1
        left_idx = idx + 1
        right_idx = assign_indices(node.left, left_idx)
        node.left_idx = left_idx
        node.right_idx = right_idx
        return assign_indices(node.right, right_idx)
    assign_indices(final_tree, 0)

    # Write weights.bin
    out = bytearray()
    out += struct.pack("<I", MAGIC)
    out += struct.pack("<I", len(nodes_list))
    for node in nodes_list:
        left = getattr(node, "left_idx", 0)
        right = getattr(node, "right_idx", 0)
        out += node.to_bytes(left, right)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (fixed). Output: one byte class (0 or 1). */
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

#pragma pack(push, 1)
typedef struct {
    uint8_t is_leaf;
    uint8_t feature;      // leaf: class label; internal: feature index
    uint8_t split_type;   // 0 ordinal, 1 categorical
    uint8_t threshold;    // for ordinal
    uint8_t bitmask[32];  // for categorical
    int32_t left;         // child indices in node array
    int32_t right;
} Node;
#pragma pack(pop)

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = *(uint32_t*)wb;
    if (magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t num_nodes = *(uint32_t*)(wb + 4);
    if (nw != 8 + num_nodes * 44) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node*)(wb + 8);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }
        int node_idx = 0;
        while (1) {
            Node *nd = &nodes[node_idx];
            if (nd->is_leaf) {
                unsigned char cls = nd->feature;
                put_rec(stdout, &cls, 1);
                break;
            }
            uint8_t v = xi[nd->feature];
            if (nd->split_type == 0) { // ordinal
                node_idx = (v <= nd->threshold) ? nd->left : nd->right;
            } else { // categorical
                int bit = (nd->bitmask[v >> 3] >> (v & 7)) & 1;
                node_idx = bit ? nd->left : nd->right;
            }
        }
    }
    return 0;
}
```