```hypothesis
label-dtree: a single decision tree (max depth 5‑7) trained with Gini impurity on all 14 byte features.
It captures feature interactions that a linear model misses, while inference is just a handful of
integer comparisons per item. If dev accuracy does not exceed label‑linear’s 0.8524 or instruction
cost rises above ~1000 Ir/item, the hypothesis fails.
```

```python
"""label-dtree: deterministic single decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Builds a binary tree with threshold splits on byte features. Hyperparameters (max_depth,
min_samples_leaf) are chosen on the last 20% of train.in (file order), then retrained on all data.
Weights format: magic 0x44545245 ('DTRE'), int32 d, int32 num_nodes, then nodes in preorder.
Each node: uint8 is_leaf; if leaf: uint8 class; else: uint8 feature, uint8 threshold, int32 left, int32 right.
"""
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH_CANDIDATES = [3, 4, 5, 6, 7]
MIN_LEAF_CANDIDATES = [10, 20, 50, 100]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class Node:
    __slots__ = ("is_leaf", "class_label", "feature", "threshold", "left", "right")
    def __init__(self, is_leaf, class_label=None, feature=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.class_label = class_label
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)


def find_best_split(X, y, min_samples_leaf):
    n, d = X.shape
    best_gini = float('inf')
    best_feature = -1
    best_threshold = -1
    best_left_mask = None

    for j in range(d):
        # histogram of class counts per value (0..255)
        hist = np.zeros((256, 2), dtype=np.int64)
        np.add.at(hist, (X[:, j], y), 1)
        cum = hist.cumsum(axis=0)
        total_left = cum[:-1]          # thresholds 0..254
        total_right = cum[-1] - total_left
        valid = (total_left[:, 0] + total_left[:, 1] >= min_samples_leaf) & \
                (total_right[:, 0] + total_right[:, 1] >= min_samples_leaf)
        if not np.any(valid):
            continue
        # Gini for left and right
        left_sum = total_left.sum(axis=1)
        right_sum = total_right.sum(axis=1)
        left_gini = 1.0 - np.sum((total_left / left_sum[:, None]) ** 2, axis=1)
        right_gini = 1.0 - np.sum((total_right / right_sum[:, None]) ** 2, axis=1)
        weighted_gini = (left_sum * left_gini + right_sum * right_gini) / n
        weighted_gini[~valid] = float('inf')
        idx = np.argmin(weighted_gini)
        if weighted_gini[idx] < best_gini:
            best_gini = weighted_gini[idx]
            best_feature = j
            best_threshold = idx
            best_left_mask = X[:, j] <= idx

    if best_feature == -1:
        return None, None, None
    return best_feature, best_threshold, best_left_mask


def build_tree(X, y, depth, max_depth, min_samples_leaf):
    # stopping criteria
    if depth >= max_depth or len(y) <= min_samples_leaf or np.all(y == y[0]):
        majority = np.bincount(y).argmax()
        return Node(is_leaf=True, class_label=int(majority))

    feat, thresh, left_mask = find_best_split(X, y, min_samples_leaf)
    if feat is None:
        majority = np.bincount(y).argmax()
        return Node(is_leaf=True, class_label=int(majority))

    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_leaf)
    right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples_leaf)
    return Node(is_leaf=False, feature=feat, threshold=thresh, left=left, right=right)


def preorder_nodes(root):
    nodes = []
    def traverse(node):
        nodes.append(node)
        if not node.is_leaf:
            traverse(node.left)
            traverse(node.right)
    traverse(root)
    return nodes


def serialize_tree(root, d):
    nodes = preorder_nodes(root)
    num_nodes = len(nodes)
    # assign indices
    for idx, node in enumerate(nodes):
        node._idx = idx
    # compute child indices
    for node in nodes:
        if not node.is_leaf:
            node.left_idx = node.left._idx
            node.right_idx = node.right._idx

    out = bytearray()
    out += struct.pack("<I", 0x44545245)  # magic 'DTRE'
    out += struct.pack("<i", d)
    out += struct.pack("<i", num_nodes)
    for node in nodes:
        out += struct.pack("<B", 1 if node.is_leaf else 0)
        if node.is_leaf:
            out += struct.pack("<B", node.class_label)
        else:
            out += struct.pack("<B B i i", node.feature, node.threshold, node.left_idx, node.right_idx)
    return bytes(out)


def evaluate_tree(root, X):
    preds = np.zeros(len(X), dtype=np.int64)
    for i, x in enumerate(X):
        node = root
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        preds[i] = node.class_label
    return preds


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)

    # tuning split: last 20%
    n_val = n // 5
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    best_acc = -1.0
    best_params = None
    for max_depth in MAX_DEPTH_CANDIDATES:
        for min_leaf in MIN_LEAF_CANDIDATES:
            tree = build_tree(X_train, y_train, 0, max_depth, min_leaf)
            preds = evaluate_tree(tree, X_val)
            acc = np.mean(preds == y_val)
            if acc > best_acc or (acc == best_acc and (max_depth < best_params[0] or (max_depth == best_params[0] and min_leaf < best_params[1]))):
                best_acc = acc
                best_params = (max_depth, min_leaf)

    # retrain on full data
    final_tree = build_tree(X, y, 0, best_params[0], best_params[1])
    weights = serialize_tree(final_tree, d)
    open(sys.argv[4], "wb").write(weights)


if __name__ == "__main__":
    main()
```

```c
/* label-dtree inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (0 or 1).
 * Weights format: magic 0x44545245, int32 d, int32 num_nodes, then nodes in preorder.
 * Node: uint8 is_leaf; if leaf: uint8 class; else: uint8 feature, uint8 threshold, int32 left, int32 right.
 */
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
    uint8_t is_leaf;
    uint8_t class_label;
    uint8_t feature;
    uint8_t threshold;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *wp = wb;

    if (nw < 12) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wp[0] | wp[1] << 8 | wp[2] << 16 | wp[3] << 24;
    wp += 4;
    if (magic != 0x44545245) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }

    int32_t d = wp[0] | wp[1] << 8 | wp[2] << 16 | wp[3] << 24;
    wp += 4;
    int32_t num_nodes = wp[0] | wp[1] << 8 | wp[2] << 16 | wp[3] << 24;
    wp += 4;

    if (d != 14) { fprintf(stderr, "bad weights: d=%d expected 14\n", d); return 3; }
    if (num_nodes <= 0) { fprintf(stderr, "bad weights: num_nodes=%d\n", num_nodes); return 3; }

    Node *nodes = calloc(num_nodes, sizeof(Node));
    if (!nodes) { fprintf(stderr, "alloc failed\n"); exit(3); }

    for (int i = 0; i < num_nodes; i++) {
        if (wp + 1 > wb + nw) { fprintf(stderr, "bad weights: truncated node header\n"); return 3; }
        nodes[i].is_leaf = *wp++;
        if (nodes[i].is_leaf) {
            if (wp + 1 > wb + nw) { fprintf(stderr, "bad weights: truncated leaf\n"); return 3; }
            nodes[i].class_label = *wp++;
        } else {
            if (wp + 10 > wb + nw) { fprintf(stderr, "bad weights: truncated internal\n"); return 3; }
            nodes[i].feature = *wp++;
            nodes[i].threshold = *wp++;
            nodes[i].left = wp[0] | wp[1] << 8 | wp[2] << 16 | (int32_t)wp[3] << 24; wp += 4;
            nodes[i].right = wp[0] | wp[1] << 8 | wp[2] << 16 | (int32_t)wp[3] << 24; wp += 4;
        }
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int node_idx = 0;
        while (1) {
            Node *nd = &nodes[node_idx];
            if (nd->is_leaf) {
                unsigned char out = nd->class_label;
                put_rec(stdout, &out, 1);
                break;
            }
            if (xi[nd->feature] <= nd->threshold)
                node_idx = nd->left;
            else
                node_idx = nd->right;
        }
    }

    free(wb);
    free(nodes);
    if (x) free(x);
    return 0;
}
```