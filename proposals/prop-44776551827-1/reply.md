```hypothesis
Family: label-tree (single CART decision tree). The 8x8 optdigits pixels are ordinal 0..16; a shallow tree can capture non-linear interactions (e.g., strokes) that linear models miss. Tuning max_depth on the last 20% of train.in should yield a sweet spot around depth 8–12, beating label-linear (0.9613) while keeping inference extremely fast (a few dozen integer comparisons per item). If dev accuracy stays ≤0.9613 or inference cost exceeds label-linear, the hypothesis fails.
```
```python
"""label-tree: single CART decision tree for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Tunes max_depth on last 20% of train.in, then trains on all data.
Weights format: magic 0x54524545, int32 num_nodes, num_classes, max_depth; then per node (int32 left, right, feature, threshold, class).
Leaf: left=right=feature=threshold=-1, class=0..C-1. Internal: left/right child indices, feature 0..d-1, threshold 0..16, class=-1.
"""
import json
import struct
import sys
import numpy as np

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ("left", "right", "feature", "threshold", "class_idx", "is_leaf")
    def __init__(self):
        self.left = -1
        self.right = -1
        self.feature = -1
        self.threshold = -1
        self.class_idx = -1
        self.is_leaf = False

def gini(y, classes):
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=classes)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, classes, min_samples_leaf=1):
    n, d = X.shape
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    for feat in range(d):
        vals = X[:, feat]
        # thresholds from 0 to 15 (since max pixel value is 16)
        for thresh in range(16):
            left_mask = vals <= thresh
            right_mask = ~left_mask
            if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                continue
            gini_left = gini(y[left_mask], classes)
            gini_right = gini(y[right_mask], classes)
            weighted = (left_mask.sum() * gini_left + right_mask.sum() * gini_right) / n
            if weighted < best_gini:
                best_gini = weighted
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh, best_gini

def build_tree(X, y, classes, max_depth, min_samples_leaf, depth=0):
    node = Node()
    # stopping criteria
    if depth >= max_depth or len(y) < 2 * min_samples_leaf or gini(y, classes) == 0.0:
        node.is_leaf = True
        node.class_idx = int(np.bincount(y, minlength=classes).argmax())
        return node
    feat, thresh, _ = best_split(X, y, classes, min_samples_leaf)
    if feat == -1:
        node.is_leaf = True
        node.class_idx = int(np.bincount(y, minlength=classes).argmax())
        return node
    node.feature = feat
    node.threshold = thresh
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    node.left = build_tree(X[left_mask], y[left_mask], classes, max_depth, min_samples_leaf, depth + 1)
    node.right = build_tree(X[right_mask], y[right_mask], classes, max_depth, min_samples_leaf, depth + 1)
    return node

def flatten_tree(node, nodes_list):
    idx = len(nodes_list)
    nodes_list.append(node)
    if not node.is_leaf:
        node.left = flatten_tree(node.left, nodes_list)
        node.right = flatten_tree(node.right, nodes_list)
    return idx

def tree_to_bytes(nodes, num_classes, max_depth):
    num_nodes = len(nodes)
    header = struct.pack("<Iiii", 0x54524545, num_nodes, num_classes, max_depth)
    body = bytearray()
    for node in nodes:
        left = node.left if not node.is_leaf else -1
        right = node.right if not node.is_leaf else -1
        feat = node.feature if not node.is_leaf else -1
        thresh = node.threshold if not node.is_leaf else -1
        class_idx = node.class_idx if node.is_leaf else -1
        body += struct.pack("<iiiii", left, right, feat, thresh, class_idx)
    return header + body

def accuracy(tree_nodes, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        idx = 0
        while True:
            node = tree_nodes[idx]
            if node.is_leaf:
                if node.class_idx == yi:
                    correct += 1
                break
            if xi[node.feature] <= node.threshold:
                idx = node.left
            else:
                idx = node.right
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    classes = card["classes"]
    n = X.shape[0]
    # split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_depth = 1
    best_acc = 0.0
    min_samples_leaf = 1  # could tune, but keep simple
    for depth in range(1, 21):
        root = build_tree(X_train, y_train, classes, depth, min_samples_leaf)
        nodes = []
        flatten_tree(root, nodes)
        acc = accuracy(nodes, X_tune, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # retrain on all data with best_depth
    root = build_tree(X, y, classes, best_depth, min_samples_leaf)
    nodes = []
    flatten_tree(root, nodes)
    out_bytes = tree_to_bytes(nodes, classes, best_depth)
    open(sys.argv[4], "wb").write(out_bytes)

if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels). Output record: 1 byte class.
 * Weights format: magic 0x54524545, int32 num_nodes, num_classes, max_depth; then nodes (int32 left, right, feature, threshold, class). */
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
    int32_t class_idx;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    int32_t num_nodes = hd[1];
    int32_t num_classes = hd[2];
    int32_t max_depth = hd[3];
    if (num_classes != 10 || num_nodes <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + (size_t)num_nodes * 5 * sizeof(int32_t);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int32_t idx = 0;
        while (1) {
            Node *nd = &nodes[idx];
            if (nd->left == -1 && nd->right == -1) { // leaf
                unsigned char cls = (unsigned char)nd->class_idx;
                put_rec(stdout, &cls, 1);
                break;
            }
            if (xi[nd->feature] <= nd->threshold) idx = nd->left;
            else idx = nd->right;
        }
    }
    return 0;
}
```