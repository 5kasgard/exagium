```hypothesis
Family: label-tree (single deterministic decision tree, max depth 6, min leaf 5).
Why: 64 ordinal pixels (0..16) are highly structured; a shallow tree captures local interactions that linear models miss, while staying extremely fast at inference (≤6 comparisons). The UCI optdigits task is known to be tree-friendly.
Failure signal: dev accuracy ≤ label-linear (0.9613) or inference cost ≥ label-linear (5058 Ir/item).
```
```python
"""label-tree: single CART decision tree for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Hyperparameters (max_depth, min_leaf) chosen on last 20% of train.in (file order),
then final tree trained on all train.in.
Weights format: int32 magic 0x54524545, n_nodes, d, C, root_idx(0);
then per node (12 bytes): uint8 feature, uint8 threshold, int32 left, int32 right, uint8 leaf_class, uint8 pad[3].
Leaf: left==-1, feature/threshold ignored, leaf_class valid.
Internal: left>=0, feature in 0..d-1, threshold in 0..15 (split: x[feature] <= threshold).
"""
import json, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
MAX_DEPTH_GRID = [3, 4, 5, 6, 7, 8]
MIN_LEAF_GRID = [1, 2, 5, 10]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, min_leaf):
    n, d = X.shape
    best_gini = 1.0
    best_feat = -1
    best_thr = -1
    for feat in range(d):
        vals = X[:, feat]
        # possible thresholds: 0..15 (since pixel max 16, split <= thr)
        for thr in range(16):
            mask = vals <= thr
            nL = np.count_nonzero(mask)
            nR = n - nL
            if nL < min_leaf or nR < min_leaf:
                continue
            gL = gini(y[mask])
            gR = gini(y[~mask])
            weighted = (nL * gL + nR * gR) / n
            if weighted < best_gini:
                best_gini = weighted
                best_feat = feat
                best_thr = thr
    return best_feat, best_thr, best_gini

class Node:
    __slots__ = ("feature", "threshold", "left", "right", "leaf_class")
    def __init__(self, feature=0, threshold=0, left=-1, right=-1, leaf_class=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_class = leaf_class

def build_tree(X, y, depth, max_depth, min_leaf, nodes):
    # stopping criteria
    if depth == max_depth or len(y) <= min_leaf or np.all(y == y[0]):
        leaf_class = int(np.bincount(y).argmax())
        node = Node(leaf_class=leaf_class)
        nodes.append(node)
        return len(nodes) - 1
    feat, thr, _ = best_split(X, y, min_leaf)
    if feat == -1:  # no valid split
        leaf_class = int(np.bincount(y).argmax())
        node = Node(leaf_class=leaf_class)
        nodes.append(node)
        return len(nodes) - 1
    mask = X[:, feat] <= thr
    left_idx = build_tree(X[mask], y[mask], depth+1, max_depth, min_leaf, nodes)
    right_idx = build_tree(X[~mask], y[~mask], depth+1, max_depth, min_leaf, nodes)
    node = Node(feature=feat, threshold=thr, left=left_idx, right=right_idx)
    nodes.append(node)
    return len(nodes) - 1

def predict_tree(nodes, x):
    idx = 0
    while True:
        node = nodes[idx]
        if node.left == -1:
            return node.leaf_class
        if x[node.feature] <= node.threshold:
            idx = node.left
        else:
            idx = node.right

def evaluate(nodes, X, y):
    correct = 0
    for i in range(len(y)):
        if predict_tree(nodes, X[i]) == y[i]:
            correct += 1
    return correct / len(y)

def write_weights(nodes, d, C, path):
    n_nodes = len(nodes)
    # header: magic, n_nodes, d, C, root_idx(0)
    header = struct.pack("<5i", 0x54524545, n_nodes, d, C, 0)
    body = bytearray()
    for node in nodes:
        # feature, threshold, left, right, leaf_class, pad[3]
        body += struct.pack("<BBiiiB3x", node.feature, node.threshold, node.left, node.right, node.leaf_class)
    open(path, "wb").write(header + body)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = len(y)
    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_acc = -1.0
    best_params = None
    best_nodes = None

    for max_depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_LEAF_GRID:
            nodes = []
            build_tree(X_train, y_train, 0, max_depth, min_leaf, nodes)
            acc = evaluate(nodes, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_nodes = nodes

    # retrain on full data with best params
    max_depth, min_leaf = best_params
    final_nodes = []
    build_tree(X, y, 0, max_depth, min_leaf, final_nodes)
    write_weights(final_nodes, d, C, sys.argv[4])

if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte class.
 * Weights format: int32 magic 0x54524545, n_nodes, d, C, root_idx;
 * then n_nodes * 12 bytes: uint8 feature, uint8 threshold, int32 left, int32 right, uint8 leaf_class, uint8 pad[3].
 * Leaf: left == -1. Internal: split on x[feature] <= threshold.
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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

typedef struct {
    uint8_t feature;
    uint8_t threshold;
    int32_t left;
    int32_t right;
    uint8_t leaf_class;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_nodes = hd[1], d = hd[2], C = hd[3], root = hd[4];
    if (n_nodes <= 0 || d != 64 || C != 10 || root != 0) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 20 + (size_t)n_nodes * 12;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 20);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int idx = 0;
        while (1) {
            Node nd = nodes[idx];
            if (nd.left == -1) {
                put_rec(stdout, &nd.leaf_class, 1);
                break;
            }
            if (xi[nd.feature] <= nd.threshold) idx = nd.left;
            else idx = nd.right;
        }
    }
    return 0;
}
```