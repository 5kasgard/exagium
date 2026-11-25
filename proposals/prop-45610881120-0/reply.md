```hypothesis
Family: label-tree (single decision tree with ordinal/categorical splits).  
Why: Tabular data with mixed types often benefits from non-linear decision boundaries; a shallow tree captures interactions that linear models miss, while remaining faster and smaller than GBDT.  
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear by >2×, the approach does not justify its complexity.
```

```python
"""label-tree: single decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports a compact binary tree with per-feature split logic."""
import json, math, struct, sys
import numpy as np

SEED = 1
TUNE_FRAC = 0.2
MAX_DEPTH_OPTS = [3, 4, 5, 6]
MIN_LEAF_OPTS = [5, 10, 20, 50]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)

class Node:
    __slots__ = ("feature", "threshold", "left", "right", "leaf_class", "is_leaf")
    def __init__(self, feature=0, threshold=0, left=-1, right=-1, leaf_class=255):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_class = leaf_class
        self.is_leaf = (leaf_class != 255)

def build_tree(X, y, feature_types, indices, depth, max_depth, min_leaf):
    n = len(indices)
    if n == 0:
        return Node(leaf_class=0)
    classes = y[indices]
    cnt = np.bincount(classes, minlength=2)
    if cnt[0] == 0 or cnt[1] == 0 or depth == max_depth or n <= min_leaf:
        return Node(leaf_class=int(np.argmax(cnt)))
    best_gain = -1.0
    best_feat = -1
    best_thresh = 0
    total_gini = gini(cnt)
    for f in range(X.shape[1]):
        vals = X[indices, f]
        if feature_types[f] == 0:  # ordinal
            # histogram of value vs class
            hist = np.zeros((256, 2), dtype=np.int32)
            np.add.at(hist, (vals, classes), 1)
            left_cnt = np.zeros(2, dtype=np.int32)
            right_cnt = hist.sum(axis=0)
            for t in range(255):
                left_cnt += hist[t]
                right_cnt -= hist[t]
                if left_cnt.sum() == 0 or right_cnt.sum() == 0:
                    continue
                gain = total_gini - (left_cnt.sum() * gini(left_cnt) + right_cnt.sum() * gini(right_cnt)) / n
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = t
        else:  # categorical
            hist = np.zeros((256, 2), dtype=np.int32)
            np.add.at(hist, (vals, classes), 1)
            for v in range(256):
                if hist[v].sum() == 0:
                    continue
                left_cnt = hist[v]
                right_cnt = cnt - left_cnt
                if right_cnt.sum() == 0:
                    continue
                gain = total_gini - (left_cnt.sum() * gini(left_cnt) + right_cnt.sum() * gini(right_cnt)) / n
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = v
    if best_gain <= 0 or best_feat == -1:
        return Node(leaf_class=int(np.argmax(cnt)))
    # split
    left_idx = indices[X[indices, best_feat] <= best_thresh] if feature_types[best_feat] == 0 else indices[X[indices, best_feat] == best_thresh]
    right_idx = indices[X[indices, best_feat] > best_thresh] if feature_types[best_feat] == 0 else indices[X[indices, best_feat] != best_thresh]
    if len(left_idx) == 0 or len(right_idx) == 0:
        return Node(leaf_class=int(np.argmax(cnt)))
    left_node = build_tree(X, y, feature_types, left_idx, depth+1, max_depth, min_leaf)
    right_node = build_tree(X, y, feature_types, right_idx, depth+1, max_depth, min_leaf)
    return Node(feature=best_feat, threshold=best_thresh, left=left_node, right=right_node)

def flatten_tree(root):
    nodes = []
    def dfs(node):
        idx = len(nodes)
        nodes.append(node)
        if not node.is_leaf:
            node.left = dfs(node.left)
            node.right = dfs(node.right)
        return idx
    dfs(root)
    # convert to flat arrays
    flat = []
    for node in nodes:
        if node.is_leaf:
            flat.append((node.feature, node.threshold, -1, -1, node.leaf_class))
        else:
            flat.append((node.feature, node.threshold, node.left, node.right, 255))
    return flat

def evaluate(tree_flat, X, y, feature_types):
    correct = 0
    for i in range(len(X)):
        node_idx = 0
        while True:
            feat, thresh, left, right, leaf = tree_flat[node_idx]
            if leaf != 255:
                if leaf == y[i]:
                    correct += 1
                break
            if feature_types[feat] == 0:
                node_idx = left if X[i, feat] <= thresh else right
            else:
                node_idx = left if X[i, feat] == thresh else right
    return correct / len(X)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    feature_types = np.array([0 if t == "ordinal" else 1 for t in card["input"]["types"]], dtype=np.uint8)
    n = len(X)
    tune_size = int(n * TUNE_FRAC)
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    best_acc = -1
    best_params = None
    best_tree = None
    for max_depth in MAX_DEPTH_OPTS:
        for min_leaf in MIN_LEAF_OPTS:
            root = build_tree(X, y, feature_types, train_idx, 0, max_depth, min_leaf)
            flat = flatten_tree(root)
            acc = evaluate(flat, X[tune_idx], y[tune_idx], feature_types)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_tree = flat
    # retrain on all data
    root = build_tree(X, y, feature_types, np.arange(n), 0, *best_params)
    flat = flatten_tree(root)
    # serialize
    magic = 0x54524545  # 'TREE'
    d = X.shape[1]
    num_nodes = len(flat)
    header = struct.pack("<III", magic, num_nodes, d) + feature_types.tobytes()
    node_bytes = b""
    for feat, thresh, left, right, leaf in flat:
        node_bytes += struct.pack("<BBiiB", feat, thresh, left, right, leaf)
    open(sys.argv[4], "wb").write(header + node_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N */
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

typedef struct __attribute__((packed)) {
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t*)wb;
    if (magic != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t num_nodes = *(uint32_t*)(wb + 4);
    uint32_t d = *(uint32_t*)(wb + 8);
    if (d != 14) { fprintf(stderr, "unexpected feature count\n"); return 3; }
    unsigned char *feature_types = wb + 12;
    Node *nodes = (Node*)(feature_types + d);
    long expected = 12 + d + (long)num_nodes * sizeof(Node);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        int node_idx = 0;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.leaf_class != 255) {
                put_rec(stdout, &nd.leaf_class, 1);
                break;
            }
            uint8_t val = xi[nd.feature];
            if (feature_types[nd.feature] == 0) { // ordinal
                node_idx = (val <= nd.threshold) ? nd.left : nd.right;
            } else { // categorical
                node_idx = (val == nd.threshold) ? nd.left : nd.right;
            }
        }
    }
    return 0;
}
```