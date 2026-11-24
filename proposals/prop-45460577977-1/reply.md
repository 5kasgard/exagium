```hypothesis
We propose a Random Forest family (label-rf) with 10 trees of max depth 10. Random forests are strong on tabular data like optdigits, often beating linear models and single trees. Inference is fast (integer comparisons only) and memory-light. If dev accuracy does not exceed label-linear (0.9613), it suggests the small dataset and low-dimensional features favor simpler linear boundaries over ensemble methods.
```

```python
"""Random Forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports trees as integer decision nodes."""
import json, struct, sys
import numpy as np

SEED = 1
N_TREES = 10
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 2
MAX_FEATURES = 8  # sqrt(64)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ("is_leaf", "feature", "threshold", "class_", "left", "right")
    def __init__(self, is_leaf, feature=0, threshold=0, class_=0, left=-1, right=-1):
        self.is_leaf = is_leaf
        self.feature = feature
        self.threshold = threshold
        self.class_ = class_
        self.left = left
        self.right = right

def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def build_tree(X, y, depth, rng, feature_idx):
    n = len(y)
    if depth >= MAX_DEPTH or n < MIN_SAMPLES_SPLIT or gini(y) == 0.0:
        cls = np.bincount(y).argmax()
        return Node(True, class_=cls)
    # choose random subset of features
    chosen = rng.choice(feature_idx, size=min(MAX_FEATURES, len(feature_idx)), replace=False)
    best_gini = 1.0
    best_feat = -1
    best_thr = -1
    for feat in chosen:
        vals = X[:, feat]
        # possible thresholds: midpoints between sorted unique values
        uniq = np.unique(vals)
        if len(uniq) == 1:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        for thr in thresholds:
            left_mask = vals <= thr
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            g = (left_mask.sum() * gini(y[left_mask]) + right_mask.sum() * gini(y[right_mask])) / n
            if g < best_gini:
                best_gini = g
                best_feat = feat
                best_thr = thr
    if best_feat == -1:
        cls = np.bincount(y).argmax()
        return Node(True, class_=cls)
    left_mask = X[:, best_feat] <= best_thr
    right_mask = ~left_mask
    left_node = build_tree(X[left_mask], y[left_mask], depth + 1, rng, feature_idx)
    right_node = build_tree(X[right_mask], y[right_mask], depth + 1, rng, feature_idx)
    # flatten: we will assign indices later
    return Node(False, feature=best_feat, threshold=int(best_thr), left=left_node, right=right_node)

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
    return nodes

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    rng = np.random.default_rng(SEED)
    feature_idx = np.arange(d)
    all_trees_nodes = []
    for t in range(N_TREES):
        # bootstrap sample
        idx = rng.choice(n, size=n, replace=True)
        Xb, yb = X[idx], y[idx]
        root = build_tree(Xb, yb, 0, rng, feature_idx)
        nodes = flatten_tree(root)
        all_trees_nodes.append(nodes)
    # serialize
    # magic: 0x52463100 ("RF1\0")
    out = struct.pack("<Iiii", 0x52463100, N_TREES, C, d)
    for nodes in all_trees_nodes:
        out += struct.pack("<i", len(nodes))
        for nd in nodes:
            # Node: uint8 is_leaf, uint8 feature, uint8 threshold, uint8 class_, int32 left, int32 right
            out += struct.pack("<BBBBii", nd.is_leaf, nd.feature, nd.threshold, nd.class_, nd.left, nd.right)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Random Forest inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the majority vote class. */
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
    uint8_t feature;
    uint8_t threshold;
    uint8_t class_;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x52463100) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t n_trees = *(int32_t*)(wb + 4);
    int32_t n_classes = *(int32_t*)(wb + 8);
    int32_t d = *(int32_t*)(wb + 12);
    if (n_trees <= 0 || n_classes <= 0 || n_classes > 256 || d <= 0) {
        fprintf(stderr, "bad header values\n"); return 3;
    }
    unsigned char *ptr = wb + 16;
    unsigned char *end = wb + nw;
    Node **trees = malloc(n_trees * sizeof(Node*));
    int32_t *tree_sizes = malloc(n_trees * sizeof(int32_t));
    for (int t = 0; t < n_trees; t++) {
        if (ptr + 4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        int32_t n_nodes = *(int32_t*)ptr; ptr += 4;
        if (n_nodes <= 0) { fprintf(stderr, "empty tree\n"); return 3; }
        size_t need = (size_t)n_nodes * sizeof(Node);
        if (ptr + need > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t] = (Node*)ptr;
        tree_sizes[t] = n_nodes;
        ptr += need;
    }
    if (ptr != end) { fprintf(stderr, "extra bytes in weights\n"); return 3; }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    int *votes = calloc(n_classes, sizeof(int));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(votes, 0, n_classes * sizeof(int));
        for (int t = 0; t < n_trees; t++) {
            Node *nodes = trees[t];
            int idx = 0;
            while (1) {
                Node *nd = &nodes[idx];
                if (nd->is_leaf) {
                    votes[nd->class_]++;
                    break;
                }
                uint8_t val = xi[nd->feature];
                if (val <= nd->threshold) idx = nd->left;
                else idx = nd->right;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < n_classes; c++) if (votes[c] > votes[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```