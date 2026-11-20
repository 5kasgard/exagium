```hypothesis
Family: label-rf (small random forest, 10 trees, max depth 8).
Why: 64 ordinal pixel features (0–16) with non-linear decision boundaries suit axis-aligned splits; 10 shallow trees add little inference cost but capture interactions label-linear misses.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or inference > 2× label-linear ns/item.
```
```python
"""label-rf: small random forest for fixed-size byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x52460001, n_trees, n_features, n_classes, max_depth;
per tree: n_nodes, then nodes (feature:uint8, threshold:uint8, left:int32, right:int32, leaf_class:uint8, is_leaf:uint8, pad:uint16)."""
import json, struct, sys
import numpy as np

SEED = 1
N_TREES = 10
MAX_DEPTH = 8
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ("feature", "threshold", "left", "right", "leaf_class", "is_leaf")
    def __init__(self, feature=0, threshold=0, left=-1, right=-1, leaf_class=0, is_leaf=False):
        self.feature = feature; self.threshold = threshold
        self.left = left; self.right = right
        self.leaf_class = leaf_class; self.is_leaf = is_leaf

def gini(y, n_classes):
    if len(y) == 0: return 0.0
    counts = np.bincount(y, minlength=n_classes)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, n_classes, feature_indices, rng):
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    n_samples = len(y)
    for f in feature_indices:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        # try midpoints between unique values
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        for t in thresholds:
            left_mask = vals <= t
            right_mask = ~left_mask
            n_left = np.sum(left_mask)
            n_right = n_samples - n_left
            if n_left < MIN_SAMPLES_LEAF or n_right < MIN_SAMPLES_LEAF:
                continue
            g = (n_left * gini(y[left_mask], n_classes) + n_right * gini(y[right_mask], n_classes)) / n_samples
            if g < best_gini:
                best_gini = g
                best_feat = f
                best_thresh = t
    return best_feat, best_thresh, best_gini

def build_tree(X, y, n_classes, n_features, depth, rng):
    node = Node()
    # stopping criteria
    if depth >= MAX_DEPTH or len(y) < MIN_SAMPLES_SPLIT or gini(y, n_classes) == 0.0:
        node.is_leaf = True
        node.leaf_class = int(np.bincount(y, minlength=n_classes).argmax())
        return node
    # feature subset: sqrt(n_features) rounded, at least 1
    n_sub = max(1, int(np.sqrt(n_features)))
    feat_indices = rng.choice(n_features, size=n_sub, replace=False)
    feat, thresh, _ = best_split(X, y, n_classes, feat_indices, rng)
    if feat == -1:
        node.is_leaf = True
        node.leaf_class = int(np.bincount(y, minlength=n_classes).argmax())
        return node
    node.feature = feat
    node.threshold = int(np.round(thresh))  # threshold as uint8
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    node.left = 0  # placeholder
    node.right = 0  # placeholder
    # build children recursively, but we need to collect nodes in a list
    return node

def collect_nodes(node, nodes):
    """Convert tree to flat array with child indices."""
    idx = len(nodes)
    nodes.append(node)
    if not node.is_leaf:
        node.left = collect_nodes(node.left, nodes) if isinstance(node.left, Node) else node.left
        node.right = collect_nodes(node.right, nodes) if isinstance(node.right, Node) else node.right
    return idx

def train_tree(X, y, n_classes, n_features, rng):
    # bootstrap sample
    n = len(y)
    idx = rng.integers(0, n, n)
    Xb, yb = X[idx], y[idx]
    # build recursively
    def build_rec(Xsub, ysub, depth):
        node = Node()
        if depth >= MAX_DEPTH or len(ysub) < MIN_SAMPLES_SPLIT or gini(ysub, n_classes) == 0.0:
            node.is_leaf = True
            node.leaf_class = int(np.bincount(ysub, minlength=n_classes).argmax())
            return node
        n_sub = max(1, int(np.sqrt(n_features)))
        feat_indices = rng.choice(n_features, size=n_sub, replace=False)
        feat, thresh, _ = best_split(Xsub, ysub, n_classes, feat_indices, rng)
        if feat == -1:
            node.is_leaf = True
            node.leaf_class = int(np.bincount(ysub, minlength=n_classes).argmax())
            return node
        node.feature = feat
        node.threshold = int(np.round(thresh))
        left_mask = Xsub[:, feat] <= thresh
        right_mask = ~left_mask
        node.left = build_rec(Xsub[left_mask], ysub[left_mask], depth + 1)
        node.right = build_rec(Xsub[right_mask], ysub[right_mask], depth + 1)
        return node
    root = build_rec(Xb, yb, 0)
    # flatten
    nodes = []
    def flatten(n):
        i = len(nodes)
        nodes.append(n)
        if not n.is_leaf:
            n.left = flatten(n.left)
            n.right = flatten(n.right)
        return i
    flatten(root)
    return nodes

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_samples, n_features = X.shape
    n_classes = card["classes"]
    rng = np.random.default_rng(SEED)
    trees = []
    for t in range(N_TREES):
        trees.append(train_tree(X, y, n_classes, n_features, rng))
    # write weights
    magic = 0x52460001
    out = struct.pack("<4i", magic, N_TREES, n_features, n_classes)
    for nodes in trees:
        out += struct.pack("<i", len(nodes))
        for n in nodes:
            out += struct.pack("<BBiiBBH", n.feature, n.threshold & 0xFF, n.left, n.right, n.leaf_class, 1 if n.is_leaf else 0, 0)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-rf inference: ./infer weights.bin in.rec N
 * Random forest with 10 trees, max depth 8. Majority vote over trees. */
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
    uint8_t is_leaf;
    uint16_t pad;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x52460001) { fprintf(stderr, "bad magic\n"); return 3; }
    int n_trees = hdr[1], n_features = hdr[2], n_classes = hdr[3];
    if (n_trees <= 0 || n_trees > 100 || n_features <= 0 || n_features > 255 || n_classes <= 0 || n_classes > 256) {
        fprintf(stderr, "bad header\n"); return 3;
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    const unsigned char *p = wb + 16;
    Node **trees = malloc(n_trees * sizeof(Node *));
    int *tree_sizes = malloc(n_trees * sizeof(int));
    for (int t = 0; t < n_trees; t++) {
        if (p + 4 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        int n_nodes = *(int32_t *)p; p += 4;
        if (n_nodes <= 0 || p + (long)n_nodes * 16 > wb + nw) { fprintf(stderr, "bad tree size\n"); return 3; }
        trees[t] = (Node *)p;
        tree_sizes[t] = n_nodes;
        p += (long)n_nodes * 16;
    }
    if (p != wb + nw) { fprintf(stderr, "extra bytes in weights\n"); return 3; }
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_features); return 3; }
        int votes[256] = {0};
        for (int t = 0; t < n_trees; t++) {
            const Node *nodes = trees[t];
            int node_idx = 0;
            while (1) {
                const Node *n = &nodes[node_idx];
                if (n->is_leaf) {
                    votes[n->leaf_class]++;
                    break;
                }
                uint8_t val = xi[n->feature];
                if (val <= n->threshold) node_idx = n->left;
                else node_idx = n->right;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < n_classes; c++) if (votes[c] > votes[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```