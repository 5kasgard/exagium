```hypothesis
Family: label-dtree (single decision tree). Optdigits has ordinal pixel features (0-16) with strong spatial structure; a greedy CART tree splits on individual pixels and captures local interactions naturally. Inference is a few branch instructions per item (tree depth ~10-15), far cheaper than the linear model's 64×10 table lookups. If dev accuracy ≤ 0.961 (label-linear's score) or Ir/item ≥ 5000, the hypothesis fails.
```
```python
import json, struct, sys
import numpy as np

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ('is_leaf', 'class_label', 'feature', 'threshold', 'left', 'right', 'idx')
    def __init__(self):
        self.is_leaf = False
        self.class_label = 0
        self.feature = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.idx = -1

def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def build_tree(X, y, max_depth, min_samples_leaf, depth=0):
    node = Node()
    n_samples = len(y)
    if depth >= max_depth or n_samples < 2 * min_samples_leaf:
        node.is_leaf = True
        node.class_label = int(np.bincount(y).argmax())
        return node
    if len(np.unique(y)) == 1:
        node.is_leaf = True
        node.class_label = int(y[0])
        return node
    best_gini = float('inf')
    best_feature = -1
    best_threshold = -1
    for f in range(X.shape[1]):
        for t in range(16):
            mask = X[:, f] <= t
            n_left = np.sum(mask)
            n_right = n_samples - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            gini_left = gini(y[mask])
            gini_right = gini(y[~mask])
            weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = f
                best_threshold = t
    if best_feature == -1:
        node.is_leaf = True
        node.class_label = int(np.bincount(y).argmax())
        return node
    node.is_leaf = False
    node.feature = best_feature
    node.threshold = best_threshold
    mask = X[:, best_feature] <= best_threshold
    node.left = build_tree(X[mask], y[mask], max_depth, min_samples_leaf, depth + 1)
    node.right = build_tree(X[~mask], y[~mask], max_depth, min_samples_leaf, depth + 1)
    return node

def assign_indices(node, idx=0):
    node.idx = idx
    next_idx = idx + 1
    if not node.is_leaf:
        next_idx = assign_indices(node.left, next_idx)
        next_idx = assign_indices(node.right, next_idx)
    return next_idx

def serialize_tree(root, n_nodes, d):
    out = bytearray()
    out += struct.pack('<3i', 0x45455254, d, n_nodes)
    def write_node(node):
        if node.is_leaf:
            out.extend(struct.pack('<5i', 1, node.class_label, 0, 0, 0))
        else:
            out.extend(struct.pack('<5i', 0, node.feature, node.threshold, node.left.idx, node.right.idx))
            write_node(node.left)
            write_node(node.right)
    write_node(root)
    return bytes(out)

def evaluate_tree(root, X, y):
    preds = []
    for xi in X:
        node = root
        while not node.is_leaf:
            if xi[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        preds.append(node.class_label)
    preds = np.array(preds)
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y_all)
    split_idx = int(n * 0.8)
    X_train, X_tune = X_all[:split_idx], X_all[split_idx:]
    y_train, y_tune = y_all[:split_idx], y_all[split_idx:]
    best_acc = -1.0
    best_params = (10, 5)
    for max_depth in [5, 8, 10, 12, 15, 20]:
        for min_samples_leaf in [1, 2, 5, 10]:
            tree = build_tree(X_train, y_train, max_depth, min_samples_leaf)
            n_nodes = assign_indices(tree)
            acc = evaluate_tree(tree, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)
    max_depth, min_samples_leaf = best_params
    final_tree = build_tree(X_all, y_all, max_depth, min_samples_leaf)
    n_nodes = assign_indices(final_tree)
    weights = serialize_tree(final_tree, n_nodes, X_all.shape[1])
    open(sys.argv[4], "wb").write(weights)

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
    int32_t is_leaf;
    int32_t class_or_feature;
    int32_t threshold;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x45455254) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    int d = hdr[1];
    int32_t n_nodes = hdr[2];
    if (nw != 12 + (long)n_nodes * 20) { fprintf(stderr, "bad weights: size mismatch\n"); return 3; }
    Node *nodes = (Node *)(wb + 12);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int32_t node_idx = 0;
        while (1) {
            Node *node = &nodes[node_idx];
            if (node->is_leaf) {
                unsigned char best = (unsigned char)node->class_or_feature;
                put_rec(stdout, &best, 1);
                break;
            } else {
                int feature = node->class_or_feature;
                int threshold = node->threshold;
                if (xi[feature] <= threshold) node_idx = node->left;
                else node_idx = node->right;
            }
        }
    }
    return 0;
}
```