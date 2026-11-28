```hypothesis
Family: label-tree (single deterministic decision tree).
Why: Optdigits has non-linear pixel interactions; a depth-limited tree captures these with few nodes, beating linear models while staying faster than MLPs.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or instructions/item ≥ 8685 (label-mlp16), the tree is not a good trade-off.
```
```python
"""label-tree: single decision tree for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seed), single thread, stdlib + numpy only.
Trains a CART tree (Gini, max_depth=10, min_samples_leaf=5) on raw pixels (0..16).
Hyperparameters chosen on last 20% of train.in (file order), then retrained on all.
Weights format: magic 0x54524545, num_nodes (int32), then per node:
  int32 feature, float32 threshold, int32 left, int32 right, int32 class.
Leaf: feature=-1, threshold=0, left=-1, right=-1, class=0..9."""
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH_CANDIDATES = [5, 6, 7, 8, 9, 10]
MIN_SAMPLES_LEAF_CANDIDATES = [1, 2, 5, 10]
TUNING_FRACTION = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(y):
    """Gini impurity of integer labels y."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)


def best_split(X, y, min_samples_leaf):
    """Find best split (feature, threshold) for node.
    Returns (feature, threshold, left_mask) or (None, None, None) if no split improves."""
    n, d = X.shape
    best_gini = gini(y)
    best_feat = None
    best_thresh = None
    best_left = None
    # For each feature, consider thresholds between sorted unique values
    for feat in range(d):
        vals = X[:, feat]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        # thresholds are midpoints between consecutive unique values
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        for thresh in thresholds:
            left = vals <= thresh
            right = ~left
            if left.sum() < min_samples_leaf or right.sum() < min_samples_leaf:
                continue
            gini_left = gini(y[left])
            gini_right = gini(y[right])
            weighted_gini = (left.sum() * gini_left + right.sum() * gini_right) / n
            if weighted_gini < best_gini - 1e-12:  # strict improvement
                best_gini = weighted_gini
                best_feat = feat
                best_thresh = thresh
                best_left = left
    return best_feat, best_thresh, best_left


def build_tree(X, y, depth, max_depth, min_samples_leaf, nodes):
    """Recursively build tree, appending nodes to list. Returns node index."""
    node_idx = len(nodes)
    # Check stopping criteria
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2 * min_samples_leaf:
        # leaf: majority class (smallest class on tie)
        cls = int(np.bincount(y).argmax())
        nodes.append((-1, 0.0, -1, -1, cls))
        return node_idx

    feat, thresh, left_mask = best_split(X, y, min_samples_leaf)
    if feat is None:
        cls = int(np.bincount(y).argmax())
        nodes.append((-1, 0.0, -1, -1, cls))
        return node_idx

    # internal node: placeholder, will fill children after recursion
    nodes.append((feat, thresh, -1, -1, -1))
    left_idx = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_leaf, nodes)
    right_idx = build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth, min_samples_leaf, nodes)
    # update children indices
    nodes[node_idx] = (feat, thresh, left_idx, right_idx, -1)
    return node_idx


def train_tree(X, y, max_depth, min_samples_leaf):
    nodes = []
    build_tree(X, y, 0, max_depth, min_samples_leaf, nodes)
    return nodes


def evaluate(nodes, X, y):
    """Return accuracy of tree on (X, y)."""
    n = X.shape[0]
    correct = 0
    for i in range(n):
        node = 0
        while True:
            feat, thresh, left, right, cls = nodes[node]
            if feat == -1:
                if cls == y[i]:
                    correct += 1
                break
            if X[i, feat] <= thresh:
                node = left
            else:
                node = right
    return correct / n


def write_weights(nodes, path):
    """Write weights.bin."""
    num_nodes = len(nodes)
    # magic, num_nodes
    out = struct.pack("<Ii", 0x54524545, num_nodes)
    for feat, thresh, left, right, cls in nodes:
        out += struct.pack("<iffii", feat, thresh, left, right, cls)
    with open(path, "wb") as f:
        f.write(out)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20% in file order
    split = int(n * (1 - TUNING_FRACTION))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc = -1.0
    best_params = None
    best_nodes = None

    for max_depth in MAX_DEPTH_CANDIDATES:
        for min_samples_leaf in MIN_SAMPLES_LEAF_CANDIDATES:
            nodes = train_tree(X_train, y_train, max_depth, min_samples_leaf)
            acc = evaluate(nodes, X_val, y_val)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)
                best_nodes = nodes

    # Retrain on full data with best params
    max_depth, min_samples_leaf = best_params
    final_nodes = train_tree(X, y, max_depth, min_samples_leaf)
    write_weights(final_nodes, sys.argv[4])


if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixel counts). Output: one byte class.
 * Weights format: magic 0x54524545, num_nodes (int32), then per node:
 *   int32 feature, float32 threshold, int32 left, int32 right, int32 class.
 * Leaf: feature=-1, threshold=0, left=-1, right=-1, class=0..9. */
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
    int32_t feature;
    float threshold;
    int32_t left;
    int32_t right;
    int32_t cls;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t *)wb;
    int32_t num_nodes = *(int32_t *)(wb + 4);
    if (magic != 0x54524545 || num_nodes <= 0) { fprintf(stderr, "bad weights magic or count\n"); return 3; }
    long expected = 8 + (long)num_nodes * (4 + 4 + 4 + 4 + 4);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 8);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    const int d = 64;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (1) {
            Node nd = nodes[node];
            if (nd.feature == -1) {
                unsigned char out = (unsigned char)nd.cls;
                put_rec(stdout, &out, 1);
                break;
            }
            float val = (float)xi[nd.feature];
            if (val <= nd.threshold) node = nd.left;
            else node = nd.right;
        }
    }
    return 0;
}
```