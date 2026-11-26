```hypothesis
Family: label-dt (single decision tree). The optdigits pixels are ordinal 0..16 with only 17 distinct values per feature, so axis-aligned splits are natural and a shallow tree can capture non-linear interactions that a linear model misses. Inference is just a handful of integer comparisons, so CPU cost should be far below label-linear. If dev accuracy does not exceed 0.9613 or instructions/item exceed label-linear's 5059, the hypothesis fails.
```

```python
"""label-dt: single decision tree for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then retrained on all data.
Weights format: magic 0x44543131, d, C, n_nodes; then per node: feature(u8), threshold(f32), left(i32), right(i32), class(u8)."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
MAX_DEPTH_OPTS = [5, 6, 7, 8, 9, 10]
MIN_SAMPLES_SPLIT_OPTS = [2, 5, 10]
MIN_SAMPLES_LEAF_OPTS = [1, 2, 5, 10]

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

def best_split(X, y, min_samples_leaf):
    n, d = X.shape
    best_gain = -1.0
    best_feature = -1
    best_thresh = -1.0
    parent_gini = gini(y)
    for f in range(d):
        vals = np.unique(X[:, f])
        if len(vals) <= 1:
            continue
        thresholds = (vals[:-1] + vals[1:]) / 2.0
        for t in thresholds:
            left_mask = X[:, f] <= t
            right_mask = ~left_mask
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            gini_left = gini(y[left_mask])
            gini_right = gini(y[right_mask])
            weighted_gini = (n_left * gini_left + n_right * gini_right) / n
            gain = parent_gini - weighted_gini
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_thresh = t
    return best_gain, best_feature, best_thresh

def build_tree(X, y, depth, max_depth, min_samples_split, min_samples_leaf, nodes):
    # returns node index
    node_idx = len(nodes)
    # Check stopping criteria
    if depth >= max_depth or len(y) < min_samples_split or len(np.unique(y)) == 1:
        # leaf
        majority = np.bincount(y).argmax()
        nodes.append({'feature': 255, 'threshold': 0.0, 'left': -1, 'right': -1, 'class': int(majority)})
        return node_idx
    gain, feature, thresh = best_split(X, y, min_samples_leaf)
    if gain <= 0 or feature == -1:
        majority = np.bincount(y).argmax()
        nodes.append({'feature': 255, 'threshold': 0.0, 'left': -1, 'right': -1, 'class': int(majority)})
        return node_idx
    left_mask = X[:, feature] <= thresh
    right_mask = ~left_mask
    left_idx = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_split, min_samples_leaf, nodes)
    right_idx = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples_split, min_samples_leaf, nodes)
    nodes.append({'feature': feature, 'threshold': thresh, 'left': left_idx, 'right': right_idx, 'class': 0})
    # Note: we append after children so that indices are correct (post-order). But we need root at index 0.
    # We'll instead build in pre-order: create node first, then children. Let's change.
    # Actually easier: build recursively and return node, then flatten later. Let's do a class-based approach.
    # We'll reimplement with a Node class and then flatten.
    raise NotImplementedError("Switching to class-based builder")

class Node:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'cls')
    def __init__(self, feature=255, threshold=0.0, left=None, right=None, cls=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.cls = cls

def build_tree_rec(X, y, depth, max_depth, min_samples_split, min_samples_leaf):
    if depth >= max_depth or len(y) < min_samples_split or len(np.unique(y)) == 1:
        majority = int(np.bincount(y).argmax())
        return Node(cls=majority)
    gain, feature, thresh = best_split(X, y, min_samples_leaf)
    if gain <= 0 or feature == -1:
        majority = int(np.bincount(y).argmax())
        return Node(cls=majority)
    left_mask = X[:, feature] <= thresh
    right_mask = ~left_mask
    left_child = build_tree_rec(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_split, min_samples_leaf)
    right_child = build_tree_rec(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples_split, min_samples_leaf)
    return Node(feature=feature, threshold=thresh, left=left_child, right=right_child)

def flatten_tree(root):
    nodes = []
    def dfs(node):
        idx = len(nodes)
        nodes.append(node)
        if node.left is not None:
            dfs(node.left)
        if node.right is not None:
            dfs(node.right)
    dfs(root)
    # Now we have pre-order. But we need to set left/right as indices.
    # We'll do a second pass to assign indices.
    node_to_idx = {id(node): i for i, node in enumerate(nodes)}
    flat = []
    for node in nodes:
        if node.left is not None:
            left_idx = node_to_idx[id(node.left)]
            right_idx = node_to_idx[id(node.right)]
            flat.append((node.feature, node.threshold, left_idx, right_idx, 0))
        else:
            flat.append((255, 0.0, -1, -1, node.cls))
    return flat

def predict_tree(flat_nodes, x):
    idx = 0
    while True:
        feature, thresh, left, right, cls = flat_nodes[idx]
        if left == -1:
            return cls
        if x[feature] <= thresh:
            idx = left
        else:
            idx = right

def accuracy(flat_nodes, X, y):
    correct = 0
    for i in range(len(X)):
        pred = predict_tree(flat_nodes, X[i])
        if pred == y[i]:
            correct += 1
    return correct / len(X)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X)
    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc = -1.0
    best_params = None
    best_flat = None

    for max_depth in MAX_DEPTH_OPTS:
        for min_samples_split in MIN_SAMPLES_SPLIT_OPTS:
            for min_samples_leaf in MIN_SAMPLES_LEAF_OPTS:
                root = build_tree_rec(X_train, y_train, 0, max_depth, min_samples_split, min_samples_leaf)
                flat = flatten_tree(root)
                acc = accuracy(flat, X_val, y_val)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (max_depth, min_samples_split, min_samples_leaf)
                    best_flat = flat

    # Retrain on full data with best params
    max_depth, min_samples_split, min_samples_leaf = best_params
    root = build_tree_rec(X, y, 0, max_depth, min_samples_split, min_samples_leaf)
    flat_nodes = flatten_tree(root)

    # Write weights.bin
    magic = 0x44543131  # "DT11"
    n_nodes = len(flat_nodes)
    # Header: magic, d, C, n_nodes (all uint32 little-endian)
    header = struct.pack("<IIII", magic, d, C, n_nodes)
    node_bytes = bytearray()
    for feature, thresh, left, right, cls in flat_nodes:
        node_bytes += struct.pack("<BfiiB", feature, thresh, left, right, cls)
    # Validate size
    expected_size = 16 + n_nodes * 14
    actual_size = len(header) + len(node_bytes)
    assert actual_size == expected_size, f"Size mismatch: {actual_size} vs {expected_size}"
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(node_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-dt inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class). */
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x44543131) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t d = wb[4] | wb[5] << 8 | wb[6] << 16 | (uint32_t)wb[7] << 24;
    uint32_t C = wb[8] | wb[9] << 8 | wb[10] << 16 | (uint32_t)wb[11] << 24;
    uint32_t n_nodes = wb[12] | wb[13] << 8 | wb[14] << 16 | (uint32_t)wb[15] << 24;
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16 + (long)n_nodes * 14;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    // Parse nodes
    uint8_t *feature = malloc(n_nodes);
    float *threshold = malloc(n_nodes * sizeof(float));
    int32_t *left = malloc(n_nodes * sizeof(int32_t));
    int32_t *right = malloc(n_nodes * sizeof(int32_t));
    uint8_t *cls = malloc(n_nodes);
    unsigned char *ptr = wb + 16;
    for (uint32_t i = 0; i < n_nodes; i++) {
        feature[i] = ptr[0];
        memcpy(&threshold[i], ptr + 1, 4);
        memcpy(&left[i], ptr + 5, 4);
        memcpy(&right[i], ptr + 9, 4);
        cls[i] = ptr[13];
        ptr += 14;
    }
    free(wb);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char pred;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        uint32_t idx = 0;
        while (1) {
            uint8_t f = feature[idx];
            float th = threshold[idx];
            int32_t l = left[idx];
            int32_t r = right[idx];
            if (l == -1) { pred = cls[idx]; break; }
            if (xi[f] <= th) idx = l; else idx = r;
        }
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```