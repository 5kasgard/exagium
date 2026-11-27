```hypothesis
Family: label-dtree (single decision tree, max depth 6, categorical/ordinal aware splits).
Why: The Adult dataset has 14 mixed-type features; a shallow tree can capture non-linear interactions (e.g., education × occupation) that linear models miss, while remaining extremely fast at inference (a few integer comparisons per record). Depth 6 gives at most 64 leaves, fitting in cache and keeping weights tiny.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear, the family does not justify its existence.
```

```python
"""label-dtree: single decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Exports: magic 0x44545245, num_nodes, num_features, num_classes,
then per node: is_leaf(u8), feature(u8), threshold(u8), is_categorical(u8),
left(u32), right(u32), prob0(f32), prob1(f32)."""
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split (last 20% of train.in)
MAX_DEPTH_GRID = [3, 4, 5, 6]
MIN_SAMPLES_LEAF_GRID = [10, 20, 50]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class Node:
    __slots__ = ("is_leaf", "feature", "threshold", "is_categorical", "left", "right", "prob")
    def __init__(self):
        self.is_leaf = 0
        self.feature = 0
        self.threshold = 0
        self.is_categorical = 0
        self.left = 0
        self.right = 0
        self.prob = np.zeros(2, dtype=np.float32)


def gini(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)


def build_tree(X, y, types, max_depth, min_samples_leaf, depth=0):
    """Returns (nodes_list, root_index). nodes_list is list of Node."""
    nodes = []
    n_samples = len(y)
    classes = np.bincount(y, minlength=2)
    node = Node()
    node.prob = classes.astype(np.float32) / n_samples
    nodes.append(node)
    root_idx = 0

    # stack of (node_idx, sample_indices, depth)
    stack = [(root_idx, np.arange(n_samples, dtype=np.int32), 0)]

    while stack:
        node_idx, idx, depth = stack.pop()
        node = nodes[node_idx]
        n = len(idx)
        if depth >= max_depth or n < min_samples_leaf or np.all(y[idx] == y[idx[0]]):
            node.is_leaf = 1
            continue

        # Find best split
        best_gain = -1.0
        best_feature = -1
        best_threshold = -1
        best_is_cat = 0
        best_left_idx = None
        best_right_idx = None

        parent_gini = gini(np.bincount(y[idx], minlength=2))

        for f in range(X.shape[1]):
            vals = X[idx, f]
            if types[f] == "ordinal":
                # Try thresholds from 0 to 254 (split <= thr)
                # Use only thresholds that actually separate data
                unique = np.unique(vals)
                if len(unique) <= 1:
                    continue
                # candidate thresholds are midpoints between consecutive unique values
                # but since values are integers 0-255, we can use unique[:-1] as thresholds (<= thr)
                for thr in unique[:-1]:
                    left_mask = vals <= thr
                    right_mask = ~left_mask
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
                    left_y = y[idx[left_mask]]
                    right_y = y[idx[right_mask]]
                    left_gini = gini(np.bincount(left_y, minlength=2))
                    right_gini = gini(np.bincount(right_y, minlength=2))
                    n_left = left_mask.sum()
                    n_right = right_mask.sum()
                    gain = parent_gini - (n_left * left_gini + n_right * right_gini) / n
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_feature = f
                        best_threshold = thr
                        best_is_cat = 0
                        best_left_idx = idx[left_mask]
                        best_right_idx = idx[right_mask]
            else:  # categorical
                # Try each possible value as equality split
                unique = np.unique(vals)
                if len(unique) <= 1:
                    continue
                for v in unique:
                    left_mask = vals == v
                    right_mask = ~left_mask
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
                    left_y = y[idx[left_mask]]
                    right_y = y[idx[right_mask]]
                    left_gini = gini(np.bincount(left_y, minlength=2))
                    right_gini = gini(np.bincount(right_y, minlength=2))
                    n_left = left_mask.sum()
                    n_right = right_mask.sum()
                    gain = parent_gini - (n_left * left_gini + n_right * right_gini) / n
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_feature = f
                        best_threshold = v
                        best_is_cat = 1
                        best_left_idx = idx[left_mask]
                        best_right_idx = idx[right_mask]

        if best_gain <= 0:
            node.is_leaf = 1
            continue

        # Create children
        node.is_leaf = 0
        node.feature = best_feature
        node.threshold = best_threshold
        node.is_categorical = best_is_cat

        left_node = Node()
        left_classes = np.bincount(y[best_left_idx], minlength=2)
        left_node.prob = left_classes.astype(np.float32) / len(best_left_idx)
        left_idx = len(nodes)
        nodes.append(left_node)

        right_node = Node()
        right_classes = np.bincount(y[best_right_idx], minlength=2)
        right_node.prob = right_classes.astype(np.float32) / len(best_right_idx)
        right_idx = len(nodes)
        nodes.append(right_node)

        node.left = left_idx
        node.right = right_idx

        # Push children (right first so left is processed next? order doesn't matter)
        stack.append((right_idx, best_right_idx, depth + 1))
        stack.append((left_idx, best_left_idx, depth + 1))

    return nodes, root_idx


def evaluate(nodes, root_idx, X, y):
    n = len(y)
    correct = 0
    for i in range(n):
        xi = X[i]
        node_idx = root_idx
        while True:
            node = nodes[node_idx]
            if node.is_leaf:
                pred = 0 if node.prob[0] >= node.prob[1] else 1
                if pred == y[i]:
                    correct += 1
                break
            f = node.feature
            thr = node.threshold
            if node.is_categorical:
                if xi[f] == thr:
                    node_idx = node.left
                else:
                    node_idx = node.right
            else:
                if xi[f] <= thr:
                    node_idx = node.left
                else:
                    node_idx = node.right
    return correct / n


def serialize(nodes, root_idx, num_features, num_classes, out_path):
    magic = 0x44545245  # 'DTRE'
    num_nodes = len(nodes)
    # header: magic (u32), num_nodes (u32), num_features (u8), num_classes (u8), root_idx (u32), padding 2 bytes
    header = struct.pack("<IIBBH", magic, num_nodes, num_features, num_classes, root_idx)
    # each node: is_leaf(u8), feature(u8), threshold(u8), is_categorical(u8), left(u32), right(u32), prob0(f32), prob1(f32)
    node_fmt = "<BBBBIff"
    node_size = struct.calcsize(node_fmt)
    with open(out_path, "wb") as f:
        f.write(header)
        for node in nodes:
            f.write(struct.pack(node_fmt,
                                node.is_leaf,
                                node.feature,
                                node.threshold,
                                node.is_categorical,
                                node.left,
                                node.right,
                                node.prob[0],
                                node.prob[1]))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    types = card["input"]["types"]
    num_features = card["input"]["bytes"]
    num_classes = card["classes"]

    n = len(y)
    # Tuning split: last 20% in file order
    split_idx = int(n * 0.8)
    train_idx = np.arange(split_idx)
    tune_idx = np.arange(split_idx, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_acc = -1.0
    best_params = None
    best_nodes = None
    best_root = 0

    for max_depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_SAMPLES_LEAF_GRID:
            nodes, root = build_tree(X_train, y_train, types, max_depth, min_leaf)
            acc = evaluate(nodes, root, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_nodes = nodes
                best_root = root

    # Retrain on full data with best params
    max_depth, min_leaf = best_params
    final_nodes, final_root = build_tree(X, y, types, max_depth, min_leaf)
    serialize(final_nodes, final_root, num_features, num_classes, sys.argv[4])


if __name__ == "__main__":
    main()
```

```c
/* label-dtree inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (fixed). Output record: 1 byte class index.
 * Weights format: magic 0x44545245, num_nodes(u32), num_features(u8), num_classes(u8), root_idx(u32),
 * then per node: is_leaf(u8), feature(u8), threshold(u8), is_categorical(u8), left(u32), right(u32), prob0(f32), prob1(f32). */
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
    uint8_t is_categorical;
    uint32_t left;
    uint32_t right;
    float prob0;
    float prob1;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }

    // Parse header
    uint32_t magic = wbuf[0] | wbuf[1] << 8 | wbuf[2] << 16 | (uint32_t)wbuf[3] << 24;
    if (magic != 0x44545245) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t num_nodes = wbuf[4] | wbuf[5] << 8 | wbuf[6] << 16 | (uint32_t)wbuf[7] << 24;
    uint8_t num_features = wbuf[8];
    uint8_t num_classes = wbuf[9];
    uint32_t root_idx = wbuf[10] | wbuf[11] << 8 | wbuf[12] << 16 | (uint32_t)wbuf[13] << 24;
    // padding 2 bytes at 14,15

    if (num_features != 14 || num_classes != 2) { fprintf(stderr, "unexpected dims\n"); return 3; }

    size_t header_size = 16;
    size_t node_size = sizeof(Node);
    if (nw != (long)(header_size + num_nodes * node_size)) {
        fprintf(stderr, "bad weights size: expected %zu got %ld\n", header_size + num_nodes * node_size, nw);
        return 3;
    }

    Node *nodes = (Node *)(wbuf + header_size);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }
        uint32_t node_idx = root_idx;
        while (1) {
            Node *node = &nodes[node_idx];
            if (node->is_leaf) {
                unsigned char best = (node->prob0 >= node->prob1) ? 0 : 1;
                put_rec(stdout, &best, 1);
                break;
            }
            uint8_t val = xi[node->feature];
            if (node->is_categorical) {
                node_idx = (val == node->threshold) ? node->left : node->right;
            } else {
                node_idx = (val <= node->threshold) ? node->left : node->right;
            }
        }
    }
    return 0;
}
```