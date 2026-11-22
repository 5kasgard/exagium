```hypothesis
Family: label-tree (single decision tree with binary splits on raw byte features).
Why: The 14 features are already quantized (ordinal as 255 quantile bins, categorical as indices). A decision tree can model non-linear interactions and threshold effects that the linear model misses, while remaining extremely fast at inference (a few comparisons per item). Training is deterministic and fast via histogram-based splitting.
Failure condition: If dev accuracy does not exceed label-linear (0.8524), it suggests the task requires either more complex ensembles or that the ordinal encoding of categorical features degrades tree performance.
```

```python
"""label-tree: single decision tree for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports a compact binary tree: magic, d, num_nodes, then per-node arrays."""
import json, struct, sys

import numpy as np

SEED = 1
# Grid search over (max_depth, min_samples_leaf) on the tuning split (last 20%)
PARAM_GRID = [
    (4, 5), (4, 10), (4, 20),
    (5, 5), (5, 10), (5, 20),
    (6, 5), (6, 10), (6, 20),
    (7, 5), (7, 10), (7, 20),
    (8, 5), (8, 10), (8, 20),
]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_tree(X, y, max_depth, min_samples_leaf):
    n, d = X.shape
    nodes_feature = []
    nodes_threshold = []
    nodes_left = []
    nodes_right = []
    nodes_leaf_class = []

    def build(indices, depth):
        n_idx = len(indices)
        if n_idx == 0:
            leaf_class = 0
            nodes_feature.append(-1)
            nodes_threshold.append(0)
            nodes_left.append(-1)
            nodes_right.append(-1)
            nodes_leaf_class.append(leaf_class)
            return len(nodes_feature) - 1

        classes = y[indices]
        count0 = np.sum(classes == 0)
        count1 = n_idx - count0
        if count0 == 0 or count1 == 0:
            leaf_class = 0 if count0 > 0 else 1
            nodes_feature.append(-1)
            nodes_threshold.append(0)
            nodes_left.append(-1)
            nodes_right.append(-1)
            nodes_leaf_class.append(leaf_class)
            return len(nodes_feature) - 1

        if depth >= max_depth or n_idx < min_samples_leaf:
            leaf_class = 0 if count0 >= count1 else 1
            nodes_feature.append(-1)
            nodes_threshold.append(0)
            nodes_left.append(-1)
            nodes_right.append(-1)
            nodes_leaf_class.append(leaf_class)
            return len(nodes_feature) - 1

        best_gini = 1.0
        best_feature = -1
        best_threshold = -1

        for f in range(d):
            hist = np.zeros((256, 2), dtype=np.int32)
            vals = X[indices, f]
            cls = y[indices]
            np.add.at(hist[:, 0], vals[cls == 0], 1)
            np.add.at(hist[:, 1], vals[cls == 1], 1)
            cum0 = np.cumsum(hist[:, 0])
            cum1 = np.cumsum(hist[:, 1])
            total0 = cum0[-1]
            total1 = cum1[-1]
            for t in range(255):
                left0 = cum0[t]
                left1 = cum1[t]
                right0 = total0 - left0
                right1 = total1 - left1
                left_n = left0 + left1
                right_n = right0 + right1
                if left_n == 0 or right_n == 0:
                    continue
                gini_left = 1.0 - (left0 / left_n) ** 2 - (left1 / left_n) ** 2
                gini_right = 1.0 - (right0 / right_n) ** 2 - (right1 / right_n) ** 2
                weighted_gini = (left_n * gini_left + right_n * gini_right) / n_idx
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = f
                    best_threshold = t

        if best_feature == -1:
            leaf_class = 0 if count0 >= count1 else 1
            nodes_feature.append(-1)
            nodes_threshold.append(0)
            nodes_left.append(-1)
            nodes_right.append(-1)
            nodes_leaf_class.append(leaf_class)
            return len(nodes_feature) - 1

        node_idx = len(nodes_feature)
        nodes_feature.append(best_feature)
        nodes_threshold.append(best_threshold)
        nodes_left.append(-1)
        nodes_right.append(-1)
        nodes_leaf_class.append(0)

        left_mask = X[indices, best_feature] <= best_threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        left_child = build(left_indices, depth + 1)
        right_child = build(right_indices, depth + 1)
        nodes_left[node_idx] = left_child
        nodes_right[node_idx] = right_child
        return node_idx

    root = build(np.arange(n), 0)
    return {
        'feature': np.array(nodes_feature, dtype=np.int8),
        'threshold': np.array(nodes_threshold, dtype=np.uint8),
        'left': np.array(nodes_left, dtype=np.int32),
        'right': np.array(nodes_right, dtype=np.int32),
        'leaf_class': np.array(nodes_leaf_class, dtype=np.uint8),
        'root': root,
    }


def evaluate(tree, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        node = tree['root']
        while tree['feature'][node] != -1:
            f = tree['feature'][node]
            t = tree['threshold'][node]
            if X[i, f] <= t:
                node = tree['left'][node]
            else:
                node = tree['right'][node]
        if tree['leaf_class'][node] == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)

    n = X.shape[0]
    split_idx = int(0.8 * n)  # last 20% for tuning
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    best_acc = -1.0
    best_params = None
    best_tree = None

    for max_depth, min_samples_leaf in PARAM_GRID:
        tree = build_tree(X_train, y_train, max_depth, min_samples_leaf)
        acc = evaluate(tree, X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_params = (max_depth, min_samples_leaf)
            best_tree = tree

    # Retrain on full data with best params
    final_tree = build_tree(X, y, *best_params)

    # Write weights.bin
    magic = 0x54524545  # "TREE"
    d = X.shape[1]
    num_nodes = len(final_tree['feature'])
    out = struct.pack("<III", magic, d, num_nodes)
    for i in range(num_nodes):
        out += struct.pack(
            "<bBiib",
            final_tree['feature'][i],
            final_tree['threshold'][i],
            final_tree['left'][i],
            final_tree['right'][i],
            final_tree['leaf_class'][i],
        )
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d from weights); output is one byte class.
 * Format: 4-byte LE length, then bytes. */
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
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t*)wbuf;
    uint32_t d = *(uint32_t*)(wbuf + 4);
    uint32_t num_nodes = *(uint32_t*)(wbuf + 8);
    if (magic != 0x54524545 || d == 0 || d > 256 || num_nodes == 0) {
        fprintf(stderr, "bad weights header\n"); return 3; }
    size_t expected = 12 + (size_t)num_nodes * 11;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    int8_t *feature = (int8_t*)(wbuf + 12);
    uint8_t *threshold = (uint8_t*)(wbuf + 12 + num_nodes);
    int32_t *left = (int32_t*)(wbuf + 12 + 2 * num_nodes);
    int32_t *right = (int32_t*)(wbuf + 12 + 2 * num_nodes + 4 * num_nodes);
    uint8_t *leaf_class = (uint8_t*)(wbuf + 12 + 2 * num_nodes + 8 * num_nodes);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        int node = 0;
        while (feature[node] != -1) {
            int f = feature[node];
            uint8_t t = threshold[node];
            if (xi[f] <= t) node = left[node];
            else node = right[node];
        }
        out_byte = leaf_class[node];
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```