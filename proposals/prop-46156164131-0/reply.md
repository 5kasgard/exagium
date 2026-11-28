```hypothesis
Family: label-tree (single decision tree with ordinal/categorical-aware splits on raw byte features).
Why: Adult has 14 byte features (mix of ordinal quantile bins and categorical indices). A single tree can capture non-linear interactions and threshold effects that linear models miss, while being far faster and smaller than GBDT. Tuning max_depth on the last 20% of train should yield a sweet spot.
Failure signal: If dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-linear, the hypothesis is wrong.
```

```python
"""label-tree: single decision tree for tabular byte features.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-threaded, stdlib+numpy only."""
import json, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
MAX_DEPTH_CANDIDATES = range(1, 21)  # will tune on last 20%


def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)


def build_tree(X, y, feature_types, depth, max_depth):
    n = len(y)
    # stopping criteria
    if depth >= max_depth or n < MIN_SAMPLES_SPLIT or gini_impurity(y) == 0.0:
        # leaf: majority class (ties -> 0)
        vals, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "class": int(vals[np.argmax(counts)])}

    parent_gini = gini_impurity(y)
    best_gain = -1.0
    best_feat = -1
    best_thresh = 0
    best_type = 0  # 0: ordinal <=, 1: categorical ==
    best_left_mask = None

    # For each feature, evaluate splits
    for feat in range(X.shape[1]):
        col = X[:, feat]
        ftype = feature_types[feat]
        unique_vals = np.unique(col)
        if ftype == "ordinal":
            # try thresholds at each unique value (split <= v)
            for v in unique_vals:
                if v == 255:
                    continue  # splitting at max gives empty right
                left_mask = col <= v
                n_left = np.sum(left_mask)
                n_right = n - n_left
                if n_left < MIN_SAMPLES_LEAF or n_right < MIN_SAMPLES_LEAF:
                    continue
                gini_left = gini_impurity(y[left_mask])
                gini_right = gini_impurity(y[~left_mask])
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = int(v)
                    best_type = 0
                    best_left_mask = left_mask
        else:  # categorical
            # try equality splits for each value
            for v in unique_vals:
                left_mask = col == v
                n_left = np.sum(left_mask)
                n_right = n - n_left
                if n_left < MIN_SAMPLES_LEAF or n_right < MIN_SAMPLES_LEAF:
                    continue
                gini_left = gini_impurity(y[left_mask])
                gini_right = gini_impurity(y[~left_mask])
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = int(v)
                    best_type = 1
                    best_left_mask = left_mask

    if best_gain <= 0 or best_left_mask is None:
        vals, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "class": int(vals[np.argmax(counts)])}

    left_tree = build_tree(X[best_left_mask], y[best_left_mask], feature_types, depth + 1, max_depth)
    right_tree = build_tree(X[~best_left_mask], y[~best_left_mask], feature_types, depth + 1, max_depth)
    return {
        "leaf": False,
        "feat": best_feat,
        "type": best_type,
        "thresh": best_thresh,
        "left": left_tree,
        "right": right_tree
    }


def tree_to_arrays(tree):
    """Flatten tree to arrays for binary storage. Returns (nodes_list) where each node is a tuple:
    (is_leaf, class, feat, type, thresh, left_idx, right_idx). Indices are positions in the list."""
    nodes = []

    def traverse(node):
        idx = len(nodes)
        if node["leaf"]:
            nodes.append((1, node["class"], 0, 0, 0, 0, 0))
        else:
            # placeholder, will fill children after recursion
            nodes.append((0, 0, node["feat"], node["type"], node["thresh"], 0, 0))
            left_idx = traverse(node["left"])
            right_idx = traverse(node["right"])
            # update this node's child indices
            nodes[idx] = (0, 0, node["feat"], node["type"], node["thresh"], left_idx, right_idx)
        return idx

    traverse(tree)
    return nodes


def evaluate_tree(nodes, X):
    """Predict using flattened nodes array (list of tuples)."""
    preds = np.zeros(len(X), dtype=np.uint8)
    for i, x in enumerate(X):
        idx = 0
        while True:
            is_leaf, cls, feat, typ, thresh, left, right = nodes[idx]
            if is_leaf:
                preds[i] = cls
                break
            val = x[feat]
            if typ == 0:  # ordinal <=
                if val <= thresh:
                    idx = left
                else:
                    idx = right
            else:  # categorical ==
                if val == thresh:
                    idx = left
                else:
                    idx = right
    return preds


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.uint8)
    feature_types = card["input"]["types"]
    n = len(y)

    # Tuning split: last 20%
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_depth = 1
    best_acc = -1.0
    rng = np.random.default_rng(SEED)

    for depth in MAX_DEPTH_CANDIDATES:
        tree = build_tree(X_train, y_train, feature_types, 0, depth)
        nodes = tree_to_arrays(tree)
        preds = evaluate_tree(nodes, X_tune)
        acc = np.mean(preds == y_tune)
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # Final training on all data
    final_tree = build_tree(X, y, feature_types, 0, best_depth)
    final_nodes = tree_to_arrays(final_tree)

    # Write weights.bin
    # Header: magic (0x54524545), d (14), num_nodes, num_classes (2), max_depth (best_depth)
    magic = 0x54524545  # "TREE" in little-endian?
    d = X.shape[1]
    num_nodes = len(final_nodes)
    num_classes = 2
    header = struct.pack("<5i", magic, d, num_nodes, num_classes, best_depth)
    # Each node: 7 int32s (is_leaf, class, feat, type, thresh, left, right)
    node_data = b""
    for node in final_nodes:
        node_data += struct.pack("<7i", *node)
    open(sys.argv[4], "wb").write(header + node_data)


if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (raw features). Output: 1 byte class (0/1). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], num_nodes = hd[2], num_classes = hd[3], max_depth = hd[4];
    if (d != 14 || num_classes != 2 || num_nodes <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 20 + (long)num_nodes * 7 * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    /* nodes: each 7 int32s: is_leaf, class, feat, type, thresh, left, right */
    int32_t *nodes = (int32_t *)(hd + 5);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int idx = 0;
        while (1) {
            int32_t *node = nodes + (long)idx * 7;
            int is_leaf = node[0];
            if (is_leaf) {
                unsigned char cls = (unsigned char)node[1];
                put_rec(stdout, &cls, 1);
                break;
            }
            int feat = node[2];
            int type = node[3];
            int thresh = node[4];
            int left = node[5];
            int right = node[6];
            unsigned char val = xi[feat];
            if (type == 0) { /* ordinal <= */
                idx = (val <= thresh) ? left : right;
            } else { /* categorical == */
                idx = (val == thresh) ? left : right;
            }
        }
    }
    return 0;
}
```