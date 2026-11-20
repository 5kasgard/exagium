```hypothesis
Family: label-tree (single deterministic CART decision tree with max-depth tuned on the last 20% of training data).
Why: The adult dataset has non-linear feature interactions (e.g., education-num vs occupation) that a linear model misses. A shallow tree captures these with few splits, beating linear accuracy while keeping inference extremely fast (only a handful of byte comparisons per item).
Failure indication: If dev accuracy does not exceed 0.8524 (label-linear) or inference cost exceeds label-linear, the family does not meet its goal.
```

```python
import json
import struct
import sys

import numpy as np

SEED = 1
# Hyperparameters to tune on the last 20% of train.in
MAX_DEPTH_CANDIDATES = [3, 4, 5, 6, 7]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4 : i + 4 + k])
        i += 4 + k
    return out


def gini(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)


def build_tree(X, y, max_depth, depth=0):
    n = len(y)
    if n == 0:
        return {"leaf": True, "class": 0}
    if np.all(y == y[0]):
        return {"leaf": True, "class": int(y[0])}
    if depth >= max_depth:
        cls = int(np.bincount(y).argmax())
        return {"leaf": True, "class": cls}

    total_counts = np.bincount(y, minlength=2)
    total_gini = gini(total_counts)
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1

    for feat in range(X.shape[1]):
        hist = np.zeros((256, 2), dtype=np.int64)
        np.add.at(hist, (X[:, feat], y), 1)
        cum = hist.cumsum(axis=0)
        left_counts = cum[:-1]  # thresholds 0..254
        right_counts = total_counts - left_counts
        left_totals = left_counts.sum(axis=1)
        right_totals = right_counts.sum(axis=1)
        valid = (left_totals > 0) & (right_totals > 0)
        if not np.any(valid):
            continue
        left_gini = 1.0 - np.sum((left_counts / left_totals[:, None]) ** 2, axis=1)
        right_gini = 1.0 - np.sum((right_counts / right_totals[:, None]) ** 2, axis=1)
        weighted_gini = (left_totals * left_gini + right_totals * right_gini) / n
        gain = total_gini - weighted_gini
        idx = np.where(valid)[0]
        max_gain_feat = gain[idx].max()
        if max_gain_feat > best_gain + 1e-12:
            best_gain = max_gain_feat
            best_thresh_feat = idx[gain[idx] == max_gain_feat][0]
            best_feat = feat
            best_thresh = int(best_thresh_feat)
        elif abs(max_gain_feat - best_gain) <= 1e-12 and best_feat == -1:
            # tie on gain, first feature in order wins (deterministic)
            best_thresh_feat = idx[gain[idx] == max_gain_feat][0]
            best_feat = feat
            best_thresh = int(best_thresh_feat)

    if best_gain <= 1e-12:
        cls = int(np.bincount(y).argmax())
        return {"leaf": True, "class": cls}

    left_mask = X[:, best_feat] <= best_thresh
    right_mask = ~left_mask
    left_node = build_tree(X[left_mask], y[left_mask], max_depth, depth + 1)
    right_node = build_tree(X[right_mask], y[right_mask], max_depth, depth + 1)
    return {
        "leaf": False,
        "feat": best_feat,
        "thresh": best_thresh,
        "left": left_node,
        "right": right_node,
    }


def flatten_tree(node):
    """Return list of nodes in pre-order with child indices."""
    if node["leaf"]:
        return [(1, node["class"], 0, -1, -1)]
    left_flat = flatten_tree(node["left"])
    right_flat = flatten_tree(node["right"])
    left_size = len(left_flat)
    this_node = (0, node["feat"], node["thresh"], 1, 1 + left_size)
    return [this_node] + left_flat + right_flat


def evaluate_tree(flat_nodes, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        node_idx = 0
        while True:
            is_leaf, feat_or_class, thresh, left, right = flat_nodes[node_idx]
            if is_leaf:
                pred = feat_or_class
                break
            if xi[feat_or_class] <= thresh:
                node_idx = left
            else:
                node_idx = right
        if pred == yi:
            correct += 1
    return correct / len(y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)

    n = len(X)
    tune_size = n // 5
    X_train, y_train = X[:-tune_size], y[:-tune_size]
    X_tune, y_tune = X[-tune_size:], y[-tune_size:]

    best_depth = MAX_DEPTH_CANDIDATES[0]
    best_acc = -1.0
    for depth in MAX_DEPTH_CANDIDATES:
        tree = build_tree(X_train, y_train, depth)
        flat = flatten_tree(tree)
        acc = evaluate_tree(flat, X_tune, y_tune)
        if acc > best_acc + 1e-12:
            best_acc = acc
            best_depth = depth

    # Retrain on full data with best depth
    final_tree = build_tree(X, y, best_depth)
    flat_nodes = flatten_tree(final_tree)

    # Write weights.bin
    magic = 0x54524545  # "TREE"
    num_nodes = len(flat_nodes)
    num_features = card["input"]["bytes"]
    header = struct.pack("<Iiii", magic, num_nodes, num_features, best_depth)
    node_bytes = b""
    for node in flat_nodes:
        is_leaf, a, b, left, right = node
        node_bytes += struct.pack("<BBBBii", is_leaf, a, b, 0, left, right)
    with open(sys.argv[4], "wb") as f:
        f.write(header + node_bytes)


if __name__ == "__main__":
    main()
```

```c
/* Decision tree inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class (0 or 1). */
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
    uint8_t feat_or_class;
    uint8_t thresh;
    uint8_t pad;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t num_nodes = *(int32_t*)(wb + 4);
    int32_t num_features = *(int32_t*)(wb + 8);
    int32_t max_depth = *(int32_t*)(wb + 12); (void)max_depth;
    if (num_nodes <= 0 || num_features != 14) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + (long)num_nodes * 12;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    Node *nodes = (Node*)(wb + 16);
    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].is_leaf) {
            if (nodes[i].feat_or_class > 1) { fprintf(stderr, "bad leaf class\n"); return 3; }
            if (nodes[i].left != -1 || nodes[i].right != -1) { fprintf(stderr, "bad leaf children\n"); return 3; }
        } else {
            if (nodes[i].feat_or_class >= num_features) { fprintf(stderr, "bad feature index\n"); return 3; }
            if (nodes[i].left < 0 || nodes[i].left >= num_nodes) { fprintf(stderr, "bad left child\n"); return 3; }
            if (nodes[i].right < 0 || nodes[i].right >= num_nodes) { fprintf(stderr, "bad right child\n"); return 3; }
        }
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != num_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, num_features); return 3; }
        int node_idx = 0;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.is_leaf) {
                out_byte = nd.feat_or_class;
                break;
            }
            if (xi[nd.feat_or_class] <= nd.thresh)
                node_idx = nd.left;
            else
                node_idx = nd.right;
        }
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```