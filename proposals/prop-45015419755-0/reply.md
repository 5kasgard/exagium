```hypothesis
Family: label-tree. A single decision tree trained with CART (Gini impurity) on raw byte features, treating all as ordinal. Hyperparameters (max_depth, min_samples_leaf) chosen on the tuning split (last 20% of train). Inference is a few integer comparisons per record, so CPU cost should be near linear. It should beat label-linear because trees capture non-linear interactions. If dev accuracy <= 0.8524 or Ir/item > 1000, the hypothesis fails.
```

```python
import json
import struct
import sys

import numpy as np

SEED = 1
MAX_DEPTH_CANDIDATES = [3, 4, 5, 6, 7, 8, 9, 10]
MIN_SAMPLES_LEAF_CANDIDATES = [1, 5, 10, 20, 50]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4 : i + 4 + k])
        i += 4 + k
    return out


def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)


def best_split(X, y, indices, min_samples_leaf):
    n = len(indices)
    if n < 2 * min_samples_leaf:
        return None, None, None
    parent_gini = gini(y[indices])
    best_gain = 0.0
    best_feat = -1
    best_thresh = -1
    for feat in range(X.shape[1]):
        vals = X[indices, feat]
        # values are 0..255, we can compute cumulative counts
        # sort indices by feature value
        order = np.argsort(vals, kind="stable")
        sorted_vals = vals[order]
        sorted_y = y[indices[order]]
        # count classes
        n0 = np.sum(sorted_y == 0)
        n1 = np.sum(sorted_y == 1)
        left0 = 0
        left1 = 0
        # iterate possible thresholds (unique values)
        prev_val = -1
        for i in range(n - 1):
            v = sorted_vals[i]
            if sorted_y[i] == 0:
                left0 += 1
            else:
                left1 += 1
            right0 = n0 - left0
            right1 = n1 - left1
            left_n = left0 + left1
            right_n = right0 + right1
            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue
            if v == prev_val:
                continue
            prev_val = v
            # threshold is v (split <= v vs > v)
            gini_left = 1.0 - (left0 / left_n) ** 2 - (left1 / left_n) ** 2
            gini_right = 1.0 - (right0 / right_n) ** 2 - (right1 / right_n) ** 2
            gain = parent_gini - (left_n / n) * gini_left - (right_n / n) * gini_right
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = int(v)
    if best_gain <= 0:
        return None, None, None
    return best_feat, best_thresh, best_gain


def build_tree(X, y, max_depth, min_samples_leaf):
    n = X.shape[0]
    nodes_feat = []
    nodes_thresh = []
    nodes_left = []
    nodes_right = []
    nodes_leaf = []

    # stack: (indices, depth, parent_node_idx, is_left)
    stack = [(np.arange(n), 0, -1, True)]
    node_idx = 0
    while stack:
        indices, depth, parent, is_left = stack.pop()
        # create node
        if parent != -1:
            if is_left:
                nodes_left[parent] = node_idx
            else:
                nodes_right[parent] = node_idx
        # check leaf conditions
        if depth == max_depth or len(indices) < 2 * min_samples_leaf or len(np.unique(y[indices])) == 1:
            leaf_class = int(np.bincount(y[indices]).argmax())
            nodes_feat.append(-1)
            nodes_thresh.append(-1)
            nodes_left.append(-1)
            nodes_right.append(-1)
            nodes_leaf.append(leaf_class)
            node_idx += 1
            continue
        feat, thresh, _ = best_split(X, y, indices, min_samples_leaf)
        if feat is None:
            leaf_class = int(np.bincount(y[indices]).argmax())
            nodes_feat.append(-1)
            nodes_thresh.append(-1)
            nodes_left.append(-1)
            nodes_right.append(-1)
            nodes_leaf.append(leaf_class)
            node_idx += 1
            continue
        # internal node
        nodes_feat.append(feat)
        nodes_thresh.append(thresh)
        nodes_left.append(-1)  # placeholder
        nodes_right.append(-1)
        nodes_leaf.append(-1)
        cur_idx = node_idx
        node_idx += 1
        # split indices
        vals = X[indices, feat]
        left_mask = vals <= thresh
        left_idx = indices[left_mask]
        right_idx = indices[~left_mask]
        # push right then left (so left processed first)
        stack.append((right_idx, depth + 1, cur_idx, False))
        stack.append((left_idx, depth + 1, cur_idx, True))
    return (
        np.array(nodes_feat, dtype=np.int32),
        np.array(nodes_thresh, dtype=np.int32),
        np.array(nodes_left, dtype=np.int32),
        np.array(nodes_right, dtype=np.int32),
        np.array(nodes_leaf, dtype=np.int32),
    )


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    best_acc = -1.0
    best_params = None
    best_tree = None
    rng = np.random.default_rng(SEED)

    for max_depth in MAX_DEPTH_CANDIDATES:
        for min_samples_leaf in MIN_SAMPLES_LEAF_CANDIDATES:
            feat, thresh, left, right, leaf = build_tree(
                X[train_idx], y[train_idx], max_depth, min_samples_leaf
            )
            # predict on tune
            preds = np.zeros(len(tune_idx), dtype=np.int64)
            for i, idx in enumerate(tune_idx):
                node = 0
                while feat[node] != -1:
                    if X[idx, feat[node]] <= thresh[node]:
                        node = left[node]
                    else:
                        node = right[node]
                preds[i] = leaf[node]
            acc = np.mean(preds == y[tune_idx])
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)
                best_tree = (feat, thresh, left, right, leaf)

    # retrain on full data with best params
    max_depth, min_samples_leaf = best_params
    feat, thresh, left, right, leaf = build_tree(X, y, max_depth, min_samples_leaf)

    # serialize
    magic = 0x54524545  # 'TREE' little-endian
    num_nodes = len(feat)
    num_features = X.shape[1]
    out = struct.pack("<3i", magic, num_nodes, num_features)
    for i in range(num_nodes):
        out += struct.pack("<5i", feat[i], thresh[i], left[i], right[i], leaf[i])
    open(sys.argv[4], "wb").write(out)


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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x54524545) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t num_nodes = hd[1];
    int32_t d = hd[2];
    if (d < 1 || d > 256 || nw != 12 + (long)num_nodes * 20) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    int32_t *feat = hd + 3;
    int32_t *thresh = feat + num_nodes;
    int32_t *left = thresh + num_nodes;
    int32_t *right = left + num_nodes;
    int32_t *leaf = right + num_nodes;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int32_t node = 0;
        while (feat[node] != -1) {
            if (xi[feat[node]] <= (unsigned char)thresh[node])
                node = left[node];
            else
                node = right[node];
        }
        unsigned char out = (unsigned char)leaf[node];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```