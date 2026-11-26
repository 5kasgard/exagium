```hypothesis
Family: label-tree (single CART decision tree).  
Why: The 8×8 digit images have strong local spatial interactions (strokes, loops) that a linear model misses. A single tree of depth ≤10 can capture these with axis-aligned splits on pixel intensities (0–16), yielding accuracy close to boosted trees but with far lower inference cost (only a few integer comparisons per sample).  
Failure signal: Dev accuracy ≤ 0.9613 (label-linear's score) or inference cost > 2000 instructions/item.
```

```python
"""label-tree: single CART decision tree for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Exports weights.bin: magic, n_features, n_classes, n_nodes, then n_nodes×12-byte records.
Node record: type(u8), feature(u8), threshold(u8), class(u8), left(u32), right(u32).
Leaf nodes use type=1 and class; split nodes use type=0, feature, threshold, left, right."""
import json, struct, sys
import numpy as np

SEED = 1
MAX_DEPTH_GRID = [5, 6, 7, 8, 9, 10]
MIN_SAMPLES_SPLIT_GRID = [2, 5, 10, 20]
MIN_SAMPLES_LEAF_GRID = [1, 2, 5, 10]

MAGIC = 0x54524545  # "TREE" little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini_from_counts(counts):
    # counts: (n_classes,) int64
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.dot(p, p)


def best_split(X, y, idx, n_classes, min_samples_leaf):
    n = len(idx)
    if n < 2:
        return None, None, None, -1.0
    best_gain = -1.0
    best_feat = best_thr = -1
    parent_counts = np.bincount(y[idx], minlength=n_classes)
    parent_gini = gini_from_counts(parent_counts)
    for feat in range(X.shape[1]):
        vals = X[idx, feat]
        # values are 0..16, thresholds 0..15
        # compute class histograms per value
        hist = np.zeros((17, n_classes), dtype=np.int64)
        for v, c in zip(vals, y[idx]):
            hist[v, c] += 1
        # cumulative sums from left
        left_counts = np.zeros(n_classes, dtype=np.int64)
        right_counts = parent_counts.copy()
        for thr in range(16):  # split <= thr
            left_counts += hist[thr]
            right_counts -= hist[thr]
            n_left = left_counts.sum()
            n_right = right_counts.sum()
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            gini_left = gini_from_counts(left_counts)
            gini_right = gini_from_counts(right_counts)
            gain = parent_gini - (n_left * gini_left + n_right * gini_right) / n
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thr = thr
    if best_gain <= 0:
        return None, None, None, -1.0
    return best_feat, best_thr, best_gain, parent_gini


def build_tree(X, y, max_depth, min_samples_split, min_samples_leaf, n_classes):
    n_samples = X.shape[0]
    # node arrays
    node_type = []      # 0 split, 1 leaf
    node_feature = []
    node_threshold = []
    node_class = []
    node_left = []
    node_right = []

    # stack of (node_index, sample_indices, depth)
    stack = [(0, np.arange(n_samples, dtype=np.int32), 0)]
    node_type.append(0)  # placeholder for root
    node_feature.append(0)
    node_threshold.append(0)
    node_class.append(0)
    node_left.append(0)
    node_right.append(0)

    while stack:
        node_idx, idx, depth = stack.pop()
        n = len(idx)
        # stopping criteria
        if depth >= max_depth or n < min_samples_split:
            # make leaf
            counts = np.bincount(y[idx], minlength=n_classes)
            leaf_class = int(np.argmax(counts))
            node_type[node_idx] = 1
            node_class[node_idx] = leaf_class
            continue
        # find best split
        feat, thr, gain, _ = best_split(X, y, idx, n_classes, min_samples_leaf)
        if feat is None:
            counts = np.bincount(y[idx], minlength=n_classes)
            leaf_class = int(np.argmax(counts))
            node_type[node_idx] = 1
            node_class[node_idx] = leaf_class
            continue
        # split
        node_type[node_idx] = 0
        node_feature[node_idx] = feat
        node_threshold[node_idx] = thr
        left_mask = X[idx, feat] <= thr
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        # create children
        left_node = len(node_type)
        right_node = left_node + 1
        node_left[node_idx] = left_node
        node_right[node_idx] = right_node
        # push children (right first so left processed first, but order doesn't matter)
        stack.append((right_node, right_idx, depth + 1))
        stack.append((left_node, left_idx, depth + 1))
        # extend arrays for children
        for _ in range(2):
            node_type.append(0)
            node_feature.append(0)
            node_threshold.append(0)
            node_class.append(0)
            node_left.append(0)
            node_right.append(0)

    return (node_type, node_feature, node_threshold, node_class, node_left, node_right)


def predict_tree(X, nodes):
    node_type, node_feature, node_threshold, node_class, node_left, node_right = nodes
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        node = 0
        while node_type[node] == 0:
            feat = node_feature[node]
            thr = node_threshold[node]
            if X[i, feat] <= thr:
                node = node_left[node]
            else:
                node = node_right[node]
        preds[i] = node_class[node]
    return preds


def serialize_tree(nodes, n_features, n_classes):
    node_type, node_feature, node_threshold, node_class, node_left, node_right = nodes
    n_nodes = len(node_type)
    header = struct.pack("<4I", MAGIC, n_features, n_classes, n_nodes)
    body = bytearray()
    for i in range(n_nodes):
        body += struct.pack("<BBBBI I",
                            node_type[i],
                            node_feature[i],
                            node_threshold[i],
                            node_class[i],
                            node_left[i],
                            node_right[i])
    return header + body


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n_classes = card["classes"]
    n_samples = X.shape[0]
    tune_size = int(0.2 * n_samples)
    train_idx = np.arange(n_samples - tune_size)
    tune_idx = np.arange(n_samples - tune_size, n_samples)

    best_acc = -1.0
    best_params = None
    best_nodes = None

    for max_depth in MAX_DEPTH_GRID:
        for min_samples_split in MIN_SAMPLES_SPLIT_GRID:
            for min_samples_leaf in MIN_SAMPLES_LEAF_GRID:
                nodes = build_tree(X[train_idx], y[train_idx], max_depth, min_samples_split, min_samples_leaf, n_classes)
                preds = predict_tree(X[tune_idx], nodes)
                acc = (preds == y[tune_idx]).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = (max_depth, min_samples_split, min_samples_leaf)
                    best_nodes = nodes

    # retrain on all data with best params
    max_depth, min_samples_split, min_samples_leaf = best_params
    final_nodes = build_tree(X, y, max_depth, min_samples_split, min_samples_leaf, n_classes)
    blob = serialize_tree(final_nodes, X.shape[1], n_classes)
    open(sys.argv[4], "wb").write(blob)


if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels). Output record: 1 byte (class 0-9).
 * Weights format: magic(4), n_features(4), n_classes(4), n_nodes(4),
 * then n_nodes * 12 bytes: type(u8), feature(u8), threshold(u8), class(u8), left(u32), right(u32). */
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
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = w[0] | w[1] << 8 | w[2] << 16 | (uint32_t)w[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t n_features = w[4] | w[5] << 8 | w[6] << 16 | (uint32_t)w[7] << 24;
    uint32_t n_classes = w[8] | w[9] << 8 | w[10] << 16 | (uint32_t)w[11] << 24;
    uint32_t n_nodes = w[12] | w[13] << 8 | w[14] << 16 | (uint32_t)w[15] << 24;
    if (n_features != 64 || n_classes != 10) { fprintf(stderr, "bad dimensions\n"); return 3; }
    size_t expected = 16 + (size_t)n_nodes * 12;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %zu\n", nw, expected); return 3; }

    const unsigned char *nodes = w + 16;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        uint32_t node = 0;
        while (1) {
            const unsigned char *nd = nodes + node * 12;
            uint8_t type = nd[0];
            if (type == 1) { // leaf
                unsigned char cls = nd[3];
                put_rec(stdout, &cls, 1);
                break;
            }
            uint8_t feat = nd[1];
            uint8_t thr = nd[2];
            uint32_t left = nd[4] | (uint32_t)nd[5] << 8 | (uint32_t)nd[6] << 16 | (uint32_t)nd[7] << 24;
            uint32_t right = nd[8] | (uint32_t)nd[9] << 8 | (uint32_t)nd[10] << 16 | (uint32_t)nd[11] << 24;
            if (xi[feat] <= thr) node = left;
            else node = right;
        }
    }
    return 0;
}
```