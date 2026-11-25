```hypothesis
Family: label-decision-tree (greedy CART on raw byte features). The Adult data has 14 byte columns (ordinal quantile-binned, categorical indices). A single decision tree with tuned max depth can capture non-linear interactions missed by linear models, while inference is a few branch-free table lookups per item. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds label-linear, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH_CANDIDATES = range(1, 16)  # depths to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)


def build_tree(X, y, max_depth, depth=0):
    """Returns node index, populates global lists."""
    n = len(y)
    # Check stopping conditions
    if depth == max_depth or n == 0:
        return make_leaf(y)
    # If pure
    if np.all(y == y[0]):
        return make_leaf(y)

    best_gain = -1.0
    best_feature = -1
    best_threshold = -1
    best_left_mask = None

    # For each feature, find best split
    for f in range(X.shape[1]):
        vals = X[:, f]
        # Since values are 0..255, we can compute class histograms per value
        # We'll use 2 classes (0,1)
        hist = np.zeros((256, 2), dtype=np.int32)
        for v, c in zip(vals, y):
            hist[v, c] += 1

        # Cumulative sums from left to right
        cum_left = np.cumsum(hist, axis=0)
        total_left = cum_left.sum(axis=1)
        total_right = n - total_left
        # Avoid splits where one side empty
        valid = (total_left > 0) & (total_right > 0)
        if not np.any(valid):
            continue

        gini_left = 1.0 - np.sum((cum_left / total_left[:, None]) ** 2, axis=1)
        gini_right = 1.0 - np.sum(((hist.sum(axis=0) - cum_left) / total_right[:, None]) ** 2, axis=1)
        weighted_gini = (total_left * gini_left + total_right * gini_right) / n
        parent_gini = gini(np.bincount(y, minlength=2))
        gains = parent_gini - weighted_gini
        gains[~valid] = -1.0

        # Find best threshold for this feature (threshold t means vals <= t go left)
        # We consider thresholds 0..254 (split between t and t+1)
        # The split after value t corresponds to cum_left[t]
        # So threshold t means left includes values <= t
        local_best_idx = np.argmax(gains[:255])  # only up to 254
        local_best_gain = gains[local_best_idx]
        if local_best_gain > best_gain:
            best_gain = local_best_gain
            best_feature = f
            best_threshold = local_best_idx
            # Build left mask for this split
            best_left_mask = (vals <= best_threshold)

    if best_gain <= 0 or best_feature == -1:
        return make_leaf(y)

    # Split
    left_idx = np.where(best_left_mask)[0]
    right_idx = np.where(~best_left_mask)[0]

    node_idx = len(features)
    # Placeholder for children
    features.append(best_feature)
    thresholds.append(best_threshold)
    lefts.append(-1)
    rights.append(-1)
    leaf_classes.append(-1)

    left_child = build_tree(X[left_idx], y[left_idx], max_depth, depth + 1)
    right_child = build_tree(X[right_idx], y[right_idx], max_depth, depth + 1)

    lefts[node_idx] = left_child
    rights[node_idx] = right_child
    return node_idx


def make_leaf(y):
    if len(y) == 0:
        cls = 0
    else:
        # Majority class, tie break by smaller class
        counts = np.bincount(y, minlength=2)
        cls = int(np.argmax(counts))
    node_idx = len(features)
    features.append(-1)
    thresholds.append(-1)
    lefts.append(-1)
    rights.append(-1)
    leaf_classes.append(cls)
    return node_idx


def evaluate_tree(X, y, root=0):
    n = len(y)
    preds = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        node = root
        xi = X[i]
        while leaf_classes[node] == -1:
            f = features[node]
            t = thresholds[node]
            if xi[f] <= t:
                node = lefts[node]
            else:
                node = rights[node]
        preds[i] = leaf_classes[node]
    return np.mean(preds == y)


def main():
    global features, thresholds, lefts, rights, leaf_classes
    features, thresholds, lefts, rights, leaf_classes = [], [], [], [], []

    card = json.load(open(sys.argv[1]))
    train_in, train_key, weights_out = sys.argv[2], sys.argv[3], sys.argv[4]

    X = np.frombuffer(b"".join(records(train_in)), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(train_key)], dtype=np.uint8)

    # Split: last 20% for tuning
    n = len(X)
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]

    best_depth = 1
    best_acc = -1.0
    for depth in MAX_DEPTH_CANDIDATES:
        features, thresholds, lefts, rights, leaf_classes = [], [], [], [], []
        build_tree(train_X, train_y, depth)
        acc = evaluate_tree(tune_X, tune_y)
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # Train on full data with best depth
    features, thresholds, lefts, rights, leaf_classes = [], [], [], [], []
    build_tree(X, y, best_depth)
    num_nodes = len(features)

    # Write weights
    # Header: magic, d, num_nodes, num_classes
    magic = 0x44545245  # 'DTRE' in little-endian ASCII? Actually 'E','R','T','D' -> 0x45525444, but we just need a unique number
    d = card["input"]["bytes"]
    num_classes = card["classes"]
    header = struct.pack("<4I", magic, d, num_nodes, num_classes)
    # Nodes: each node: feature (uint8), threshold (uint8), left (uint32), right (uint32), leaf_class (uint8)
    # We'll pack as 1+1+4+4+1 = 11 bytes, but align to 4? We'll write as struct with 1B,1B,4B,4B,1B and pad to 16? Simpler: write as 5 int32.
    # Use int32 for all: feature, threshold, left, right, leaf_class (with -1 for non-leaf)
    node_data = bytearray()
    for i in range(num_nodes):
        node_data += struct.pack("<5i", features[i], thresholds[i], lefts[i], rights[i], leaf_classes[i])

    with open(weights_out, "wb") as f:
        f.write(header)
        f.write(node_data)


if __name__ == "__main__":
    main()
```

```c
/* Decision tree inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class index.
 * Weights format:
 *   uint32_t magic (0x44545245)
 *   uint32_t d (14)
 *   uint32_t num_nodes
 *   uint32_t num_classes (2)
 *   For each node (num_nodes times):
 *     int32_t feature (0..d-1, or -1 for leaf)
 *     int32_t threshold (0..255, or -1 for leaf)
 *     int32_t left (child index, or -1 for leaf)
 *     int32_t right (child index, or -1 for leaf)
 *     int32_t leaf_class (0 or 1 for leaf, -1 otherwise)
 */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x44545245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], num_nodes = hd[2], num_classes = hd[3];
    if (d != 14 || num_classes != 2 || num_nodes <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + (long)num_nodes * 20; // 5 int32 per node = 20 bytes
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    int32_t *nodes = hd + 4; // each node: feature, threshold, left, right, leaf_class
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (1) {
            int32_t feature = nodes[node * 5 + 0];
            int32_t threshold = nodes[node * 5 + 1];
            int32_t left = nodes[node * 5 + 2];
            int32_t right = nodes[node * 5 + 3];
            int32_t leaf_class = nodes[node * 5 + 4];
            if (leaf_class != -1) {
                unsigned char out = (unsigned char)leaf_class;
                put_rec(stdout, &out, 1);
                break;
            }
            if (xi[feature] <= threshold) node = left;
            else node = right;
        }
    }
    return 0;
}
```