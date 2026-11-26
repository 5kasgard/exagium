```hypothesis
We propose a Gradient Boosted Decision Trees (GBDT) family with shallow trees (max depth 3) and a small number of trees (50-100). This captures feature interactions missed by linear models while keeping inference fast (only a few comparisons per tree). It should beat label-linear's 0.8524 accuracy and approach label-gbdt's 0.8664 with lower cost. If dev accuracy is not above 0.855, the model does not justify its complexity.
```
```python
"""GBDT family for label tasks with fixed-size byte items.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = {
    "n_trees": [50, 100, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4],
}
MIN_SAMPLES_LEAF = 10
MAGIC = 0x47424454  # "GBDT"
VERSION = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self, feature=-1, threshold=0, left=-1, right=-1, value=0.0):
        self.feature = feature      # uint8
        self.threshold = threshold  # uint8
        self.left = left            # int32 index in flat array
        self.right = right          # int32 index in flat array
        self.value = value          # float32 (leaf value)

def build_tree(X, g, indices, depth, max_depth, rng):
    """Build a regression tree on gradients g for samples indices.
    Returns list of nodes in flat array (pre-order)."""
    n = len(indices)
    if depth == max_depth or n < MIN_SAMPLES_LEAF:
        val = float(g[indices].mean())
        return [Node(value=val)]
    # Find best split
    best_gain = -1.0
    best_feat = 0
    best_thresh = 0
    total_sum = g[indices].sum()
    total_sq = (g[indices]**2).sum()
    total_mse = total_sq - total_sum**2 / n
    for feat in range(X.shape[1]):
        # Bin gradients by feature value (0-255)
        vals = X[indices, feat]
        # Use bincount for speed
        bin_sum = np.bincount(vals, weights=g[indices], minlength=256)
        bin_cnt = np.bincount(vals, minlength=256)
        # Cumulative sums for thresholds 0..254
        cum_sum = np.cumsum(bin_sum[:-1])   # left sum for threshold t
        cum_cnt = np.cumsum(bin_cnt[:-1])   # left count
        right_sum = total_sum - cum_sum
        right_cnt = n - cum_cnt
        # Avoid empty splits
        mask = (cum_cnt > 0) & (right_cnt > 0)
        if not np.any(mask):
            continue
        left_mse = (np.cumsum(np.bincount(vals, weights=g[indices]**2, minlength=256)[:-1]) - cum_sum**2 / cum_cnt)[mask]
        right_mse = ((np.cumsum(np.bincount(vals, weights=g[indices]**2, minlength=256)[::-1])[::-1][:-1] - right_sum**2 / right_cnt))[mask]
        # Actually simpler: compute MSE reduction directly
        # MSE reduction = total_mse - (left_mse + right_mse)
        # We can compute left_mse and right_mse using the binned sums of squares.
        # Let's do a simpler but correct approach: iterate thresholds where count changes.
        # Since 256 is small, we can just loop.
        for t in range(255):
            left_idx = indices[vals <= t]
            right_idx = indices[vals > t]
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue
            left_sum = g[left_idx].sum()
            right_sum = g[right_idx].sum()
            left_sq = (g[left_idx]**2).sum()
            right_sq = (g[right_idx]**2).sum()
            left_mse = left_sq - left_sum**2 / len(left_idx)
            right_mse = right_sq - right_sum**2 / len(right_idx)
            gain = total_mse - (left_mse + right_mse)
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = t
    if best_gain <= 0:
        val = float(g[indices].mean())
        return [Node(value=val)]
    # Split
    left_indices = indices[X[indices, best_feat] <= best_thresh]
    right_indices = indices[X[indices, best_feat] > best_thresh]
    left_nodes = build_tree(X, g, left_indices, depth+1, max_depth, rng)
    right_nodes = build_tree(X, g, right_indices, depth+1, max_depth, rng)
    # Build flat array: root, then left subtree, then right subtree
    left_start = 1
    right_start = 1 + len(left_nodes)
    root = Node(feature=best_feat, threshold=best_thresh, left=left_start, right=right_start, value=0.0)
    return [root] + left_nodes + right_nodes

def predict_tree(nodes, x):
    """Traverse tree nodes (flat array) for a single sample x (uint8 array)."""
    idx = 0
    while True:
        node = nodes[idx]
        if node.left == -1:  # leaf
            return node.value
        if x[node.feature] <= node.threshold:
            idx = node.left
        else:
            idx = node.right

def train_gbdt(X, y, n_trees, learning_rate, max_depth, rng):
    """Train GBDT for binary classification. Returns (initial_bias, list_of_trees)."""
    n = X.shape[0]
    # Initial bias: log odds
    pos = y.sum()
    initial_bias = math.log((pos + 1e-9) / (n - pos + 1e-9))
    pred = np.full(n, initial_bias, dtype=np.float32)
    trees = []
    for _ in range(n_trees):
        p = 1.0 / (1.0 + np.exp(-pred))
        g = y - p  # gradient of log loss
        # Fit tree to gradients
        indices = np.arange(n)
        nodes = build_tree(X, g, indices, 0, max_depth, rng)
        # Update predictions
        for i in range(n):
            pred[i] += learning_rate * predict_tree(nodes, X[i])
        trees.append(nodes)
    return initial_bias, trees

def evaluate(X, y, initial_bias, trees, learning_rate):
    n = X.shape[0]
    pred = np.full(n, initial_bias, dtype=np.float32)
    for nodes in trees:
        for i in range(n):
            pred[i] += learning_rate * predict_tree(nodes, X[i])
    p = 1.0 / (1.0 + np.exp(-pred))
    y_pred = (p > 0.5).astype(np.int64)
    return (y_pred == y).mean()

def flatten_trees(trees, max_depth):
    """Convert list of tree node lists to flat arrays for storage.
    Returns (features, thresholds, lefts, rights, values) as 1D arrays,
    and max_nodes per tree."""
    max_nodes = (1 << (max_depth + 1)) - 1
    n_trees = len(trees)
    features = np.zeros(n_trees * max_nodes, dtype=np.uint8)
    thresholds = np.zeros(n_trees * max_nodes, dtype=np.uint8)
    lefts = np.full(n_trees * max_nodes, -1, dtype=np.int32)
    rights = np.full(n_trees * max_nodes, -1, dtype=np.int32)
    values = np.zeros(n_trees * max_nodes, dtype=np.float32)
    for t, nodes in enumerate(trees):
        base = t * max_nodes
        for i, node in enumerate(nodes):
            features[base + i] = node.feature
            thresholds[base + i] = node.threshold
            lefts[base + i] = node.left
            rights[base + i] = node.right
            values[base + i] = node.value
    return features, thresholds, lefts, rights, values, max_nodes

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y_bytes = b"".join(records(sys.argv[3]))
    y = np.frombuffer(y_bytes, dtype=np.uint8).astype(np.int64)
    n = X.shape[0]
    # Split: last 20% for tuning
    val_size = n // 5
    train_X, train_y = X[:-val_size], y[:-val_size]
    val_X, val_y = X[-val_size:], y[-val_size:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_trees = None
    best_bias = 0.0
    # Grid search
    for n_trees in GRID["n_trees"]:
        for lr in GRID["learning_rate"]:
            for depth in GRID["max_depth"]:
                bias, trees = train_gbdt(train_X, train_y, n_trees, lr, depth, rng)
                acc = evaluate(val_X, val_y, bias, trees, lr)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_trees, lr, depth)
                    best_trees = trees
                    best_bias = bias
    print(f"Best params: n_trees={best_params[0]}, lr={best_params[1]}, depth={best_params[2]}, val_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on all data with best params
    n_trees, lr, depth = best_params
    bias, trees = train_gbdt(X, y, n_trees, lr, depth, rng)
    # Flatten for storage
    features, thresholds, lefts, rights, values, max_nodes = flatten_trees(trees, depth)
    # Write weights.bin
    # Header: magic, version, n_trees, max_depth, n_features, initial_bias, max_nodes
    header = struct.pack("<IIIIIfI", MAGIC, VERSION, n_trees, depth, d, bias, max_nodes)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(features.tobytes())
        f.write(thresholds.tobytes())
        f.write(lefts.astype("<i4").tobytes())
        f.write(rights.astype("<i4").tobytes())
        f.write(values.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```
```c
/* GBDT inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (0 or 1). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    if (nw < 28) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x47424454) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t version = hd[1];
    if (version != 1) { fprintf(stderr, "bad version\n"); return 3; }
    uint32_t n_trees = hd[2];
    uint32_t max_depth = hd[3];
    uint32_t d = hd[4];
    float initial_bias = *(float *)(hd + 5);
    uint32_t max_nodes = hd[6];
    // Pointers to arrays
    uint8_t *features = wb + 28;
    uint8_t *thresholds = features + n_trees * max_nodes;
    int32_t *lefts = (int32_t *)(thresholds + n_trees * max_nodes);
    int32_t *rights = lefts + n_trees * max_nodes;
    float *values = (float *)(rights + n_trees * max_nodes);
    // Validate size
    size_t expected = 28
        + n_trees * max_nodes * (1 + 1 + 4 + 4 + 4);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != (int)d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        float sum = initial_bias;
        for (uint32_t t = 0; t < n_trees; t++) {
            uint32_t base = t * max_nodes;
            uint32_t idx = 0;
            while (1) {
                int32_t l = lefts[base + idx];
                if (l == -1) { // leaf
                    sum += values[base + idx];
                    break;
                }
                uint8_t feat = features[base + idx];
                uint8_t thresh = thresholds[base + idx];
                if (xi[feat] <= thresh)
                    idx = (uint32_t)l;
                else
                    idx = (uint32_t)rights[base + idx];
            }
        }
        float p = 1.0f / (1.0f + expf(-sum));
        out_byte = (p > 0.5f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```