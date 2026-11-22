```hypothesis
Family: label-rf (random forest). Random forests capture non-linear pixel interactions well and are robust on small image datasets like optdigits. With 20 trees of depth 10 they should exceed label-linear's 0.9613 accuracy while keeping inference fast (a few hundred comparisons per item). If dev accuracy stays ≤0.9613 or inference cost exceeds label-mlp16, the hypothesis fails.
```
```python
"""Random forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys
import numpy as np

SEED = 1
N_TREES = 20
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 5
MIN_SAMPLES_SPLIT = 10
MAX_FEATURES = 8  # sqrt(64)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self, feature=-1, threshold=0, left=-1, right=-1, value=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, feature_idx):
    best_gain = -1.0
    best_thr = 0
    parent_gini = gini(y)
    n = len(y)
    for f in feature_idx:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2
        for thr in thresholds:
            left_mask = vals <= thr
            right_mask = ~left_mask
            nl, nr = np.sum(left_mask), np.sum(right_mask)
            if nl < MIN_SAMPLES_LEAF or nr < MIN_SAMPLES_LEAF:
                continue
            gain = parent_gini - (nl/n)*gini(y[left_mask]) - (nr/n)*gini(y[right_mask])
            if gain > best_gain:
                best_gain = gain
                best_thr = thr
    return best_gain, best_thr

def build_tree(X, y, depth, rng, n_features):
    n_samples = len(y)
    if depth >= MAX_DEPTH or n_samples < MIN_SAMPLES_SPLIT or len(np.unique(y)) == 1:
        leaf_val = np.bincount(y).argmax()
        return Node(value=leaf_val)
    feature_idx = rng.choice(n_features, size=min(MAX_FEATURES, n_features), replace=False)
    gain, thr = best_split(X, y, feature_idx)
    if gain <= 0:
        leaf_val = np.bincount(y).argmax()
        return Node(value=leaf_val)
    feature = feature_idx[np.argmax([gain])]  # gain is scalar, but we need the feature that gave it
    # Actually best_split returns gain for the best feature among the subset; we need to know which feature.
    # Redo: best_split should return feature index as well.
    # Let's refactor best_split to return feature index.
    # We'll do a quick fix: recompute to find feature.
    # But for simplicity, we'll modify best_split to return (feature, threshold, gain).
    # Since we already have the code above, we'll adjust.
    # Let's restart the function with proper return.

# Rewriting build_tree and best_split properly:

def best_split(X, y, feature_idx):
    best_gain = -1.0
    best_f = -1
    best_thr = 0
    parent_gini = gini(y)
    n = len(y)
    for f in feature_idx:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2
        for thr in thresholds:
            left_mask = vals <= thr
            right_mask = ~left_mask
            nl, nr = np.sum(left_mask), np.sum(right_mask)
            if nl < MIN_SAMPLES_LEAF or nr < MIN_SAMPLES_LEAF:
                continue
            gain = parent_gini - (nl/n)*gini(y[left_mask]) - (nr/n)*gini(y[right_mask])
            if gain > best_gain:
                best_gain = gain
                best_f = f
                best_thr = thr
    return best_f, best_thr, best_gain

def build_tree(X, y, depth, rng, n_features):
    n_samples = len(y)
    if depth >= MAX_DEPTH or n_samples < MIN_SAMPLES_SPLIT or len(np.unique(y)) == 1:
        leaf_val = np.bincount(y).argmax()
        return Node(value=leaf_val)
    feature_idx = rng.choice(n_features, size=min(MAX_FEATURES, n_features), replace=False)
    f, thr, gain = best_split(X, y, feature_idx)
    if gain <= 0 or f == -1:
        leaf_val = np.bincount(y).argmax()
        return Node(value=leaf_val)
    left_mask = X[:, f] <= thr
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], depth+1, rng, n_features)
    right = build_tree(X[right_mask], y[right_mask], depth+1, rng, n_features)
    return Node(feature=f, threshold=thr, left=left, right=right)

def flatten_tree(root):
    nodes = []
    def dfs(node):
        idx = len(nodes)
        nodes.append(node)
        if node.feature != -1:
            node.left = dfs(node.left)
            node.right = dfs(node.right)
        return idx
    dfs(root)
    # Convert to arrays
    n = len(nodes)
    left_arr = np.zeros(n, dtype=np.int32)
    right_arr = np.zeros(n, dtype=np.int32)
    feature_arr = np.zeros(n, dtype=np.int32)
    threshold_arr = np.zeros(n, dtype=np.uint8)
    value_arr = np.zeros(n, dtype=np.uint8)
    for i, node in enumerate(nodes):
        if node.feature != -1:
            left_arr[i] = node.left
            right_arr[i] = node.right
            feature_arr[i] = node.feature
            threshold_arr[i] = int(node.threshold)
        else:
            left_arr[i] = -1
            right_arr[i] = -1
            feature_arr[i] = -1
            threshold_arr[i] = 0
            value_arr[i] = node.value
    return left_arr, right_arr, feature_arr, threshold_arr, value_arr

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_samples, n_features = X.shape
    n_classes = card["classes"]
    rng = np.random.default_rng(SEED)

    # Train forest on all data
    trees = []
    for t in range(N_TREES):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X[idx], y[idx]
        tree = build_tree(X_boot, y_boot, 0, rng, n_features)
        trees.append(flatten_tree(tree))

    # Serialize
    MAGIC = 0x52465354  # 'RFST'
    header = struct.pack("<5I", MAGIC, N_TREES, n_features, n_classes, MAX_DEPTH)
    out = bytearray(header)
    for left, right, feature, threshold, value in trees:
        n_nodes = len(left)
        out += struct.pack("<I", n_nodes)
        out += left.astype("<i4").tobytes()
        out += right.astype("<i4").tobytes()
        out += feature.astype("<i4").tobytes()
        out += threshold.astype("<u1").tobytes()
        out += value.astype("<u1").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Random forest inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the majority vote class. */
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
    int32_t *left;
    int32_t *right;
    int32_t *feature;
    uint8_t *threshold;
    uint8_t *value;
    int32_t n_nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)w;
    if (hd[0] != 0x52465354) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t n_trees = hd[1], n_features = hd[2], n_classes = hd[3], max_depth = hd[4];
    if (n_classes < 1 || n_classes > 256 || n_features != 64) { fprintf(stderr, "bad header\n"); return 3; }

    Tree *trees = malloc(n_trees * sizeof(Tree));
    unsigned char *p = w + 20;
    unsigned char *end = w + nw;
    for (uint32_t t = 0; t < n_trees; t++) {
        if (p + 4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t n_nodes = *(uint32_t *)p; p += 4;
        size_t need = (size_t)n_nodes * (4*3 + 1 + 1);
        if (p + need > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t].n_nodes = n_nodes;
        trees[t].left = (int32_t *)p; p += n_nodes * 4;
        trees[t].right = (int32_t *)p; p += n_nodes * 4;
        trees[t].feature = (int32_t *)p; p += n_nodes * 4;
        trees[t].threshold = p; p += n_nodes;
        trees[t].value = p; p += n_nodes;
    }
    if (p != end) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    uint8_t *votes = malloc(n_classes);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, n_features); return 3; }
        memset(votes, 0, n_classes);
        for (uint32_t t = 0; t < n_trees; t++) {
            Tree *tr = &trees[t];
            int32_t node = 0;
            while (tr->feature[node] != -1) {
                int f = tr->feature[node];
                uint8_t thr = tr->threshold[node];
                node = (xi[f] <= thr) ? tr->left[node] : tr->right[node];
            }
            votes[tr->value[node]]++;
        }
        uint8_t best = 0;
        for (uint32_t c = 1; c < n_classes; c++)
            if (votes[c] > votes[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```