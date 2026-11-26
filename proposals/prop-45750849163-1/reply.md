```hypothesis
Family: label-tree (single CART decision tree). The 8x8 optdigits pixels are ordinal with only 17 values; a shallow tree can capture local pixel interactions (strokes, loops) that linear models miss. Tuning max_depth and min_samples_leaf on the last 20% of train.in should yield >96.2% dev accuracy. If dev accuracy ≤ label-linear (0.9613) the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
TUNING_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(hist):
    # hist: (classes,) counts
    n = hist.sum()
    if n == 0:
        return 0.0
    p = hist / n
    return 1.0 - np.sum(p * p)

def build_tree(X, y, max_depth, min_leaf, classes):
    n, d = X.shape
    # node: (feature, threshold, left, right, leaf_class)
    # feature = -1 for leaf
    nodes = []
    stack = [(np.arange(n), 0)]  # (indices, depth)
    while stack:
        idx, depth = stack.pop()
        if len(idx) < min_leaf or depth == max_depth:
            # leaf
            hist = np.bincount(y[idx], minlength=classes)
            leaf_class = int(np.argmax(hist))
            nodes.append((-1, 0, 0, 0, leaf_class))
            continue
        # find best split
        best_gain = -1.0
        best_feat = -1
        best_thresh = -1
        parent_hist = np.bincount(y[idx], minlength=classes)
        parent_gini = gini(parent_hist)
        for feat in range(d):
            vals = X[idx, feat]
            # histogram per value (0..16) per class
            hist = np.zeros((17, classes), dtype=np.int32)
            np.add.at(hist, (vals, y[idx]), 1)
            # cumulative from left
            left_hist = np.cumsum(hist[:-1], axis=0)   # thresholds 0..15
            right_hist = parent_hist - left_hist
            left_n = left_hist.sum(axis=1)
            right_n = right_hist.sum(axis=1)
            # avoid empty splits
            mask = (left_n > 0) & (right_n > 0)
            if not np.any(mask):
                continue
            left_gini = np.array([gini(h) for h in left_hist])
            right_gini = np.array([gini(h) for h in right_hist])
            weighted_gini = (left_n * left_gini + right_n * right_gini) / len(idx)
            gain = parent_gini - weighted_gini
            valid_gain = gain[mask]
            if len(valid_gain) == 0:
                continue
            local_best = np.argmax(valid_gain)
            # map back to threshold index
            thresh_indices = np.where(mask)[0]
            thresh = thresh_indices[local_best]
            if valid_gain[local_best] > best_gain:
                best_gain = valid_gain[local_best]
                best_feat = feat
                best_thresh = thresh
        if best_gain <= 1e-9 or best_feat == -1:
            hist = np.bincount(y[idx], minlength=classes)
            leaf_class = int(np.argmax(hist))
            nodes.append((-1, 0, 0, 0, leaf_class))
            continue
        # split
        left_mask = X[idx, best_feat] <= best_thresh
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        # push right then left so left is processed first (depth-first)
        this_node_idx = len(nodes)
        nodes.append((best_feat, best_thresh, 0, 0, 0))  # placeholder
        stack.append((right_idx, depth + 1))
        stack.append((left_idx, depth + 1))
        # after children are added, fix left/right indices
        # left child is next node added (this_node_idx + 1)
        # right child is after left subtree
        # We'll fill after building by traversing again? Simpler: build recursively.
    # The iterative approach above doesn't easily set left/right indices because we don't know sizes beforehand.
    # Let's switch to a recursive builder that returns node index.

    # Re-implement with recursion (depth <= 20 safe)
    sys.setrecursionlimit(10000)
    def build(idx, depth):
        if len(idx) < min_leaf or depth == max_depth:
            hist = np.bincount(y[idx], minlength=classes)
            leaf_class = int(np.argmax(hist))
            node_idx = len(nodes)
            nodes.append((-1, 0, 0, 0, leaf_class))
            return node_idx
        best_gain = -1.0
        best_feat = -1
        best_thresh = -1
        parent_hist = np.bincount(y[idx], minlength=classes)
        parent_gini = gini(parent_hist)
        for feat in range(d):
            vals = X[idx, feat]
            hist = np.zeros((17, classes), dtype=np.int32)
            np.add.at(hist, (vals, y[idx]), 1)
            left_hist = np.cumsum(hist[:-1], axis=0)
            right_hist = parent_hist - left_hist
            left_n = left_hist.sum(axis=1)
            right_n = right_hist.sum(axis=1)
            mask = (left_n > 0) & (right_n > 0)
            if not np.any(mask):
                continue
            left_gini = np.array([gini(h) for h in left_hist])
            right_gini = np.array([gini(h) for h in right_hist])
            weighted_gini = (left_n * left_gini + right_n * right_gini) / len(idx)
            gain = parent_gini - weighted_gini
            valid_gain = gain[mask]
            if len(valid_gain) == 0:
                continue
            local_best = np.argmax(valid_gain)
            thresh_indices = np.where(mask)[0]
            thresh = thresh_indices[local_best]
            if valid_gain[local_best] > best_gain:
                best_gain = valid_gain[local_best]
                best_feat = feat
                best_thresh = thresh
        if best_gain <= 1e-9 or best_feat == -1:
            hist = np.bincount(y[idx], minlength=classes)
            leaf_class = int(np.argmax(hist))
            node_idx = len(nodes)
            nodes.append((-1, 0, 0, 0, leaf_class))
            return node_idx
        left_mask = X[idx, best_feat] <= best_thresh
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        node_idx = len(nodes)
        nodes.append((best_feat, best_thresh, 0, 0, 0))  # placeholder
        left_child = build(left_idx, depth + 1)
        right_child = build(right_idx, depth + 1)
        nodes[node_idx] = (best_feat, best_thresh, left_child, right_child, 0)
        return node_idx

    nodes.clear()
    build(np.arange(n), 0)
    return nodes

def predict_tree(nodes, x):
    # iterative traversal
    node = 0
    while True:
        feat, thresh, left, right, leaf = nodes[node]
        if feat == -1:
            return leaf
        if x[feat] <= thresh:
            node = left
        else:
            node = right

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    # tuning split: last 20%
    split = int(n * (1 - TUNING_FRAC))
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # hyperparameter search
    best_acc = -1.0
    best_params = None
    best_nodes = None
    for max_depth in [5, 8, 10, 12, 15, 20]:
        for min_leaf in [1, 2, 5, 10, 20]:
            nodes = build_tree(X_train, y_train, max_depth, min_leaf, C)
            # evaluate on tuning
            correct = 0
            for i in range(len(X_tune)):
                pred = predict_tree(nodes, X_tune[i])
                if pred == y_tune[i]:
                    correct += 1
            acc = correct / len(X_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_nodes = nodes
    print(f"Best tuning: depth={best_params[0]}, min_leaf={best_params[1]}, acc={best_acc:.4f}", file=sys.stderr)
    # retrain on all data
    final_nodes = build_tree(X, y, best_params[0], best_params[1], C)
    # serialize
    # magic: 0x54524545 ("TREE")
    # num_nodes (int32), max_depth (int32), min_leaf (int32), num_classes (int32)
    # then per node: int8 feature, int8 threshold, int32 left, int32 right, int8 leaf_class
    # feature: -1 for leaf, else 0..63; threshold 0..15; left/right indices; leaf_class 0..9
    magic = 0x54524545
    num_nodes = len(final_nodes)
    out = struct.pack("<5i", magic, num_nodes, best_params[0], best_params[1], C)
    for feat, thresh, left, right, leaf in final_nodes:
        out += struct.pack("<bbiii", feat, thresh, left, right, leaf)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N */
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
    int8_t feature;
    int8_t threshold;
    int32_t left;
    int32_t right;
    int8_t leaf_class;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t num_nodes = hd[1];
    int32_t max_depth = hd[2];
    int32_t min_leaf = hd[3];
    int32_t C = hd[4];
    (void)max_depth; (void)min_leaf; (void)C;
    size_t expected = 20 + (size_t)num_nodes * 12;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 20);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int32_t node = 0;
        while (1) {
            Node nd = nodes[node];
            if (nd.feature == -1) {
                put_rec(stdout, (unsigned char *)&nd.leaf_class, 1);
                break;
            }
            if (xi[nd.feature] <= nd.threshold)
                node = nd.left;
            else
                node = nd.right;
        }
    }
    return 0;
}
```