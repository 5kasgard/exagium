```hypothesis
We propose a decision tree family (label-tree). Decision trees capture non-linear pixel interactions (strokes, loops) that linear models miss, and inference is just a few integer comparisons per sample. On optdigits, a shallow tree (depth ≤15) should exceed label-linear's 0.9613 accuracy while being faster than label-mlp16. If the data is essentially linearly separable or the tree overfits the small training set, accuracy may not surpass the linear baseline.
```

```python
"""Decision tree family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Tunes max_depth and min_samples_split on last 20% of train,
then retrains on all data. Exports a flat array of nodes (pre-order)."""
import json, math, struct, sys

import numpy as np

SEED = 1
TUNE_FRACTION = 0.2

# Hyperparameter grid for tuning
MAX_DEPTH_GRID = [5, 10, 15, 20]
MIN_SAMPLES_SPLIT_GRID = [2, 5, 10, 20]

MAGIC = 0x44545245  # "TREE" in little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_classes=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes = n_classes
        self.nodes = []  # each node: (is_leaf, class_label, feature, threshold, left, right)

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs * probs)

    def _best_split(self, X, y):
        n, d = X.shape
        best_gain = -1.0
        best_feat = -1
        best_thresh = -1
        parent_gini = self._gini(y)

        for feat in range(d):
            # values are 0..16, thresholds 0..15
            for thresh in range(16):
                mask = X[:, feat] <= thresh
                left_y = y[mask]
                right_y = y[~mask]
                if len(left_y) < self.min_samples_split or len(right_y) < self.min_samples_split:
                    continue
                w_left = len(left_y) / n
                w_right = len(right_y) / n
                gain = parent_gini - (w_left * self._gini(left_y) + w_right * self._gini(right_y))
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
                # tie-breaking: smaller feat, then smaller thresh automatically by loop order

        if best_gain <= 1e-12:
            return None, None, None
        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth):
        # stopping conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            leaf_class = np.argmax(np.bincount(y, minlength=self.n_classes))
            self.nodes.append((1, int(leaf_class), 0, 0, -1, -1))
            return len(self.nodes) - 1

        # if pure
        if len(np.unique(y)) == 1:
            leaf_class = y[0]
            self.nodes.append((1, int(leaf_class), 0, 0, -1, -1))
            return len(self.nodes) - 1

        feat, thresh, gain = self._best_split(X, y)
        if feat is None:
            leaf_class = np.argmax(np.bincount(y, minlength=self.n_classes))
            self.nodes.append((1, int(leaf_class), 0, 0, -1, -1))
            return len(self.nodes) - 1

        mask = X[:, feat] <= thresh
        left_idx = self._build(X[mask], y[mask], depth + 1)
        right_idx = self._build(X[~mask], y[~mask], depth + 1)
        node_idx = len(self.nodes)
        self.nodes.append((0, 0, feat, thresh, left_idx, right_idx))
        return node_idx

    def fit(self, X, y):
        self.nodes = []
        self._build(X, y, 0)
        return self

    def predict(self, X):
        n = X.shape[0]
        preds = np.zeros(n, dtype=np.int64)
        for i in range(n):
            node_idx = 0
            while True:
                is_leaf, class_label, feat, thresh, left, right = self.nodes[node_idx]
                if is_leaf:
                    preds[i] = class_label
                    break
                if X[i, feat] <= thresh:
                    node_idx = left
                else:
                    node_idx = right
        return preds


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = X_all.shape[0]
    n_tune = int(n_total * TUNE_FRACTION)
    # tuning split is last 20% in file order
    X_train, X_tune = X_all[:-n_tune], X_all[-n_tune:]
    y_train, y_tune = y_all[:-n_tune], y_all[-n_tune:]

    best_acc = -1.0
    best_params = None
    rng = np.random.default_rng(SEED)

    for max_depth in MAX_DEPTH_GRID:
        for min_samples_split in MIN_SAMPLES_SPLIT_GRID:
            tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, n_classes=card["classes"])
            tree.fit(X_train, y_train)
            preds = tree.predict(X_tune)
            acc = np.mean(preds == y_tune)
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_params = (max_depth, min_samples_split)

    # retrain on all data with best params
    max_depth, min_samples_split = best_params
    final_tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, n_classes=card["classes"])
    final_tree.fit(X_all, y_all)

    # serialize
    nodes = final_tree.nodes
    num_nodes = len(nodes)
    # header: magic (int32), num_nodes (int32)
    out = struct.pack("<2i", MAGIC, num_nodes)
    for node in nodes:
        is_leaf, class_label, feat, thresh, left, right = node
        out += struct.pack("<6i", is_leaf, class_label, feat, thresh, left, right)

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Decision tree inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the predicted class.
 * Weights format: int32 magic (0x44545245), int32 num_nodes,
 * then num_nodes * 6 int32: is_leaf, class_label, feature, threshold, left, right. */
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
    if (nw < 8 || hd[0] != 0x44545245) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t num_nodes = hd[1];
    if (num_nodes < 1) { fprintf(stderr, "empty tree\n"); return 3; }
    long expected = 8 + (long)num_nodes * 6 * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    // nodes array: each node 6 int32
    int32_t *nodes = hd + 2;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int32_t node_idx = 0;
        while (1) {
            int32_t *node = nodes + node_idx * 6;
            int32_t is_leaf = node[0];
            int32_t class_label = node[1];
            int32_t feature = node[2];
            int32_t threshold = node[3];
            int32_t left = node[4];
            int32_t right = node[5];
            if (is_leaf) {
                unsigned char out = (unsigned char)class_label;
                put_rec(stdout, &out, 1);
                break;
            }
            if (xi[feature] <= threshold) {
                node_idx = left;
            } else {
                node_idx = right;
            }
        }
    }
    return 0;
}
```