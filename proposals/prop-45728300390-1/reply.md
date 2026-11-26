```hypothesis
Family: label-rf (random forest of shallow decision trees).
Why: Random forests handle ordinal features well, are robust to overfitting on small datasets, and with a small number of shallow trees (e.g., 10-20 trees, depth 5-10) they can achieve high accuracy while keeping inference extremely fast (only a few integer comparisons per tree). The optdigits dataset has 64 low-cardinality ordinal features (0..16), which are ideal for decision-tree splits.
Failure: If the best forest on the tuning split does not exceed 96.13% accuracy, or if its inference cost (Ir/item) exceeds that of label-linear (5058.6).
```

```python
"""Random forest for label tasks with fixed-size byte items (ordinal features).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, uses only stdlib and numpy.
Chooses hyperparameters on last 20% of train.in (file order), then trains on all data.
Exports forest as flat binary: magic 0x464F5246, n_trees, d, C; then per tree: n_nodes, nodes (feature,threshold,left,right,class)."""
import json, math, struct, sys

import numpy as np

SEED = 1

# Hyperparameter grid for tuning
GRID = {
    "n_trees": [10, 20, 30],
    "max_depth": [5, 7, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features_frac": [0.5, 0.7, 1.0],  # fraction of features
}


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_leaf=1, max_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.nodes = []  # list of tuples (feature, threshold, left, right, class)

    def fit(self, X, y):
        n, d = X.shape
        if self.max_features is None:
            self.max_features = d
        self.nodes = [None]  # root at index 0
        queue = [(0, np.arange(n), 0)]
        while queue:
            node_idx, idxs, depth = queue.pop(0)
            n_idxs = len(idxs)
            # Leaf conditions
            if depth >= self.max_depth or n_idxs <= self.min_samples_leaf or len(np.unique(y[idxs])) == 1:
                classes, counts = np.unique(y[idxs], return_counts=True)
                best_class = int(classes[np.argmax(counts)])
                self.nodes[node_idx] = (-1, 0, -1, -1, best_class)
                continue

            # Choose feature subset
            feat_indices = self.rng.choice(d, size=self.max_features, replace=False)
            best_gini = 1.0
            best_feat = -1
            best_thresh = -1
            best_left = None
            best_right = None

            # For each feature, compute class histograms per value (0..16)
            for feat in feat_indices:
                vals = X[idxs, feat]
                # Histogram: 17 values x C classes
                hist = np.zeros((17, 10), dtype=np.int32)
                for v, c in zip(vals, y[idxs]):
                    hist[v, c] += 1
                # Cumulative sums for thresholds 0..15
                cum = np.cumsum(hist, axis=0)  # cum[t] = sum_{v<=t} hist[v]
                total = cum[-1]
                for t in range(16):
                    left_counts = cum[t]
                    right_counts = total - left_counts
                    n_left = left_counts.sum()
                    n_right = right_counts.sum()
                    if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                        continue
                    # Gini impurity
                    gini_left = 1.0 - np.sum((left_counts / n_left) ** 2)
                    gini_right = 1.0 - np.sum((right_counts / n_right) ** 2)
                    gini = (n_left * gini_left + n_right * gini_right) / n_idxs
                    if gini < best_gini:
                        best_gini = gini
                        best_feat = feat
                        best_thresh = t
                        best_left = idxs[vals <= t]
                        best_right = idxs[vals > t]

            if best_feat == -1:
                classes, counts = np.unique(y[idxs], return_counts=True)
                best_class = int(classes[np.argmax(counts)])
                self.nodes[node_idx] = (-1, 0, -1, -1, best_class)
            else:
                left_node_idx = len(self.nodes)
                self.nodes.append(None)
                right_node_idx = len(self.nodes)
                self.nodes.append(None)
                self.nodes[node_idx] = (best_feat, best_thresh, left_node_idx, right_node_idx, -1)
                queue.append((left_node_idx, best_left, depth + 1))
                queue.append((right_node_idx, best_right, depth + 1))
        return self

    def to_bytes(self):
        out = struct.pack("<i", len(self.nodes))
        for node in self.nodes:
            out += struct.pack("<iiiii", *node)
        return out


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_leaf=1, max_features=None, seed=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trees = []
        self.d = 0
        self.C = 0

    def fit(self, X, y):
        n, d = X.shape
        self.d = d
        self.C = int(y.max() + 1)
        if self.max_features is None:
            self.max_features = int(math.sqrt(d))
        self.trees = []
        for i in range(self.n_trees):
            boot_idx = self.rng.integers(0, n, size=n)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]
            tree_seed = int(self.rng.integers(0, 2**31))
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                seed=tree_seed,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict_one(self, x):
        votes = np.zeros(self.C, dtype=np.int32)
        for tree in self.trees:
            node_idx = 0
            while True:
                feat, thresh, left, right, cls = tree.nodes[node_idx]
                if feat == -1:
                    votes[cls] += 1
                    break
                if x[feat] <= thresh:
                    node_idx = left
                else:
                    node_idx = right
        return int(np.argmax(votes))

    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict_one(X[i]) == y[i]:
                correct += 1
        return correct / len(X)

    def to_bytes(self):
        magic = 0x464F5246  # "FORF"
        header = struct.pack("<4i", magic, self.n_trees, self.d, self.C)
        body = b"".join(tree.to_bytes() for tree in self.trees)
        return header + body


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X_all)
    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    best_acc = -1.0
    best_params = None
    for n_trees in GRID["n_trees"]:
        for max_depth in GRID["max_depth"]:
            for min_samples_leaf in GRID["min_samples_leaf"]:
                for frac in GRID["max_features_frac"]:
                    max_feat = max(1, int(frac * d))
                    rf = RandomForest(
                        n_trees=n_trees,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_feat,
                        seed=SEED,
                    )
                    rf.fit(X_train, y_train)
                    acc = rf.accuracy(X_val, y_val)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (n_trees, max_depth, min_samples_leaf, max_feat)
    # Train final model on all data
    n_trees, max_depth, min_samples_leaf, max_feat = best_params
    rf = RandomForest(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_feat,
        seed=SEED,
    )
    rf.fit(X_all, y_all)
    open(sys.argv[4], "wb").write(rf.to_bytes())


if __name__ == "__main__":
    main()
```

```c
/* Random forest inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the majority-vote class.
 * Weights format: magic 0x464F5246, n_trees, d, C; then per tree: n_nodes, nodes (feature,threshold,left,right,class). */
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
    int32_t feature;
    int32_t threshold;
    int32_t left;
    int32_t right;
    int32_t class_label;
} Node;

typedef struct {
    int32_t n_nodes;
    Node *nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wbuf;
    if (hd[0] != 0x464F5246) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_trees = hd[1], d = hd[2], C = hd[3];
    if (n_trees <= 0 || d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    unsigned char *ptr = wbuf + 16;
    unsigned char *end = wbuf + nw;

    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (int t = 0; t < n_trees; t++) {
        if (ptr + 4 > end) { fprintf(stderr, "truncated tree count\n"); return 3; }
        int32_t n_nodes = *(int32_t *)ptr; ptr += 4;
        if (n_nodes <= 0) { fprintf(stderr, "bad node count\n"); return 3; }
        if (ptr + (long)n_nodes * 20 > end) { fprintf(stderr, "truncated nodes\n"); return 3; }
        trees[t].n_nodes = n_nodes;
        trees[t].nodes = (Node *)ptr;
        ptr += (long)n_nodes * 20;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    int *votes = malloc(C * sizeof(int));
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(votes, 0, C * sizeof(int));
        for (int t = 0; t < n_trees; t++) {
            Node *nodes = trees[t].nodes;
            int node_idx = 0;
            while (1) {
                Node nd = nodes[node_idx];
                if (nd.feature == -1) {
                    votes[nd.class_label]++;
                    break;
                }
                if (xi[nd.feature] <= nd.threshold) node_idx = nd.left;
                else node_idx = nd.right;
            }
        }
        int best = 0;
        for (int c = 1; c < C; c++) if (votes[c] > votes[best]) best = c;
        out_byte = (unsigned char)best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```