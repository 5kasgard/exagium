```hypothesis
Family: label-rf (Random Forest on native 14-byte features).
Why: Adult has 14 columns already quantized to 256 bins; a shallow forest (10 trees, depth 10) captures non-linear interactions without one-hot explosion, beating linear's 0.8524 at far lower cost than GBDT.
Fail signal: dev accuracy ≤ 0.853 or Ir/item ≥ 2000.
```
```python
"""Random Forest for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Features used directly as 14 uint8 columns (0-255).
Trains n_trees trees with bootstrap sampling, max_depth, min_leaf, feature subsampling.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.
Weights format: magic 0x52465354, int32 n_trees, n_features, max_depth; then per tree:
  int32 n_nodes, then n_nodes * (uint8 feature, uint8 thresh, int32 left, int32 right, float prob)."""
import json, math, struct, sys

import numpy as np

SEED = 1
TUNE_FRACTION = 0.2

# Hyperparameter grid for tuning
GRID = {
    "n_trees": [5, 10, 15],
    "max_depth": [5, 8, 10, 12],
    "min_leaf": [1, 5, 10],
    "subsample_feat": [3, 4, 5],  # features per split
}


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class Tree:
    __slots__ = ("feat", "thresh", "left", "right", "prob", "n_nodes")

    def __init__(self, max_depth, min_leaf, n_features, subsample_feat, rng):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_features = n_features
        self.subsample_feat = subsample_feat
        self.rng = rng
        self.feat = []
        self.thresh = []
        self.left = []
        self.right = []
        self.prob = []

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 2.0 * p * (1.0 - p)

    def _best_split(self, X, y, idxs):
        n = len(idxs)
        if n < 2 * self.min_leaf:
            return None, None, None
        parent_gini = self._gini(y[idxs])
        best_gain = 0.0
        best_f = best_t = -1
        features = self.rng.choice(self.n_features, self.subsample_feat, replace=False)
        for f in features:
            vals = X[idxs, f]
            # try thresholds at midpoints between sorted unique values
            uniq = np.unique(vals)
            if len(uniq) <= 1:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) // 2
            for t in thresholds:
                left_mask = vals <= t
                n_left = left_mask.sum()
                n_right = n - n_left
                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue
                gini_left = self._gini(y[idxs[left_mask]])
                gini_right = self._gini(y[idxs[~left_mask]])
                gain = parent_gini - (n_left * gini_left + n_right * gini_right) / n
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_t = t
        if best_gain <= 0:
            return None, None, None
        return best_f, best_t, best_gain

    def _build(self, X, y, idxs, depth):
        node_idx = len(self.feat)
        self.feat.append(0)
        self.thresh.append(0)
        self.left.append(-1)
        self.right.append(-1)
        self.prob.append(y[idxs].mean())
        if depth >= self.max_depth or len(idxs) < 2 * self.min_leaf:
            return node_idx
        f, t, _ = self._best_split(X, y, idxs)
        if f is None:
            return node_idx
        left_mask = X[idxs, f] <= t
        left_idxs = idxs[left_mask]
        right_idxs = idxs[~left_mask]
        self.feat[node_idx] = f
        self.thresh[node_idx] = t
        self.left[node_idx] = self._build(X, y, left_idxs, depth + 1)
        self.right[node_idx] = self._build(X, y, right_idxs, depth + 1)
        return node_idx

    def fit(self, X, y):
        n = X.shape[0]
        # bootstrap sample
        idxs = self.rng.integers(0, n, n, dtype=np.int64)
        self._build(X, y, idxs, 0)
        self.n_nodes = len(self.feat)

    def predict_proba(self, x):
        node = 0
        while self.left[node] != -1:
            if x[self.feat[node]] <= self.thresh[node]:
                node = self.left[node]
            else:
                node = self.right[node]
        return self.prob[node]


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n_total = X.shape[0]
    n_tune = int(n_total * TUNE_FRACTION)
    # tuning split is last 20% in file order
    X_train, X_tune = X[:-n_tune], X[-n_tune:]
    y_train, y_tune = y[:-n_tune], y[-n_tune:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None

    for n_trees in GRID["n_trees"]:
        for max_depth in GRID["max_depth"]:
            for min_leaf in GRID["min_leaf"]:
                for subsample_feat in GRID["subsample_feat"]:
                    # train forest on training split
                    trees = []
                    for _ in range(n_trees):
                        tree_rng = np.random.default_rng(rng.integers(2**31))
                        tree = Tree(max_depth, min_leaf, X.shape[1], subsample_feat, tree_rng)
                        tree.fit(X_train, y_train)
                        trees.append(tree)
                    # evaluate on tuning split
                    preds = np.zeros(len(y_tune), dtype=np.float32)
                    for tree in trees:
                        for i, x in enumerate(X_tune):
                            preds[i] += tree.predict_proba(x)
                    preds /= n_trees
                    acc = ((preds > 0.5) == y_tune).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (n_trees, max_depth, min_leaf, subsample_feat)

    # retrain on full data with best params
    n_trees, max_depth, min_leaf, subsample_feat = best_params
    trees = []
    for _ in range(n_trees):
        tree_rng = np.random.default_rng(rng.integers(2**31))
        tree = Tree(max_depth, min_leaf, X.shape[1], subsample_feat, tree_rng)
        tree.fit(X, y)
        trees.append(tree)

    # serialize
    out = struct.pack("<4i", 0x52465354, n_trees, X.shape[1], max_depth)
    for tree in trees:
        out += struct.pack("<i", tree.n_nodes)
        for i in range(tree.n_nodes):
            out += struct.pack("<BBiiiif", tree.feat[i], tree.thresh[i], tree.left[i], tree.right[i], 0, 0, tree.prob[i])
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Random Forest inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (one per feature). Output: one byte class (0/1). */
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
    uint8_t feat;
    uint8_t thresh;
    int32_t left;
    int32_t right;
    float prob;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t *)wb;
    if (magic != 0x52465354) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_trees = *(int32_t *)(wb + 4);
    int32_t n_feats = *(int32_t *)(wb + 8);
    int32_t max_depth = *(int32_t *)(wb + 12);
    if (n_trees <= 0 || n_trees > 100 || n_feats != 14) { fprintf(stderr, "bad header\n"); return 3; }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *ptr = wb + 16;
    unsigned char *end = wb + nw;
    Node **trees = malloc(n_trees * sizeof(Node *));
    int32_t *n_nodes_arr = malloc(n_trees * sizeof(int32_t));
    for (int t = 0; t < n_trees; t++) {
        if (ptr + 4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        int32_t n_nodes = *(int32_t *)ptr; ptr += 4;
        if (n_nodes <= 0) { fprintf(stderr, "bad n_nodes\n"); return 3; }
        if (ptr + (long)n_nodes * sizeof(Node) > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t] = (Node *)ptr;
        n_nodes_arr[t] = n_nodes;
        ptr += (long)n_nodes * sizeof(Node);
    }
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_feats) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_feats); return 3; }
        float sum = 0.0f;
        for (int t = 0; t < n_trees; t++) {
            int node = 0;
            while (1) {
                Node nd = trees[t][node];
                if (nd.left == -1) { sum += nd.prob; break; }
                if (xi[nd.feat] <= nd.thresh) node = nd.left;
                else node = nd.right;
            }
        }
        unsigned char cls = (sum / n_trees > 0.5f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```