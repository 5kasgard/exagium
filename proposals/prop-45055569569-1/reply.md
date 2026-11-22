```hypothesis
Propose label-rf: a small random forest (10 trees, max depth 10) with bootstrap sampling and sqrt(64)=8 features per split. Random forests often match gradient boosting on small tabular data with less tuning and inference is just a few dozen comparisons per tree. On optdigits, 10 trees of depth 10 should beat linear (0.9613) and approach gbdt (0.9791) at lower instruction cost than gbdt. If dev accuracy ≤ 0.9613 or instructions per item exceed label-gbdt (41742), the family does not meet its goal.
```

```python
"""label-rf: Random forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning (last 20% of train.in)
N_TREES_GRID = [5, 10, 20]
MAX_DEPTH_GRID = [5, 8, 10, 12]
MIN_SAMPLES_LEAF_GRID = [1, 5, 10]
MAX_FEATURES = 8  # sqrt(64)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class DecisionTree:
    def __init__(self, max_depth, min_samples_leaf, max_features, rng):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng
        self.nodes = []  # each node: (feature, threshold, left, right) or ( -1, class, -1, -1 ) for leaf

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self._grow(X, y, list(range(X.shape[0])), depth=0)

    def _grow(self, X, y, idxs, depth):
        n = len(idxs)
        if n < self.min_samples_leaf or depth == self.max_depth or len(np.unique(y[idxs])) == 1:
            # leaf
            counts = np.bincount(y[idxs], minlength=self.n_classes)
            cls = int(np.argmax(counts))
            self.nodes.append((-1, cls, -1, -1))
            return len(self.nodes) - 1

        # choose best split
        best_gini = 1.0
        best_feat = -1
        best_thresh = -1
        best_left = best_right = None

        features = self.rng.choice(X.shape[1], self.max_features, replace=False)
        for f in features:
            vals = X[idxs, f]
            unique_vals = np.unique(vals)
            if len(unique_vals) == 1:
                continue
            # try thresholds between unique values
            for thresh in unique_vals[:-1]:
                left_mask = vals <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_idxs = np.array(idxs)[left_mask]
                right_idxs = np.array(idxs)[right_mask]
                if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
                    continue
                gini = self._gini_impurity(y[left_idxs], y[right_idxs])
                if gini < best_gini:
                    best_gini = gini
                    best_feat = f
                    best_thresh = thresh
                    best_left = left_idxs
                    best_right = right_idxs

        if best_feat == -1:
            counts = np.bincount(y[idxs], minlength=self.n_classes)
            cls = int(np.argmax(counts))
            self.nodes.append((-1, cls, -1, -1))
            return len(self.nodes) - 1

        node_idx = len(self.nodes)
        self.nodes.append((best_feat, best_thresh, -1, -1))  # placeholder
        left_child = self._grow(X, y, best_left.tolist(), depth + 1)
        right_child = self._grow(X, y, best_right.tolist(), depth + 1)
        self.nodes[node_idx] = (best_feat, best_thresh, left_child, right_child)
        return node_idx

    def _gini_impurity(self, y_left, y_right):
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        def gini(arr):
            if len(arr) == 0:
                return 0.0
            _, counts = np.unique(arr, return_counts=True)
            p = counts / len(arr)
            return 1.0 - np.sum(p * p)
        return (n_left * gini(y_left) + n_right * gini(y_right)) / n_total

    def predict_one(self, x):
        node = 0
        while True:
            feat, thresh, left, right = self.nodes[node]
            if feat == -1:
                return thresh
            if x[feat] <= thresh:
                node = left
            else:
                node = right

    def serialize(self):
        # returns bytes: n_nodes (int32), then for each node: feature (int32), threshold (int32), left (int32), right (int32)
        out = struct.pack("<i", len(self.nodes))
        for node in self.nodes:
            out += struct.pack("<iiii", *node)
        return out

def train_forest(X, y, n_trees, max_depth, min_samples_leaf, max_features, base_rng):
    trees = []
    n = X.shape[0]
    for t in range(n_trees):
        # deterministic seed per tree
        tree_rng = np.random.default_rng(base_rng.integers(2**32))
        # bootstrap sample
        idxs = tree_rng.choice(n, n, replace=True)
        tree = DecisionTree(max_depth, min_samples_leaf, max_features, tree_rng)
        tree.fit(X[idxs], y[idxs])
        trees.append(tree)
    return trees

def evaluate_forest(trees, X, y):
    correct = 0
    for i in range(X.shape[0]):
        votes = {}
        for tree in trees:
            cls = tree.predict_one(X[i])
            votes[cls] = votes.get(cls, 0) + 1
        pred = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if pred == y[i]:
            correct += 1
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    # split tuning (last 20%) and training (first 80%)
    n = X_all.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]

    base_rng = np.random.default_rng(SEED)

    best_acc = -1.0
    best_params = None
    for n_trees in N_TREES_GRID:
        for max_depth in MAX_DEPTH_GRID:
            for min_leaf in MIN_SAMPLES_LEAF_GRID:
                trees = train_forest(X_train, y_train, n_trees, max_depth, min_leaf, MAX_FEATURES, base_rng)
                acc = evaluate_forest(trees, X_tune, y_tune)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_trees, max_depth, min_leaf)
    print(f"Best tuning acc: {best_acc:.4f} with params: {best_params}", file=sys.stderr)

    # retrain on all data with best params
    n_trees, max_depth, min_leaf = best_params
    final_trees = train_forest(X_all, y_all, n_trees, max_depth, min_leaf, MAX_FEATURES, base_rng)

    # serialize weights
    # magic: 0x52465354 ('RFST' little-endian)
    # version: 1
    # n_trees, n_features, n_classes
    # then each tree serialized
    out = struct.pack("<iiiii", 0x52465354, 1, n_trees, d, C)
    for tree in final_trees:
        out += tree.serialize()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-rf inference: ./infer weights.bin in.rec N
 * Random forest with binary trees. Each input record is d bytes; output is one byte class.
 * Weights format: magic 0x52465354, version, n_trees, d, C; then per tree: n_nodes, nodes (feat, thresh, left, right all int32).
 * Leaf node: feat=-1, thresh=class, left=right=-1. */
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
    int32_t feat;
    int32_t thresh;
    int32_t left;
    int32_t right;
} Node;

typedef struct {
    int32_t n_nodes;
    Node *nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x52465354 || hd[1] != 1) { fprintf(stderr, "bad weights magic/version\n"); return 3; }
    int32_t n_trees = hd[2], d = hd[3], C = hd[4];
    if (n_trees <= 0 || d != 64 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }

    unsigned char *p = wb + 20;
    long remaining = nw - 20;
    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (int t = 0; t < n_trees; t++) {
        if (remaining < 4) { fprintf(stderr, "truncated weights\n"); return 3; }
        int32_t n_nodes = *(int32_t *)p;
        p += 4; remaining -= 4;
        if (remaining < (long)n_nodes * 16) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t].n_nodes = n_nodes;
        trees[t].nodes = (Node *)p;
        p += n_nodes * 16;
        remaining -= n_nodes * 16;
    }
    if (remaining != 0) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *votes = malloc(C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(votes, 0, C);
        for (int t = 0; t < n_trees; t++) {
            Node *nodes = trees[t].nodes;
            int32_t node = 0;
            while (1) {
                int32_t feat = nodes[node].feat;
                int32_t thresh = nodes[node].thresh;
                if (feat == -1) {
                    votes[thresh]++;
                    break;
                }
                if (xi[feat] <= thresh)
                    node = nodes[node].left;
                else
                    node = nodes[node].right;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (votes[c] > votes[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```