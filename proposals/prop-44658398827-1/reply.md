```hypothesis
Family: label-rf (Random Forest). Random forests capture non-linear feature interactions in the 8x8 pixel space better than linear models, and with 10 shallow trees (max_depth=10) they should exceed label-linear's 0.9613 accuracy while keeping inference cost low (a few hundred simple operations per item). If dev accuracy ≤ 0.9613, it indicates axis-aligned splits are insufficient for this dataset or the forest is too small.
```

```python
"""label-rf: Random Forest for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, uses stdlib + numpy only.
Trains an ensemble of CART trees with bootstrap sampling and feature subsampling.
Hyperparameters (n_estimators, max_depth) chosen on last 20% of train (tuning split),
then final model trained on all train data.
Weights format: magic 0x52465246, n_trees, then per tree: num_nodes, root(0),
feature[u8]*n, threshold[u8]*n, left[i32]*n, right[i32]*n, class[u8]*n."""
import json, math, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = 8  # sqrt(64)
PARAM_GRID = {
    'n_estimators': [5, 10, 20],
    'max_depth': [5, 10, 15]
}

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class DecisionTree:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'cls')
    def __init__(self):
        self.feature = []
        self.threshold = []
        self.left = []
        self.right = []
        self.cls = []

    def add_node(self, feature=255, threshold=0, left=-1, right=-1, cls=0):
        self.feature.append(feature)
        self.threshold.append(threshold)
        self.left.append(left)
        self.right.append(right)
        self.cls.append(cls)
        return len(self.feature) - 1

    def fit(self, X, y, max_depth, rng):
        n_samples, n_features = X.shape
        # queue entries: (node_idx, sample_indices, depth)
        root_idx = self.add_node()
        queue = [(root_idx, np.arange(n_samples), 0)]
        while queue:
            node_idx, idxs, depth = queue.pop(0)
            if len(idxs) < MIN_SAMPLES_LEAF or depth >= max_depth:
                self._make_leaf(node_idx, y[idxs])
                continue
            # check purity
            unique, counts = np.unique(y[idxs], return_counts=True)
            if len(unique) == 1:
                self._make_leaf(node_idx, y[idxs])
                continue
            # feature subsampling
            feat_choices = rng.choice(n_features, size=min(MAX_FEATURES, n_features), replace=False)
            best_gain = -1.0
            best_feat = -1
            best_thresh = 0
            parent_gini = self._gini(y[idxs])
            for f in feat_choices:
                vals = X[idxs, f]
                uniq_vals = np.unique(vals)
                if len(uniq_vals) <= 1:
                    continue
                # try thresholds at midpoints
                for v in uniq_vals[:-1]:
                    left_mask = vals <= v
                    right_mask = ~left_mask
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
                    n_left = left_mask.sum()
                    n_right = right_mask.sum()
                    gini_left = self._gini(y[idxs[left_mask]])
                    gini_right = self._gini(y[idxs[right_mask]])
                    gain = parent_gini - (n_left * gini_left + n_right * gini_right) / len(idxs)
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = f
                        best_thresh = v
            if best_gain <= 0:
                self._make_leaf(node_idx, y[idxs])
                continue
            # split
            self.feature[node_idx] = best_feat
            self.threshold[node_idx] = best_thresh
            left_mask = X[idxs, best_feat] <= best_thresh
            left_idxs = idxs[left_mask]
            right_idxs = idxs[~left_mask]
            left_child = self.add_node()
            right_child = self.add_node()
            self.left[node_idx] = left_child
            self.right[node_idx] = right_child
            queue.append((left_child, left_idxs, depth + 1))
            queue.append((right_child, right_idxs, depth + 1))

    def _gini(self, y_sub):
        if len(y_sub) == 0:
            return 0.0
        _, counts = np.unique(y_sub, return_counts=True)
        probs = counts / len(y_sub)
        return 1.0 - np.sum(probs * probs)

    def _make_leaf(self, node_idx, y_sub):
        if len(y_sub) == 0:
            self.cls[node_idx] = 0
        else:
            vals, counts = np.unique(y_sub, return_counts=True)
            self.cls[node_idx] = vals[np.argmax(counts)]
        self.left[node_idx] = -1
        self.right[node_idx] = -1

    def predict_one(self, x):
        node = 0
        while self.left[node] != -1:
            f = self.feature[node]
            if x[f] <= self.threshold[node]:
                node = self.left[node]
            else:
                node = self.right[node]
        return self.cls[node]

def train_forest(X, y, n_estimators, max_depth, rng):
    n_samples = X.shape[0]
    trees = []
    for t in range(n_estimators):
        # bootstrap sample
        boot_idx = rng.integers(0, n_samples, n_samples)
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]
        tree_rng = np.random.default_rng(rng.integers(2**32))
        tree = DecisionTree()
        tree.fit(X_boot, y_boot, max_depth, tree_rng)
        trees.append(tree)
    return trees

def predict_forest(trees, X):
    n = X.shape[0]
    votes = np.zeros((n, 10), dtype=np.int32)
    for tree in trees:
        for i in range(n):
            c = tree.predict_one(X[i])
            votes[i, c] += 1
    return np.argmax(votes, axis=1)

def serialize_forest(trees):
    # magic, n_trees
    out = bytearray()
    out += struct.pack("<II", 0x52465246, len(trees))
    for tree in trees:
        n_nodes = len(tree.feature)
        out += struct.pack("<II", n_nodes, 0)  # root index 0
        out += bytes(tree.feature)
        out += bytes(tree.threshold)
        out += struct.pack("<" + "i" * n_nodes, *tree.left)
        out += struct.pack("<" + "i" * n_nodes, *tree.right)
        out += bytes(tree.cls)
    return bytes(out)

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = len(y_all)
    split = int(0.8 * n_total)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    master_rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_trees = None

    for n_est in PARAM_GRID['n_estimators']:
        for md in PARAM_GRID['max_depth']:
            rng = np.random.default_rng(master_rng.integers(2**32))
            trees = train_forest(X_train, y_train, n_est, md, rng)
            preds = predict_forest(trees, X_val)
            acc = np.mean(preds == y_val)
            if acc > best_acc:
                best_acc = acc
                best_params = (n_est, md)
                best_trees = trees
    print(f"Selected: n_estimators={best_params[0]}, max_depth={best_params[1]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on full data with best params
    final_rng = np.random.default_rng(SEED)
    final_trees = train_forest(X_all, y_all, best_params[0], best_params[1], final_rng)
    weights = serialize_forest(final_trees)
    open(sys.argv[4], "wb").write(weights)

if __name__ == "__main__":
    main()
```

```c
/* label-rf inference: ./infer weights.bin in.rec N
 * Random Forest with axis-aligned splits. Each tree stored as flat arrays.
 * Input record: 64 bytes. Output: one byte class (majority vote). */
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
    int num_nodes;
    const uint8_t *feature;
    const uint8_t *threshold;
    const int32_t *left;
    const int32_t *right;
    const uint8_t *cls;
} Tree;

static int32_t read_i32(const unsigned char *p) {
    return (int32_t)(p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24);
}

static uint32_t read_u32(const unsigned char *p) {
    return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]);
    unsigned char *w = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    if (nw < 8) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = read_u32(w);
    if (magic != 0x52465246) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t n_trees = read_u32(w + 4);
    size_t offset = 8;
    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (uint32_t t = 0; t < n_trees; t++) {
        if (offset + 8 > (size_t)nw) { fprintf(stderr, "truncated tree header\n"); return 3; }
        int32_t num_nodes = read_i32(w + offset); offset += 4;
        int32_t root = read_i32(w + offset); offset += 4;
        if (root != 0) { fprintf(stderr, "root not zero\n"); return 3; }
        if (num_nodes <= 0) { fprintf(stderr, "invalid num_nodes\n"); return 3; }
        size_t need = (size_t)num_nodes * (1 + 1 + 4 + 4 + 1);
        if (offset + need > (size_t)nw) { fprintf(stderr, "truncated tree data\n"); return 3; }
        trees[t].num_nodes = num_nodes;
        trees[t].feature = w + offset; offset += num_nodes;
        trees[t].threshold = w + offset; offset += num_nodes;
        trees[t].left = (const int32_t *)(w + offset); offset += num_nodes * 4;
        trees[t].right = (const int32_t *)(w + offset); offset += num_nodes * 4;
        trees[t].cls = w + offset; offset += num_nodes;
    }
    if (offset != (size_t)nw) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    long pos = 0;
    int *votes = calloc(10, sizeof(int));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        memset(votes, 0, 10 * sizeof(int));
        for (uint32_t t = 0; t < n_trees; t++) {
            const Tree *tr = &trees[t];
            int node = 0;
            while (tr->left[node] != -1) {
                uint8_t f = tr->feature[node];
                if (xi[f] <= tr->threshold[node])
                    node = tr->left[node];
                else
                    node = tr->right[node];
            }
            votes[tr->cls[node]]++;
        }
        unsigned char best = 0;
        for (int c = 1; c < 10; c++)
            if (votes[c] > votes[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(votes);
    free(trees);
    free(w);
    if (x) free(x);
    return 0;
}
```