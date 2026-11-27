```hypothesis
Family: label-rf (random forest). Random forests capture non-linear pixel interactions and are strong on small tabular/image data. With 10–30 trees of depth ≤10, inference requires only a few hundred integer comparisons per sample, so CPU cost should stay near label-linear. We tune n_trees, max_depth, min_samples_leaf on the last 20% of train.in. Failure: dev accuracy ≤0.965 or instructions/item >2× label-linear.
```
```python
"""Random forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = {
    'n_trees': [10, 20, 30],
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [8],  # sqrt(64) = 8
}

MAGIC = 0x46524F46  # "FORF" little-endian

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
        self.nodes = []  # list of dicts: {'leaf': bool, 'feature': int, 'thresh': int, 'left': int, 'right': int, 'class': int}

    def gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p * p)

    def best_split(self, X, y, feat_idxs):
        best_gini = 1.0
        best_feat = -1
        best_thresh = -1
        for f in feat_idxs:
            vals = np.unique(X[:, f])
            if len(vals) <= 1:
                continue
            # thresholds midway between adjacent values
            threshes = (vals[:-1] + vals[1:]) // 2
            for t in threshes:
                left = y[X[:, f] <= t]
                right = y[X[:, f] > t]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                g = (len(left) * self.gini(left) + len(right) * self.gini(right)) / len(y)
                if g < best_gini:
                    best_gini = g
                    best_feat = f
                    best_thresh = t
        return best_feat, best_thresh, best_gini

    def build(self, X, y, depth=0):
        node_idx = len(self.nodes)
        self.nodes.append(None)  # placeholder
        # stopping criteria
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or self.gini(y) == 0.0:
            cls = int(np.bincount(y).argmax())
            self.nodes[node_idx] = {'leaf': True, 'class': cls}
            return node_idx
        # choose random subset of features
        n_feats = X.shape[1]
        feat_idxs = self.rng.choice(n_feats, self.max_features, replace=False)
        feat, thresh, _ = self.best_split(X, y, feat_idxs)
        if feat == -1:
            cls = int(np.bincount(y).argmax())
            self.nodes[node_idx] = {'leaf': True, 'class': cls}
            return node_idx
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        left_idx = self.build(X[left_mask], y[left_mask], depth + 1)
        right_idx = self.build(X[right_mask], y[right_mask], depth + 1)
        self.nodes[node_idx] = {'leaf': False, 'feature': feat, 'thresh': thresh, 'left': left_idx, 'right': right_idx}
        return node_idx

    def fit(self, X, y):
        self.nodes = []
        self.build(X, y)
        return self

    def predict_one(self, x):
        idx = 0
        while True:
            node = self.nodes[idx]
            if node['leaf']:
                return node['class']
            if x[node['feature']] <= node['thresh']:
                idx = node['left']
            else:
                idx = node['right']

    def serialize(self):
        # each node: 16 bytes (uint8 is_leaf, uint8 feature, uint8 thresh, uint8 pad, int32 left, int32 right, uint8 class, uint8 pad[3])
        out = bytearray()
        for node in self.nodes:
            if node['leaf']:
                out += struct.pack('<BBBBiiBxxx', 1, 0, 0, 0, -1, -1, node['class'])
            else:
                out += struct.pack('<BBBBiiBxxx', 0, node['feature'], node['thresh'], 0, node['left'], node['right'], 0)
        return bytes(out)

class RandomForest:
    def __init__(self, n_trees, max_depth, min_samples_leaf, max_features, seed):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        n = len(y)
        self.trees = []
        for i in range(self.n_trees):
            rng = np.random.default_rng(self.seed + i)
            # bootstrap sample
            idx = rng.integers(0, n, n)
            Xb, yb = X[idx], y[idx]
            tree = DecisionTree(self.max_depth, self.min_samples_leaf, self.max_features, rng)
            tree.fit(Xb, yb)
            self.trees.append(tree)
        return self

    def predict(self, X):
        # majority vote
        preds = np.zeros((len(X), self.n_trees), dtype=np.uint8)
        for t, tree in enumerate(self.trees):
            for i, x in enumerate(X):
                preds[i, t] = tree.predict_one(x)
        # vote
        out = np.zeros(len(X), dtype=np.uint8)
        for i in range(len(X)):
            counts = np.bincount(preds[i], minlength=10)
            out[i] = counts.argmax()  # ties broken by smallest index
        return out

    def serialize(self):
        # header: magic, n_trees, n_features, n_classes
        # then each tree: n_nodes (uint32), node data
        out = struct.pack('<IIII', MAGIC, self.n_trees, 64, 10)
        for tree in self.trees:
            nodes_bytes = tree.serialize()
            n_nodes = len(nodes_bytes) // 16
            out += struct.pack('<I', n_nodes)
            out += nodes_bytes
        return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = len(y)
    # tuning split: last 20%
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_acc = -1.0
    best_params = None
    rng = np.random.default_rng(SEED)

    for n_trees in GRID['n_trees']:
        for max_depth in GRID['max_depth']:
            for min_samples_leaf in GRID['min_samples_leaf']:
                for max_features in GRID['max_features']:
                    rf = RandomForest(n_trees, max_depth, min_samples_leaf, max_features, SEED)
                    rf.fit(X_train, y_train)
                    preds = rf.predict(X_tune)
                    acc = np.mean(preds == y_tune)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (n_trees, max_depth, min_samples_leaf, max_features)

    # train on all data with best params
    n_trees, max_depth, min_samples_leaf, max_features = best_params
    rf = RandomForest(n_trees, max_depth, min_samples_leaf, max_features, SEED)
    rf.fit(X, y)

    weights = rf.serialize()
    open(sys.argv[4], "wb").write(weights)

if __name__ == "__main__":
    main()
```
```c
/* Random forest inference for label tasks: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixel counts). Output: one byte class (0-9). */
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
    uint8_t is_leaf;
    uint8_t feature;
    uint8_t thresh;
    uint8_t pad;
    int32_t left;
    int32_t right;
    uint8_t class_label;
    uint8_t pad2[3];
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)w;
    if (hd[0] != 0x46524F46) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t n_trees = hd[1], n_features = hd[2], n_classes = hd[3];
    if (n_features != 64 || n_classes != 10) { fprintf(stderr, "bad header dims\n"); return 3; }

    // parse trees
    const unsigned char *ptr = w + 16;
    const unsigned char *end = w + nw;
    Node **trees = malloc(n_trees * sizeof(Node *));
    uint32_t *tree_nnodes = malloc(n_trees * sizeof(uint32_t));
    for (uint32_t t = 0; t < n_trees; t++) {
        if (ptr + 4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t nn = *(uint32_t *)ptr; ptr += 4;
        tree_nnodes[t] = nn;
        if (ptr + (long)nn * 16 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t] = (Node *)ptr;
        ptr += (long)nn * 16;
    }
    if (ptr != end) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    uint8_t vote[10];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        memset(vote, 0, 10);
        for (uint32_t t = 0; t < n_trees; t++) {
            const Node *nodes = trees[t];
            uint32_t idx = 0;
            while (1) {
                const Node *node = &nodes[idx];
                if (node->is_leaf) {
                    vote[node->class_label]++;
                    break;
                }
                if (xi[node->feature] <= node->thresh)
                    idx = node->left;
                else
                    idx = node->right;
            }
        }
        uint8_t best = 0;
        for (int c = 1; c < 10; c++)
            if (vote[c] > vote[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```