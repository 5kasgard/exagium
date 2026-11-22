```hypothesis
Family: label-forest (random forest of shallow decision trees).
Why: Optdigits pixels are ordinal (0..16) and spatially correlated; trees capture non-linear interactions efficiently. A small forest (10 trees, depth 5) should exceed linear accuracy (96.1%) with very low inference cost (few comparisons per tree).
Failure: If dev accuracy <= 0.9613 or instructions per item exceed label-linear (5058 Ir/item).
```

```python
"""Random forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Uses tuning split (last 20%) to pick n_trees and max_depth.
Exports a compact binary format for fast inference."""
import json, math, struct, sys
import numpy as np

SEED = 1
TUNE_FRAC = 0.2
# Hyperparameter grid for tuning
GRID = [
    {"n_trees": 5, "max_depth": 3},
    {"n_trees": 10, "max_depth": 3},
    {"n_trees": 10, "max_depth": 4},
    {"n_trees": 10, "max_depth": 5},
    {"n_trees": 20, "max_depth": 4},
    {"n_trees": 20, "max_depth": 5},
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class DecisionTree:
    def __init__(self, max_depth, n_classes, n_features, rng):
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.n_features = n_features
        self.rng = rng
        self.nodes = []  # each node: (is_leaf, feature, threshold, left, right, class)

    def gini(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes)
        p = counts / len(y)
        return 1.0 - np.sum(p * p)

    def best_split(self, X, y, feature_indices):
        best_gini = float('inf')
        best_feat = -1
        best_thresh = -1
        for f in feature_indices:
            vals = X[:, f]
            # thresholds are midpoints between unique sorted values
            uniq = np.unique(vals)
            if len(uniq) <= 1:
                continue
            for t in (uniq[:-1] + uniq[1:]) / 2:
                left = vals <= t
                if not np.any(left) or not np.any(~left):
                    continue
                gini_left = self.gini(y[left])
                gini_right = self.gini(y[~left])
                gini = (np.sum(left) * gini_left + np.sum(~left) * gini_right) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feat = f
                    best_thresh = t
        return best_feat, best_thresh, best_gini

    def build(self, X, y, depth):
        node_idx = len(self.nodes)
        # stopping criteria
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2:
            cls = np.argmax(np.bincount(y, minlength=self.n_classes))
            self.nodes.append((1, 0, 0, 0, 0, cls))
            return node_idx
        # random feature subset
        n_sub = max(1, int(math.sqrt(self.n_features)))
        feat_idx = self.rng.choice(self.n_features, n_sub, replace=False)
        feat, thresh, gini = self.best_split(X, y, feat_idx)
        if feat == -1:
            cls = np.argmax(np.bincount(y, minlength=self.n_classes))
            self.nodes.append((1, 0, 0, 0, 0, cls))
            return node_idx
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        left_idx = self.build(X[left_mask], y[left_mask], depth + 1)
        right_idx = self.build(X[right_mask], y[right_mask], depth + 1)
        # threshold is integer? We store as uint8, so round to nearest int (0..16)
        thresh_int = int(round(thresh))
        thresh_int = max(0, min(16, thresh_int))
        self.nodes.append((0, feat, thresh_int, left_idx, right_idx, 0))
        return node_idx

    def fit(self, X, y):
        self.nodes = []
        self.build(X, y, 0)
        return self

    def predict_one(self, x):
        idx = 0
        while True:
            is_leaf, feat, thresh, left, right, cls = self.nodes[idx]
            if is_leaf:
                return cls
            if x[feat] <= thresh:
                idx = left
            else:
                idx = right

def train_forest(X, y, n_trees, max_depth, n_classes, rng):
    trees = []
    n_samples = X.shape[0]
    for i in range(n_trees):
        # bootstrap sample
        idx = rng.choice(n_samples, n_samples, replace=True)
        tree = DecisionTree(max_depth, n_classes, X.shape[1], rng)
        tree.fit(X[idx], y[idx])
        trees.append(tree)
    return trees

def evaluate_forest(trees, X, y):
    correct = 0
    for i in range(len(y)):
        votes = np.zeros(10, dtype=int)
        for tree in trees:
            votes[tree.predict_one(X[i])] += 1
        pred = np.argmax(votes)
        if pred == y[i]:
            correct += 1
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_classes = card["classes"]
    n_features = card["input"]["bytes"]

    # split into train_main (first 80%) and tune (last 20%)
    n = len(X)
    split = int(n * (1 - TUNE_FRAC))
    X_main, y_main = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_trees = None

    for cfg in GRID:
        trees = train_forest(X_main, y_main, cfg["n_trees"], cfg["max_depth"], n_classes, rng)
        acc = evaluate_forest(trees, X_tune, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_trees = trees

    # retrain on all data with best config
    rng_all = np.random.default_rng(SEED)
    final_trees = train_forest(X, y, best_cfg["n_trees"], best_cfg["max_depth"], n_classes, rng_all)

    # serialize
    # magic: b'FOR1' (0x31524F46 little-endian)
    magic = 0x31524F46
    version = 1
    n_trees = best_cfg["n_trees"]
    out = struct.pack("<IIIII", magic, version, n_trees, n_classes, n_features)
    for tree in final_trees:
        n_nodes = len(tree.nodes)
        out += struct.pack("<I", n_nodes)
        for node in tree.nodes:
            is_leaf, feat, thresh, left, right, cls = node
            out += struct.pack("<BBBiiB", is_leaf, feat, thresh, left, right, cls)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Random forest inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the majority vote class.
 * Weights format: magic(4), version(4), n_trees(4), n_classes(4), n_features(4),
 * then per tree: n_nodes(4), then n_nodes * (is_leaf u8, feature u8, threshold u8,
 * left i32, right i32, class u8). */
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

static uint32_t read_u32(const unsigned char **p) {
    uint32_t v = (*p)[0] | (*p)[1] << 8 | (*p)[2] << 16 | (uint32_t)(*p)[3] << 24;
    *p += 4;
    return v;
}

static int32_t read_i32(const unsigned char **p) {
    int32_t v = (int32_t)read_u32(p);
    return v;
}

static uint8_t read_u8(const unsigned char **p) {
    uint8_t v = **p;
    *p += 1;
    return v;
}

typedef struct {
    uint8_t is_leaf;
    uint8_t feature;
    uint8_t threshold;
    int32_t left;
    int32_t right;
    uint8_t cls;
} Node;

typedef struct {
    int n_nodes;
    Node *nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    const unsigned char *wp = wb;
    uint32_t magic = read_u32(&wp);
    if (magic != 0x31524F46) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t version = read_u32(&wp);
    if (version != 1) { fprintf(stderr, "bad weights version\n"); return 3; }
    int n_trees = (int)read_u32(&wp);
    int n_classes = (int)read_u32(&wp);
    int n_features = (int)read_u32(&wp);
    if (n_classes < 1 || n_classes > 256 || n_features != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }

    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (int t = 0; t < n_trees; t++) {
        int n_nodes = (int)read_u32(&wp);
        trees[t].n_nodes = n_nodes;
        trees[t].nodes = malloc(n_nodes * sizeof(Node));
        for (int i = 0; i < n_nodes; i++) {
            Node *nd = &trees[t].nodes[i];
            nd->is_leaf = read_u8(&wp);
            nd->feature = read_u8(&wp);
            nd->threshold = read_u8(&wp);
            nd->left = read_i32(&wp);
            nd->right = read_i32(&wp);
            nd->cls = read_u8(&wp);
        }
    }
    free(wb);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    int *votes = malloc(n_classes * sizeof(int));
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_features); return 3; }
        memset(votes, 0, n_classes * sizeof(int));
        for (int t = 0; t < n_trees; t++) {
            int idx = 0;
            while (1) {
                Node *nd = &trees[t].nodes[idx];
                if (nd->is_leaf) {
                    votes[nd->cls]++;
                    break;
                }
                if (xi[nd->feature] <= nd->threshold)
                    idx = nd->left;
                else
                    idx = nd->right;
            }
        }
        int best = 0;
        for (int c = 1; c < n_classes; c++)
            if (votes[c] > votes[best]) best = c;
        out_byte = (unsigned char)best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```