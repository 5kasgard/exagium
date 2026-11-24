```hypothesis
Family: label-rf (Random Forest). Optdigits has 64 ordinal pixel features with local spatial structure; trees capture interactions without heavy computation. A small forest (10-30 trees, depth 10-15) should exceed linear accuracy (0.961) while staying faster than label-gbdt (41k Ir/item). If dev accuracy ≤ 0.965 or Ir/item > 20k, the approach fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
GRID = {
    "n_trees": [10, 20],
    "max_depth": [5, 10, 15],
    "min_samples_leaf": [1, 5],
    "max_features": [4, 8, 16],
}

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class DecisionTree:
    __slots__ = ("nodes", "n_classes")
    def __init__(self, n_classes):
        self.nodes = []          # each node: (is_leaf, class_or_feat, threshold, left, right)
        self.n_classes = n_classes

    def add_node(self, is_leaf, class_or_feat, threshold=0.0, left=-1, right=-1):
        self.nodes.append((is_leaf, class_or_feat, threshold, left, right))
        return len(self.nodes) - 1

    def build(self, X, y, idx, depth, max_depth, min_samples_leaf, max_features, rng):
        n = len(idx)
        # compute class counts
        counts = np.bincount(y[idx], minlength=self.n_classes)
        majority = int(np.argmax(counts))
        if depth >= max_depth or n <= min_samples_leaf or counts[majority] == n:
            return self.add_node(1, majority)

        # choose random subset of features
        n_feats = X.shape[1]
        feats = rng.choice(n_feats, size=min(max_features, n_feats), replace=False)
        best_gini = 1.0
        best_feat = -1
        best_thresh = 0.0
        best_left = best_right = None

        for f in feats:
            vals = X[idx, f]
            uniq = np.unique(vals)
            if len(uniq) <= 1:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) * 0.5
            for th in thresholds:
                left_mask = vals <= th
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                left_idx = idx[left_mask]
                right_idx = idx[right_mask]
                # gini
                lc = np.bincount(y[left_idx], minlength=self.n_classes)
                rc = np.bincount(y[right_idx], minlength=self.n_classes)
                ln, rn = len(left_idx), len(right_idx)
                gini_l = 1.0 - np.sum((lc / ln) ** 2)
                gini_r = 1.0 - np.sum((rc / rn) ** 2)
                gini = (ln * gini_l + rn * gini_r) / n
                if gini < best_gini:
                    best_gini = gini
                    best_feat = f
                    best_thresh = th
                    best_left = left_idx
                    best_right = right_idx

        if best_feat == -1:
            return self.add_node(1, majority)

        left_child = self.build(X, y, best_left, depth + 1, max_depth, min_samples_leaf, max_features, rng)
        right_child = self.build(X, y, best_right, depth + 1, max_depth, min_samples_leaf, max_features, rng)
        return self.add_node(0, best_feat, float(best_thresh), left_child, right_child)

    def predict_one(self, x):
        node = 0
        while True:
            is_leaf, cof, th, left, right = self.nodes[node]
            if is_leaf:
                return cof
            if x[cof] <= th:
                node = left
            else:
                node = right

def train_forest(X, y, n_trees, max_depth, min_samples_leaf, max_features, rng):
    n = len(X)
    forest = []
    for t in range(n_trees):
        # bootstrap sample
        boot_idx = rng.choice(n, size=n, replace=True)
        tree = DecisionTree(10)
        tree.build(X, y, boot_idx, 0, max_depth, min_samples_leaf, max_features, rng)
        forest.append(tree)
    return forest

def evaluate_forest(forest, X, y):
    correct = 0
    for i in range(len(X)):
        votes = np.zeros(10, dtype=np.int32)
        for tree in forest:
            votes[tree.predict_one(X[i])] += 1
        pred = int(np.argmax(votes))
        if pred == y[i]:
            correct += 1
    return correct / len(X)

def serialize_forest(forest):
    # magic: 0x52463131 ('RF11')
    out = struct.pack("<4I", 0x52463131, len(forest), 10, 64)
    for tree in forest:
        nodes = tree.nodes
        out += struct.pack("<I", len(nodes))
        for is_leaf, cof, th, left, right in nodes:
            out += struct.pack("<BBHfii", is_leaf, cof, 0, th, left, right)
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    tune_size = int(0.2 * n)
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    base_rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_forest = None

    # grid search
    for n_trees in GRID["n_trees"]:
        for max_depth in GRID["max_depth"]:
            for min_samples_leaf in GRID["min_samples_leaf"]:
                for max_features in GRID["max_features"]:
                    # deterministic seed per config
                    seed = SEED + hash((n_trees, max_depth, min_samples_leaf, max_features)) % (2**32)
                    rng = np.random.default_rng(seed)
                    forest = train_forest(X[train_idx], y[train_idx], n_trees, max_depth, min_samples_leaf, max_features, rng)
                    acc = evaluate_forest(forest, X[tune_idx], y[tune_idx])
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (n_trees, max_depth, min_samples_leaf, max_features)
                        best_forest = forest

    # retrain on all data with best params
    n_trees, max_depth, min_samples_leaf, max_features = best_params
    rng = np.random.default_rng(SEED)
    final_forest = train_forest(X, y, n_trees, max_depth, min_samples_leaf, max_features, rng)
    weights = serialize_forest(final_forest)
    open(sys.argv[4], "wb").write(weights)

if __name__ == "__main__":
    main()
```

```c
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
    uint8_t class_or_feat;
    float threshold;
    int32_t left;
    int32_t right;
} Node;

typedef struct {
    uint32_t n_nodes;
    Node *nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x52463131) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t n_trees = hdr[1], n_classes = hdr[2], n_features = hdr[3];
    if (n_classes != 10 || n_features != 64) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t offset = 16;
    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (uint32_t t = 0; t < n_trees; t++) {
        if (offset + 4 > (size_t)nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t n_nodes = *(uint32_t *)(wb + offset); offset += 4;
        if (offset + n_nodes * 16 > (size_t)nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t].n_nodes = n_nodes;
        trees[t].nodes = (Node *)(wb + offset);
        offset += n_nodes * 16;
    }
    if (offset != (size_t)nw) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *sample = malloc(n_features);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, n_features); return 3; }
        memcpy(sample, xi, n_features);
        int votes[10] = {0};
        for (uint32_t t = 0; t < n_trees; t++) {
            uint32_t node = 0;
            while (1) {
                Node nd = trees[t].nodes[node];
                if (nd.is_leaf) {
                    votes[nd.class_or_feat]++;
                    break;
                }
                if (sample[nd.class_or_feat] <= nd.threshold)
                    node = nd.left;
                else
                    node = nd.right;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (votes[c] > votes[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```