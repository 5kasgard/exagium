```hypothesis
Family: label-rf (random forest of shallow decision trees on raw pixel values).
Why: optdigits has 64 ordinal features (0–16) where axis-aligned splits capture local patterns; an ensemble reduces variance and typically reaches ~0.97–0.98 accuracy. Inference is a few hundred integer comparisons, cheaper than an MLP.
Failure: if dev accuracy < 0.96 or instruction count per item exceeds label-linear (5058).
```

```python
import json, struct, sys, math
import numpy as np

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def train_forest(X, y, n_trees, max_depth, min_samples_leaf, d, C, seed):
    master_rng = np.random.default_rng(seed)
    trees = []
    n = X.shape[0]
    max_features = int(np.sqrt(d))
    for t in range(n_trees):
        # deterministic per-tree seed
        tree_seed = int(master_rng.integers(0, 2**31))
        rng = np.random.default_rng(tree_seed)
        # bootstrap sample
        indices = rng.integers(0, n, n)
        X_boot = X[indices]
        y_boot = y[indices]
        # feature subset for this tree
        feat_subset = rng.choice(d, max_features, replace=False)
        feats = []
        threshs = []
        lefts = []
        rights = []
        leaf_classes = []
        def build_tree_rec(X_node, y_node, depth):
            if depth >= max_depth or len(np.unique(y_node)) == 1 or len(y_node) < 2 * min_samples_leaf:
                leaf_class = int(np.bincount(y_node).argmax())
                idx = len(feats)
                feats.append(-1); threshs.append(-1); lefts.append(-1); rights.append(-1); leaf_classes.append(leaf_class)
                return idx
            best_gini = float('inf')
            best_feat = -1
            best_thresh = -1
            for f in feat_subset:
                vals = X_node[:, f]
                unique = np.unique(vals)
                for thresh in unique[:-1]:
                    left_mask = vals <= thresh
                    right_mask = ~left_mask
                    n_left = left_mask.sum()
                    n_right = right_mask.sum()
                    if n_left < min_samples_leaf or n_right < min_samples_leaf:
                        continue
                    # gini
                    left_counts = np.bincount(y_node[left_mask], minlength=C)
                    right_counts = np.bincount(y_node[right_mask], minlength=C)
                    gini_left = 1.0 - np.sum((left_counts / n_left) ** 2)
                    gini_right = 1.0 - np.sum((right_counts / n_right) ** 2)
                    gini = (n_left * gini_left + n_right * gini_right) / (n_left + n_right)
                    if gini < best_gini:
                        best_gini = gini
                        best_feat = f
                        best_thresh = thresh
            if best_feat == -1:
                leaf_class = int(np.bincount(y_node).argmax())
                idx = len(feats)
                feats.append(-1); threshs.append(-1); lefts.append(-1); rights.append(-1); leaf_classes.append(leaf_class)
                return idx
            left_mask = X_node[:, best_feat] <= best_thresh
            right_mask = ~left_mask
            idx = len(feats)
            feats.append(best_feat); threshs.append(best_thresh); lefts.append(-1); rights.append(-1); leaf_classes.append(-1)
            left_idx = build_tree_rec(X_node[left_mask], y_node[left_mask], depth+1)
            right_idx = build_tree_rec(X_node[right_mask], y_node[right_mask], depth+1)
            lefts[idx] = left_idx
            rights[idx] = right_idx
            return idx
        root_idx = build_tree_rec(X_boot, y_boot, 0)
        assert root_idx == 0
        trees.append({
            'feats': np.array(feats, dtype=np.int32),
            'threshs': np.array(threshs, dtype=np.int32),
            'lefts': np.array(lefts, dtype=np.int32),
            'rights': np.array(rights, dtype=np.int32),
            'leaf_classes': np.array(leaf_classes, dtype=np.int32)
        })
    return trees

def predict_forest(trees, X, C):
    n = X.shape[0]
    votes = np.zeros((n, C), dtype=np.int32)
    for tree in trees:
        feats = tree['feats']
        threshs = tree['threshs']
        lefts = tree['lefts']
        rights = tree['rights']
        leaf_classes = tree['leaf_classes']
        for i in range(n):
            node = 0
            while leaf_classes[node] == -1:
                f = feats[node]
                thresh = threshs[node]
                if X[i, f] <= thresh:
                    node = lefts[node]
                else:
                    node = rights[node]
            votes[i, leaf_classes[node]] += 1
    return votes.argmax(axis=1)

def write_weights(path, trees, d, C):
    magic = 0x52463130  # "RF10"
    version = 1
    num_trees = len(trees)
    with open(path, "wb") as f:
        f.write(struct.pack("<iiiii", magic, version, num_trees, d, C))
        for tree in trees:
            num_nodes = len(tree['feats'])
            f.write(struct.pack("<i", num_nodes))
            for arr in (tree['feats'], tree['threshs'], tree['lefts'], tree['rights'], tree['leaf_classes']):
                f.write(arr.astype("<i4").tobytes())

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]
    param_grid = {
        'n_trees': [10, 20],
        'max_depth': [10, 15, 20],
        'min_samples_leaf': [1, 2],
    }
    best_acc = -1.0
    best_params = None
    for n_trees in param_grid['n_trees']:
        for max_depth in param_grid['max_depth']:
            for min_samples_leaf in param_grid['min_samples_leaf']:
                trees = train_forest(X_train, y_train, n_trees, max_depth, min_samples_leaf, d, C, seed=1)
                preds = predict_forest(trees, X_tune, C)
                acc = np.mean(preds == y_tune)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_trees, max_depth, min_samples_leaf)
    n_trees, max_depth, min_samples_leaf = best_params
    trees = train_forest(X, y, n_trees, max_depth, min_samples_leaf, d, C, seed=1)
    write_weights(sys.argv[4], trees, d, C)

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
    int32_t *feats;
    int32_t *threshs;
    int32_t *lefts;
    int32_t *rights;
    int32_t *leaf_classes;
    int32_t num_nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x52463130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t version = hd[1];
    int32_t num_trees = hd[2];
    int32_t num_features = hd[3];
    int32_t num_classes = hd[4];
    if (num_classes < 1 || num_classes > 256) { fprintf(stderr, "bad num_classes\n"); return 3; }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    Tree *trees = calloc(num_trees, sizeof(Tree));
    uint8_t *ptr = wb + 20;
    uint8_t *end = wb + nw;
    for (int t = 0; t < num_trees; t++) {
        if (ptr + 4 > end) { fprintf(stderr, "truncated weights (tree count)\n"); return 3; }
        int32_t num_nodes = *(int32_t *)ptr; ptr += 4;
        trees[t].num_nodes = num_nodes;
        size_t need = (size_t)num_nodes * 5 * 4;
        if (ptr + need > end) { fprintf(stderr, "truncated weights (tree nodes)\n"); return 3; }
        trees[t].feats = (int32_t *)ptr; ptr += num_nodes * 4;
        trees[t].threshs = (int32_t *)ptr; ptr += num_nodes * 4;
        trees[t].lefts = (int32_t *)ptr; ptr += num_nodes * 4;
        trees[t].rights = (int32_t *)ptr; ptr += num_nodes * 4;
        trees[t].leaf_classes = (int32_t *)ptr; ptr += num_nodes * 4;
    }
    if (ptr != end) { fprintf(stderr, "extra bytes in weights\n"); return 3; }
    int32_t *votes = calloc(num_classes, sizeof(int32_t));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != num_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, num_features); return 3; }
        memset(votes, 0, num_classes * sizeof(int32_t));
        for (int t = 0; t < num_trees; t++) {
            Tree *tr = &trees[t];
            int node = 0;
            while (tr->leaf_classes[node] == -1) {
                int f = tr->feats[node];
                int thresh = tr->threshs[node];
                if (xi[f] <= thresh) node = tr->lefts[node];
                else node = tr->rights[node];
            }
            votes[tr->leaf_classes[node]]++;
        }
        int best = 0;
        for (int c = 1; c < num_classes; c++) if (votes[c] > votes[best]) best = c;
        unsigned char out = (unsigned char)best;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```