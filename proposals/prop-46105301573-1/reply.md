```hypothesis
We propose a random forest family "label-rf". Random forests excel on small tabular datasets like optdigits (64 ordinal features, 10 classes) and typically achieve >98% accuracy, surpassing the linear baseline (0.9613). Inference is fast (a few dozen shallow tree traversals per item) and the model size is small (~200 KiB). If dev accuracy does not beat label-linear, it would indicate that the default hyperparameters (trees, depth, leaf size) are insufficient for this dataset or that the feature space requires more sophisticated interactions than axis-aligned splits.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning (last 20% of train.in)
N_TREES_OPTS = [20, 50, 100]
MAX_DEPTH_OPTS = [5, 10, 15]
MIN_LEAF_OPTS = [1, 5, 10]
MAX_FEATURES = 8  # sqrt(64)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def build_tree(X, y, depth, max_depth, min_leaf, max_feats, rng):
    """Returns list of nodes in pre-order. Each node: (feature, threshold, left, right, class)."""
    nodes = []
    def recurse(idx_samples, depth):
        n = len(idx_samples)
        if n <= min_leaf or depth >= max_depth or gini(y[idx_samples]) == 0.0:
            # leaf: majority class
            cls = np.bincount(y[idx_samples], minlength=10).argmax()
            nodes.append((-1, 0, -1, -1, int(cls)))
            return len(nodes) - 1
        # choose random feature subset
        feats = rng.choice(64, size=max_feats, replace=False)
        best_gini = 1.0
        best_feat = -1
        best_thresh = -1
        best_left = None
        best_right = None
        for f in feats:
            vals = X[idx_samples, f]
            # thresholds 0..15 (since pixel values 0..16)
            for t in range(16):
                left_mask = vals <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_idx = idx_samples[left_mask]
                right_idx = idx_samples[right_mask]
                g = (len(left_idx) * gini(y[left_idx]) + len(right_idx) * gini(y[right_idx])) / n
                if g < best_gini:
                    best_gini = g
                    best_feat = f
                    best_thresh = t
                    best_left = left_idx
                    best_right = right_idx
        if best_feat == -1:  # no good split
            cls = np.bincount(y[idx_samples], minlength=10).argmax()
            nodes.append((-1, 0, -1, -1, int(cls)))
            return len(nodes) - 1
        # create internal node (placeholder for children indices)
        node_idx = len(nodes)
        nodes.append((best_feat, best_thresh, -1, -1, 0))
        left_child = recurse(best_left, depth + 1)
        right_child = recurse(best_right, depth + 1)
        nodes[node_idx] = (best_feat, best_thresh, left_child, right_child, 0)
        return node_idx
    recurse(np.arange(len(y)), 0)
    return nodes

def train_forest(X, y, n_trees, max_depth, min_leaf, max_feats, seed):
    trees = []
    base_rng = np.random.default_rng(seed)
    for i in range(n_trees):
        tree_seed = int(base_rng.integers(2**31))
        tree_rng = np.random.default_rng(tree_seed)
        # bootstrap sample
        idx = tree_rng.choice(len(y), size=len(y), replace=True)
        Xb, yb = X[idx], y[idx]
        tree = build_tree(Xb, yb, 0, max_depth, min_leaf, max_feats, tree_rng)
        trees.append(tree)
    return trees

def evaluate_forest(trees, X, y):
    n = len(y)
    correct = 0
    for i in range(n):
        votes = np.zeros(10, dtype=int)
        xi = X[i]
        for tree in trees:
            node_idx = 0
            while True:
                feat, thresh, left, right, cls = tree[node_idx]
                if feat == -1:
                    votes[cls] += 1
                    break
                if xi[feat] <= thresh:
                    node_idx = left
                else:
                    node_idx = right
        pred = votes.argmax()  # ties broken by smallest index
        if pred == y[i]:
            correct += 1
    return correct / n

def write_weights(trees, n_features, n_classes, path):
    magic = 0x52463130  # "RF10"
    n_trees = len(trees)
    # header: magic, n_trees, n_features, n_classes
    out = struct.pack("<iiii", magic, n_trees, n_features, n_classes)
    for tree in trees:
        n_nodes = len(tree)
        out += struct.pack("<i", n_nodes)
        for feat, thresh, left, right, cls in tree:
            out += struct.pack("<iiiii", feat, thresh, left, right, cls)
    with open(path, "wb") as f:
        f.write(out)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_acc = -1.0
    best_params = None
    best_trees = None
    for n_trees in N_TREES_OPTS:
        for max_depth in MAX_DEPTH_OPTS:
            for min_leaf in MIN_LEAF_OPTS:
                trees = train_forest(X_train, y_train, n_trees, max_depth, min_leaf, MAX_FEATURES, SEED)
                acc = evaluate_forest(trees, X_tune, y_tune)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_trees, max_depth, min_leaf)
                    best_trees = trees
    print(f"Best tuning acc: {best_acc:.4f} params: {best_params}", file=sys.stderr)
    # retrain on all data with best params
    n_trees, max_depth, min_leaf = best_params
    final_trees = train_forest(X, y, n_trees, max_depth, min_leaf, MAX_FEATURES, SEED)
    write_weights(final_trees, card["input"]["bytes"], card["classes"], sys.argv[4])

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
    int32_t *nodes;      // flat array of 5 * n_nodes int32_t
    int32_t n_nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x52463130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int n_trees = hd[1], n_features = hd[2], n_classes = hd[3];
    if (n_trees <= 0 || n_features != 64 || n_classes != 10) { fprintf(stderr, "bad weights header\n"); return 3; }

    int32_t *ptr = hd + 4;
    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (int t = 0; t < n_trees; t++) {
        if ((char*)ptr - (char*)hd > nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        int32_t n_nodes = *ptr++;
        trees[t].nodes = ptr;
        trees[t].n_nodes = n_nodes;
        ptr += n_nodes * 5;
    }
    if ((char*)ptr - (char*)hd != nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    int votes[10];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_features); return 3; }
        memset(votes, 0, sizeof(votes));
        for (int t = 0; t < n_trees; t++) {
            int32_t *nodes = trees[t].nodes;
            int idx = 0;
            while (1) {
                int32_t feat = nodes[idx * 5 + 0];
                int32_t thresh = nodes[idx * 5 + 1];
                int32_t left = nodes[idx * 5 + 2];
                int32_t right = nodes[idx * 5 + 3];
                int32_t cls = nodes[idx * 5 + 4];
                if (feat == -1) {
                    votes[cls]++;
                    break;
                }
                if (xi[feat] <= thresh) idx = left;
                else idx = right;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (votes[c] > votes[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```