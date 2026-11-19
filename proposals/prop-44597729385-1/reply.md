```hypothesis
A small random forest (10-30 trees, depth 5-7) exploits pixel interactions that linear models miss.
It should surpass label-linear's 96.1% dev accuracy while keeping inference fast (only integer comparisons).
If it fails to beat linear, the task may need deeper trees or more trees, increasing cost.
```

```python
import json, struct, sys
import numpy as np

SEED = 1

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p*p)

def best_split(X, y, feature_indices, rng):
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    for feat in feature_indices:
        vals = X[:, feat]
        unique_vals = np.unique(vals)
        for thresh in unique_vals:
            left = vals <= thresh
            right = ~left
            if left.sum() == 0 or right.sum() == 0:
                continue
            g = (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / len(y)
            if g < best_gini:
                best_gini = g
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh, best_gini

def build_tree(X, y, max_depth, min_samples_split, rng):
    n_samples, n_features = X.shape
    nodes = [None]  # placeholder for root at index 0
    queue = [(0, np.arange(n_samples), 0)]
    while queue:
        idx, sample_idx, depth = queue.pop(0)
        y_sub = y[sample_idx]
        if depth == max_depth or len(sample_idx) < min_samples_split or gini(y_sub) == 0:
            leaf_class = int(np.bincount(y_sub).argmax())
            nodes[idx] = (1, 0, 0, leaf_class, -1, -1)
            continue
        feat_indices = rng.choice(n_features, size=min(8, n_features), replace=False)
        feat, thresh, _ = best_split(X[sample_idx], y_sub, feat_indices, rng)
        if feat == -1:
            leaf_class = int(np.bincount(y_sub).argmax())
            nodes[idx] = (1, 0, 0, leaf_class, -1, -1)
            continue
        left_mask = X[sample_idx, feat] <= thresh
        left_idx = sample_idx[left_mask]
        right_idx = sample_idx[~left_mask]
        left_node_idx = len(nodes)
        nodes.append(None)
        right_node_idx = len(nodes)
        nodes.append(None)
        nodes[idx] = (0, int(feat), int(thresh), 0, left_node_idx, right_node_idx)
        queue.append((left_node_idx, left_idx, depth+1))
        queue.append((right_node_idx, right_idx, depth+1))
    return nodes

def train_forest(X, y, n_trees, max_depth, min_samples_split, rng):
    forest = []
    for t in range(n_trees):
        tree_rng = np.random.default_rng(SEED + t + 1000)
        n = X.shape[0]
        boot_idx = tree_rng.choice(n, size=n, replace=True)
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]
        nodes = build_tree(X_boot, y_boot, max_depth, min_samples_split, tree_rng)
        forest.append(nodes)
    return forest

def evaluate_forest(forest, X, y):
    n = X.shape[0]
    votes = np.zeros((n, 10), dtype=np.int32)
    for nodes in forest:
        for i in range(n):
            idx = 0
            while True:
                is_leaf, feat, thresh, leaf_class, left, right = nodes[idx]
                if is_leaf:
                    votes[i, leaf_class] += 1
                    break
                if X[i, feat] <= thresh:
                    idx = left
                else:
                    idx = right
    preds = votes.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split_idx = int(n * 0.8)
    X_train, X_tune = X[:split_idx], X[split_idx:]
    y_train, y_tune = y[:split_idx], y[split_idx:]
    
    best_acc = -1.0
    best_params = (10, 5, 2)
    for n_trees in [10, 20, 30]:
        for max_depth in [5, 6, 7]:
            for min_samples_split in [2, 5]:
                rng = np.random.default_rng(SEED)
                forest = train_forest(X_train, y_train, n_trees, max_depth, min_samples_split, rng)
                acc = evaluate_forest(forest, X_tune, y_tune)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_trees, max_depth, min_samples_split)
    print(f"Best params: {best_params}, tune acc: {best_acc:.4f}", file=sys.stderr)
    
    rng = np.random.default_rng(SEED)
    forest = train_forest(X, y, *best_params, rng)
    
    n_trees, max_depth, min_samples_split = best_params
    n_classes = card["classes"]
    n_features = card["input"]["bytes"]
    out = struct.pack("<5i", 0x52464C31, n_trees, max_depth, n_classes, n_features)
    for nodes in forest:
        out += struct.pack("<i", len(nodes))
        for node in nodes:
            is_leaf, feat, thresh, leaf_class, left, right = node
            out += struct.pack("<BBBBii", is_leaf, feat, thresh, leaf_class, left, right)
    open(sys.argv[4], "wb").write(out)

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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

typedef struct {
    uint8_t is_leaf;
    uint8_t feat;
    uint8_t thresh;
    uint8_t leaf_class;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t*)wb;
    if (hd[0] != 0x52464C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int n_trees = hd[1];
    int max_depth = hd[2];
    int n_classes = hd[3];
    int n_features = hd[4];
    if (n_classes < 1 || n_classes > 256 || n_features != 64) { fprintf(stderr, "bad weights params\n"); return 3; }
    Node **trees = malloc(n_trees * sizeof(Node*));
    int *node_counts = malloc(n_trees * sizeof(int));
    unsigned char *p = wb + 20;
    unsigned char *end = wb + nw;
    for (int t = 0; t < n_trees; t++) {
        if (p + 4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        int nc = *(int32_t*)p; p += 4;
        if (nc <= 0) { fprintf(stderr, "bad node count\n"); return 3; }
        node_counts[t] = nc;
        trees[t] = malloc(nc * sizeof(Node));
        if (p + (long)nc * 12 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        for (int i = 0; i < nc; i++) {
            Node *n = &trees[t][i];
            n->is_leaf = p[0];
            n->feat = p[1];
            n->thresh = p[2];
            n->leaf_class = p[3];
            n->left = *(int32_t*)(p+4);
            n->right = *(int32_t*)(p+8);
            p += 12;
        }
    }
    if (p != end) { fprintf(stderr, "extra bytes in weights\n"); return 3; }
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_features); return 3; }
        int votes[10] = {0};
        for (int t = 0; t < n_trees; t++) {
            int idx = 0;
            while (1) {
                Node *n = &trees[t][idx];
                if (n->is_leaf) {
                    votes[n->leaf_class]++;
                    break;
                }
                if (xi[n->feat] <= n->thresh) idx = n->left;
                else idx = n->right;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < n_classes; c++) if (votes[c] > votes[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```