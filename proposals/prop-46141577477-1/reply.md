```hypothesis
Family: label-rf (Random Forest). Random forests capture non-linear pixel interactions crucial for digit recognition, typically exceeding linear models on optdigits (target >97% dev accuracy). Inference stays fast: ~100 trees * depth 10 = ~1000 node visits per item. If dev accuracy fails to beat label-linear (0.9613), the forest is underfitting (too few trees, too shallow, or excessive regularization).
```
```python
"""Random Forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x52463100, n_features, n_classes, n_trees; then per tree:
  n_nodes (int32), then for each node: feature (uint8), threshold (float32),
  left (int32), right (int32), class (uint8), is_leaf (uint8)."""
import json, math, struct, sys

import numpy as np

SEED = 1
N_TREES = 150
MAX_DEPTH = 18
MIN_SAMPLES_LEAF = 3
MAX_FEATURES = 8  # sqrt(64)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(y):
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, rng, max_features):
    n_samples, n_features = X.shape
    if n_samples < 2: return None, None
    feat_idx = rng.choice(n_features, max_features, replace=False)
    best_gini = 1.0
    best_feat = -1
    best_thresh = 0.0
    for f in feat_idx:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) == 1: continue
        thresholds = (uniq[:-1] + uniq[1:]) * 0.5
        for t in thresholds:
            left = y[vals <= t]
            right = y[vals > t]
            if len(left) == 0 or len(right) == 0: continue
            g = (len(left) * gini(left) + len(right) * gini(right)) / n_samples
            if g < best_gini:
                best_gini = g
                best_feat = f
                best_thresh = t
    return best_feat, best_thresh

def build_tree(X, y, rng, depth=0):
    n_samples = len(y)
    node = {}
    if depth >= MAX_DEPTH or n_samples <= MIN_SAMPLES_LEAF or gini(y) == 0.0:
        node['is_leaf'] = True
        node['class'] = int(np.bincount(y).argmax())
        return [node]
    feat, thresh = best_split(X, y, rng, MAX_FEATURES)
    if feat is None:
        node['is_leaf'] = True
        node['class'] = int(np.bincount(y).argmax())
        return [node]
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    node['is_leaf'] = False
    node['feature'] = int(feat)
    node['threshold'] = float(thresh)
    left_nodes = build_tree(X[left_mask], y[left_mask], rng, depth+1)
    right_nodes = build_tree(X[right_mask], y[right_mask], rng, depth+1)
    # left child index will be 1 (next node), right child after left subtree
    node['left'] = 1
    node['right'] = 1 + len(left_nodes)
    return [node] + left_nodes + right_nodes

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_features = card["input"]["bytes"]
    n_classes = card["classes"]
    rng = np.random.default_rng(SEED)

    # Tuning split: last 20%
    n = len(X)
    split = int(0.8 * n)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_idx, tune_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Simple hyperparameter search (fixed grid, could be extended)
    best_params = (N_TREES, MAX_DEPTH, MIN_SAMPLES_LEAF, MAX_FEATURES)
    best_acc = -1.0
    # We'll just use the fixed params; tuning omitted for brevity and determinism.
    # If tuning were desired, uncomment and iterate over a small grid.

    # Train final forest on all data
    trees = []
    for _ in range(N_TREES):
        boot_idx = rng.choice(n, n, replace=True)
        X_boot, y_boot = X[boot_idx], y[boot_idx]
        tree_nodes = build_tree(X_boot, y_boot, rng)
        # Convert to flat arrays
        n_nodes = len(tree_nodes)
        features = np.zeros(n_nodes, dtype=np.uint8)
        thresholds = np.zeros(n_nodes, dtype=np.float32)
        lefts = np.zeros(n_nodes, dtype=np.int32)
        rights = np.zeros(n_nodes, dtype=np.int32)
        classes = np.zeros(n_nodes, dtype=np.uint8)
        is_leaf = np.zeros(n_nodes, dtype=np.uint8)
        for i, nd in enumerate(tree_nodes):
            features[i] = nd.get('feature', 0)
            thresholds[i] = nd.get('threshold', 0.0)
            lefts[i] = nd.get('left', -1)
            rights[i] = nd.get('right', -1)
            classes[i] = nd.get('class', 0)
            is_leaf[i] = 1 if nd['is_leaf'] else 0
        trees.append((n_nodes, features, thresholds, lefts, rights, classes, is_leaf))

    # Serialize
    magic = 0x52463100  # 'RF1\0'
    out = struct.pack("<4I", magic, n_features, n_classes, N_TREES)
    for (n_nodes, features, thresholds, lefts, rights, classes, is_leaf) in trees:
        out += struct.pack("<i", n_nodes)
        out += features.tobytes()
        out += thresholds.tobytes()
        out += lefts.tobytes()
        out += rights.tobytes()
        out += classes.tobytes()
        out += is_leaf.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Random Forest inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: 1 byte class.
 * Weights format: magic(0x52463100), d, C, n_trees (all int32).
 * Per tree: n_nodes (int32), then arrays of length n_nodes:
 *   feature[uint8], threshold[float32], left[int32], right[int32], class[uint8], is_leaf[uint8]. */
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
    int n_nodes;
    uint8_t *feature;
    float *threshold;
    int32_t *left;
    int32_t *right;
    uint8_t *class;
    uint8_t *is_leaf;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    uint32_t *hdr = (uint32_t *)wb;
    if (nw < 16 || hdr[0] != 0x52463100) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], C = hdr[2], n_trees = hdr[3];
    if (d != 64 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }
    unsigned char *p = wb + 16;
    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (int t = 0; t < n_trees; t++) {
        if (p + 4 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        int n_nodes = *(int32_t *)p; p += 4;
        size_t need = n_nodes * (1 + 4 + 4 + 4 + 1 + 1);
        if (p + need > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t].n_nodes = n_nodes;
        trees[t].feature = p; p += n_nodes;
        trees[t].threshold = (float *)p; p += n_nodes * 4;
        trees[t].left = (int32_t *)p; p += n_nodes * 4;
        trees[t].right = (int32_t *)p; p += n_nodes * 4;
        trees[t].class = p; p += n_nodes;
        trees[t].is_leaf = p; p += n_nodes;
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *vote = malloc(C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(vote, 0, C);
        for (int t = 0; t < n_trees; t++) {
            Tree *tr = &trees[t];
            int node = 0;
            while (1) {
                if (tr->is_leaf[node]) {
                    vote[tr->class[node]]++;
                    break;
                }
                uint8_t feat = tr->feature[node];
                float thresh = tr->threshold[node];
                if (xi[feat] <= thresh) node = tr->left[node];
                else node = tr->right[node];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (vote[c] > vote[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```