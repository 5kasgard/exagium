```hypothesis
Family: label-rf (Random Forest, 5 trees, max depth 10, √d features per split).
Why: Optdigits has nonlinear pixel interactions; a tiny RF captures them with ~50 node visits/item, far below linear's 5k Ir/item. Bootstrap + fixed seed ensures determinism.
Failure: dev accuracy ≤ 0.9613 or Ir/item ≥ 5000.
```
```python
"""label-rf: Random Forest for fixed-size byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, numpy only. Exports 5 depth-10 trees."""
import json, struct, sys
import numpy as np

N_TREES = 5
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ("feat", "thresh", "left", "right", "leaf", "counts")
    def __init__(self):
        self.feat = -1
        self.thresh = 0
        self.left = -1
        self.right = -1
        self.leaf = False
        self.counts = None

def gini_impurity(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)

def best_split(X, y, feat_indices, n_classes, rng):
    best_gain = -1.0
    best_feat = -1
    best_thresh = 0
    parent_counts = np.bincount(y, minlength=n_classes)
    parent_gini = gini_impurity(parent_counts)
    n_samples = len(y)
    for f in feat_indices:
        vals = X[:, f]
        unique = np.unique(vals)
        if len(unique) <= 1:
            continue
        thresholds = (unique[:-1] + unique[1:]) * 0.5
        for t in thresholds:
            left_mask = vals <= t
            right_mask = ~left_mask
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            if n_left < MIN_SAMPLES_SPLIT or n_right < MIN_SAMPLES_SPLIT:
                continue
            left_counts = np.bincount(y[left_mask], minlength=n_classes)
            right_counts = np.bincount(y[right_mask], minlength=n_classes)
            gain = parent_gini - (n_left * gini_impurity(left_counts) + n_right * gini_impurity(right_counts)) / n_samples
            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thresh = t
    return best_feat, best_thresh, best_gain

def build_tree(X, y, n_classes, depth, rng, feat_indices):
    node = Node()
    counts = np.bincount(y, minlength=n_classes)
    node.counts = counts.astype(np.uint32)
    if depth >= MAX_DEPTH or len(y) < MIN_SAMPLES_SPLIT * 2 or np.all(counts[:-1] == 0):
        node.leaf = True
        return node
    feat, thresh, gain = best_split(X, y, feat_indices, n_classes, rng)
    if gain <= 0:
        node.leaf = True
        return node
    node.feat = feat
    node.thresh = thresh
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    node.left = len(nodes)
    nodes.append(Node())
    build_tree(X[left_mask], y[left_mask], n_classes, depth + 1, rng, feat_indices)
    node.right = len(nodes)
    nodes.append(Node())
    build_tree(X[right_mask], y[right_mask], n_classes, depth + 1, rng, feat_indices)
    return node

def serialize_tree(root):
    # BFS serialization
    queue = [root]
    idx = 0
    while idx < len(queue):
        n = queue[idx]
        idx += 1
        if not n.leaf:
            queue.append(nodes[n.left])
            queue.append(nodes[n.right])
    # Write arrays
    n_nodes = len(queue)
    feats = np.zeros(n_nodes, dtype=np.int16)
    threshes = np.zeros(n_nodes, dtype=np.float32)
    lefts = np.zeros(n_nodes, dtype=np.int32)
    rights = np.zeros(n_nodes, dtype=np.int32)
    leaves = np.zeros(n_nodes, dtype=np.uint8)
    leaf_counts = np.zeros((n_nodes, n_classes), dtype=np.uint32)
    for i, n in enumerate(queue):
        feats[i] = n.feat
        threshes[i] = n.thresh
        lefts[i] = n.left
        rights[i] = n.right
        leaves[i] = 1 if n.leaf else 0
        if n.leaf:
            leaf_counts[i] = n.counts
    return feats, threshes, lefts, rights, leaves, leaf_counts

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    global n_classes
    n_classes = card["classes"]
    d = X.shape[1]
    n_features_per_split = max(1, int(np.sqrt(d)))
    rng = np.random.default_rng(SEED)
    all_trees = []
    for t in range(N_TREES):
        # Bootstrap sample
        idx = rng.integers(0, len(X), len(X))
        Xb, yb = X[idx], y[idx]
        global nodes
        nodes = [Node()]
        build_tree(Xb, yb, n_classes, 0, rng, rng.choice(d, n_features_per_split, replace=False))
        all_trees.append(serialize_tree(nodes[0]))
    # Serialize weights
    # Magic: 0x52463100 (RF1\0)
    # int32: magic, n_trees, n_classes, d, max_depth, n_features_per_split, min_samples_split
    # Per tree: int32 n_nodes, then arrays
    out = struct.pack("<7i", 0x52463100, N_TREES, n_classes, d, MAX_DEPTH, n_features_per_split, MIN_SAMPLES_SPLIT)
    for feats, threshes, lefts, rights, leaves, leaf_counts in all_trees:
        n_nodes = len(feats)
        out += struct.pack("<i", n_nodes)
        out += feats.astype("<i2").tobytes()
        out += threshes.astype("<f4").tobytes()
        out += lefts.astype("<i4").tobytes()
        out += rights.astype("<i4").tobytes()
        out += leaves.astype("<u1").tobytes()
        out += leaf_counts.astype("<u4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-rf inference: ./infer weights.bin in.rec N
 * Random Forest with fixed-format weights. Each record -> majority vote. */
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
    int16_t *feat;
    float *thresh;
    int32_t *left;
    int32_t *right;
    uint8_t *leaf;
    uint32_t *counts;
    int32_t n_nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hdr = (int32_t *)wb;
    if (nw < 28 || hdr[0] != 0x52463100) { fprintf(stderr, "bad weights\n"); return 3; }
    int n_trees = hdr[1], n_classes = hdr[2], d = hdr[3], max_depth = hdr[4], n_feat_split = hdr[5], min_samples = hdr[6];
    if (n_trees <= 0 || n_trees > 100 || n_classes <= 0 || n_classes > 256 || d <= 0 || d > 1024) { fprintf(stderr, "bad header\n"); return 3; }
    unsigned char *p = wb + 28;
    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (int t = 0; t < n_trees; t++) {
        if (p + 4 > wb + nw) { fprintf(stderr, "truncated tree count\n"); return 3; }
        int32_t n_nodes = *(int32_t *)p; p += 4;
        if (n_nodes <= 0 || n_nodes > 10000) { fprintf(stderr, "bad n_nodes\n"); return 3; }
        trees[t].n_nodes = n_nodes;
        trees[t].feat = (int16_t *)p; p += n_nodes * 2;
        trees[t].thresh = (float *)p; p += n_nodes * 4;
        trees[t].left = (int32_t *)p; p += n_nodes * 4;
        trees[t].right = (int32_t *)p; p += n_nodes * 4;
        trees[t].leaf = (uint8_t *)p; p += n_nodes;
        trees[t].counts = (uint32_t *)p; p += n_nodes * n_classes * 4;
        if (p > wb + nw) { fprintf(stderr, "truncated tree data\n"); return 3; }
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    uint32_t *votes = malloc(n_classes * sizeof(uint32_t));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(votes, 0, n_classes * sizeof(uint32_t));
        for (int t = 0; t < n_trees; t++) {
            int node = 0;
            while (!trees[t].leaf[node]) {
                int f = trees[t].feat[node];
                float th = trees[t].thresh[node];
                if (xi[f] <= th) node = trees[t].left[node];
                else node = trees[t].right[node];
            }
            uint32_t *cnts = &trees[t].counts[(long)node * n_classes];
            uint32_t best = 0;
            for (int c = 1; c < n_classes; c++) if (cnts[c] > cnts[best]) best = c;
            votes[best]++;
        }
        unsigned char best = 0;
        for (int c = 1; c < n_classes; c++) if (votes[c] > votes[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```