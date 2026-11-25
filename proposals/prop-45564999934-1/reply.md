```hypothesis
Family: label-rf (Random Forest with 20 trees, max depth 10, sqrt feature sampling).
Why: Optdigits pixels have local correlations that linear models miss; shallow trees capture interactions. 20 trees × ≤10 node visits = ~200 comparisons/item, far fewer than linear's 640 multiply-adds, so both higher accuracy and lower CPU cost are expected.
Failure signal: If dev accuracy ≤ 0.9613 or instructions/item ≥ 5000, the forest is too weak or too deep for this data.
```

```python
"""Random Forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 42), single thread, stdlib+numpy only.
Exports: magic 0x52464C31, n_trees, n_classes, n_features; then per tree: n_nodes, nodes (feature u8, threshold u8, is_leaf u8, leaf_class u8, left i32, right i32)."""
import json, struct, sys
import numpy as np

SEED = 42
N_TREES = 20
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.dot(p, p)


def best_split(X, y, indices, features, rng, n_classes):
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    n = len(indices)
    if n < MIN_SAMPLES_SPLIT:
        return -1, -1, 1.0
    parent_counts = np.bincount(y[indices], minlength=n_classes)
    parent_gini = gini(parent_counts)
    for feat in features:
        vals = X[indices, feat]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) // 2  # integer midpoints
        for thresh in thresholds:
            left_mask = vals <= thresh
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue
            left_idx = indices[left_mask]
            right_idx = indices[right_mask]
            left_counts = np.bincount(y[left_idx], minlength=n_classes)
            right_counts = np.bincount(y[right_idx], minlength=n_classes)
            w_left = len(left_idx) / n
            w_right = len(right_idx) / n
            weighted_gini = w_left * gini(left_counts) + w_right * gini(right_counts)
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh, best_gini


def build_tree(X, y, indices, depth, rng, n_classes, n_features):
    nodes = []
    def recurse(idx_list, depth):
        node_idx = len(nodes)
        nodes.append(None)  # placeholder
        counts = np.bincount(y[idx_list], minlength=n_classes)
        leaf_class = int(np.argmax(counts))
        if depth >= MAX_DEPTH or len(idx_list) < MIN_SAMPLES_SPLIT or len(np.unique(y[idx_list])) == 1:
            nodes[node_idx] = (255, 255, 1, leaf_class, -1, -1)
            return node_idx
        # random feature subset
        all_feats = np.arange(n_features)
        rng.shuffle(all_feats)
        subset = all_feats[:int(np.sqrt(n_features)) + 1]
        feat, thresh, split_gini = best_split(X, y, idx_list, subset, rng, n_classes)
        if feat == -1 or split_gini >= gini(counts):
            nodes[node_idx] = (255, 255, 1, leaf_class, -1, -1)
            return node_idx
        left_mask = X[idx_list, feat] <= thresh
        right_mask = ~left_mask
        left_idx = idx_list[left_mask]
        right_idx = idx_list[right_mask]
        left_child = recurse(left_idx, depth + 1)
        right_child = recurse(right_idx, depth + 1)
        nodes[node_idx] = (feat, thresh, 0, 0, left_child, right_child)
        return node_idx
    recurse(indices, 0)
    return nodes


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    trees = []
    for t in range(N_TREES):
        boot_idx = rng.choice(n, size=n, replace=True)
        nodes = build_tree(X, y, boot_idx, 0, rng, C, d)
        trees.append(nodes)

    magic = 0x52464C31  # "RFL1"
    out = struct.pack("<Iiii", magic, N_TREES, C, d)
    for nodes in trees:
        out += struct.pack("<i", len(nodes))
        for (feat, thresh, is_leaf, leaf_class, left, right) in nodes:
            out += struct.pack("<BBBBii", feat, thresh, is_leaf, leaf_class, left, right)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Random Forest inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index, majority vote, ties -> smallest). */
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
    uint8_t feature;
    uint8_t threshold;
    uint8_t is_leaf;
    uint8_t leaf_class;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x52464C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_trees = *(int32_t*)(wb+4);
    int32_t n_classes = *(int32_t*)(wb+8);
    int32_t n_features = *(int32_t*)(wb+12);
    if (n_trees <= 0 || n_trees > 1000 || n_classes <= 0 || n_classes > 256 || n_features <= 0 || n_features > 256) {
        fprintf(stderr, "bad header values\n"); return 3;
    }
    unsigned char *ptr = wb + 16;
    unsigned char *end = wb + nw;
    Node **trees = malloc(n_trees * sizeof(Node*));
    int32_t *tree_sizes = malloc(n_trees * sizeof(int32_t));
    for (int t = 0; t < n_trees; t++) {
        if (ptr + 4 > end) { fprintf(stderr, "truncated tree size\n"); return 3; }
        int32_t n_nodes = *(int32_t*)ptr; ptr += 4;
        if (n_nodes <= 0) { fprintf(stderr, "empty tree\n"); return 3; }
        tree_sizes[t] = n_nodes;
        trees[t] = (Node*)ptr;
        ptr += n_nodes * sizeof(Node);
        if (ptr > end) { fprintf(stderr, "truncated tree nodes\n"); return 3; }
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *votes = malloc(n_classes);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_features); return 3; }
        memset(votes, 0, n_classes);
        for (int t = 0; t < n_trees; t++) {
            Node *nodes = trees[t];
            int node_idx = 0;
            while (1) {
                Node *nd = &nodes[node_idx];
                if (nd->is_leaf) {
                    votes[nd->leaf_class]++;
                    break;
                }
                if (xi[nd->feature] <= nd->threshold) node_idx = nd->left;
                else node_idx = nd->right;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < n_classes; c++) if (votes[c] > votes[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```