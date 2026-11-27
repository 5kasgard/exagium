```hypothesis
label-rf: Random Forest with bootstrap sampling and random feature subsets (sqrt(d) features per split).
Ordinal pixel features (0-16) have strong spatial interactions that trees capture naturally.
Inference uses only integer compares and array lookups, no floating point.
Expected: >0.965 accuracy at <6000 Ir/item (beats label-linear on both).
Fails if dev accuracy <= 0.9613 or Ir/item >= 8000.
```
```python
import json, struct, sys
from collections import deque
import numpy as np

SEED = 1
N_TREES = 25
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 3
MAX_FEATURES = 8  # sqrt(64)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini_impurity(counts):
    total = counts.sum()
    if total == 0: return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)

def find_best_split(X, y, n_classes, feat_indices, rng):
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    best_left_mask = None
    for f in feat_indices:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) <= 1: continue
        threshs = (uniq[:-1] + uniq[1:]) // 2
        for t in threshs:
            left_mask = vals <= t
            n_left = left_mask.sum()
            n_right = len(y) - n_left
            if n_left < MIN_SAMPLES_LEAF or n_right < MIN_SAMPLES_LEAF:
                continue
            left_counts = np.bincount(y[left_mask], minlength=n_classes)
            right_counts = np.bincount(y[~left_mask], minlength=n_classes)
            g = (n_left * gini_impurity(left_counts) + n_right * gini_impurity(right_counts)) / len(y)
            if g < best_gini:
                best_gini = g
                best_feat = f
                best_thresh = int(t)
                best_left_mask = left_mask
    return best_feat, best_thresh, best_left_mask

def build_tree_bfs(X, y, n_classes, rng):
    max_nodes = (1 << (MAX_DEPTH + 1)) - 1
    nodes = []
    queue = deque()
    root_idx = 0
    queue.append((0, X, y, 0))
    while queue:
        idx, X_node, y_node, depth = queue.popleft()
        while len(nodes) <= idx:
            nodes.append(None)
        counts = np.bincount(y_node, minlength=n_classes)
        if depth >= MAX_DEPTH or len(y_node) < 2 * MIN_SAMPLES_LEAF or np.max(counts) == len(y_node):
            nodes[idx] = (1, 0, 0, 0, 0, counts.astype(np.uint16))
            continue
        n_feats = X_node.shape[1]
        feat_indices = rng.choice(n_feats, size=min(MAX_FEATURES, n_feats), replace=False)
        feat, thresh, left_mask = find_best_split(X_node, y_node, n_classes, feat_indices, rng)
        if feat == -1:
            nodes[idx] = (1, 0, 0, 0, 0, counts.astype(np.uint16))
            continue
        left_idx = len(nodes)
        right_idx = left_idx + 1
        nodes[idx] = (0, np.uint8(feat), np.uint8(thresh), np.uint16(left_idx), np.uint16(right_idx), None)
        queue.append((left_idx, X_node[left_mask], y_node[left_mask], depth + 1))
        queue.append((right_idx, X_node[~left_mask], y_node[~left_mask], depth + 1))
    return nodes

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    rng = np.random.default_rng(SEED)
    
    all_trees = []
    for _ in range(N_TREES):
        idx = rng.integers(0, n, n)
        X_boot, y_boot = X[idx], y[idx]
        tree = build_tree_bfs(X_boot, y_boot, C, rng)
        all_trees.append(tree)
    
    out = struct.pack("<Iiiii", 0x52465354, N_TREES, MAX_DEPTH, C, d)
    for tree in all_trees:
        out += struct.pack("<I", len(tree))
        for node in tree:
            is_leaf, feat, thresh, left, right, counts = node
            out += struct.pack("<BBBHH", is_leaf, feat, thresh, left, right)
            if is_leaf:
                out += counts.tobytes()
            else:
                out += b"\x00" * (2 * C)
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
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = w[0] | w[1]<<8 | w[2]<<16 | w[3]<<24;
    if (magic != 0x52465354) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_trees = *(int32_t*)(w+4);
    int32_t max_depth = *(int32_t*)(w+8);
    int32_t n_classes = *(int32_t*)(w+12);
    int32_t n_features = *(int32_t*)(w+16);
    if (n_classes < 1 || n_classes > 256 || n_trees < 1 || n_trees > 1000 || n_features != 64) {
        fprintf(stderr, "bad header\n"); return 3; }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *p = w + 20;
    unsigned char *end = w + nw;
    uint32_t *tree_base = malloc(n_trees * sizeof(uint32_t));
    uint32_t *tree_nodes = malloc(n_trees * sizeof(uint32_t));
    for (int t = 0; t < n_trees; t++) {
        if (p + 4 > end) { fprintf(stderr, "truncated trees\n"); return 3; }
        tree_nodes[t] = *(uint32_t*)p; p += 4;
        tree_base[t] = p - w;
        uint32_t node_size = 7 + 2 * n_classes;
        if (p + (long)tree_nodes[t] * node_size > end) { fprintf(stderr, "truncated nodes\n"); return 3; }
        p += tree_nodes[t] * node_size;
    }
    if (p != end) { fprintf(stderr, "extra bytes\n"); return 3; }
    uint32_t node_size = 7 + 2 * n_classes;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_features) { fprintf(stderr, "item %ld bad len\n", i); return 3; }
        uint32_t votes[256] = {0};
        for (int t = 0; t < n_trees; t++) {
            unsigned char *node = w + tree_base[t];
            uint32_t idx = 0;
            while (1) {
                if (idx >= tree_nodes[t]) { fprintf(stderr, "bad idx\n"); return 3; }
                unsigned char *nd = node + idx * node_size;
                uint8_t is_leaf = nd[0];
                if (is_leaf) {
                    uint16_t *counts = (uint16_t*)(nd + 7);
                    for (int c = 0; c < n_classes; c++) votes[c] += counts[c];
                    break;
                } else {
                    uint8_t feat = nd[1];
                    uint8_t thresh = nd[2];
                    uint16_t left = nd[3] | (nd[4]<<8);
                    uint16_t right = nd[5] | (nd[6]<<8);
                    idx = (xi[feat] <= thresh) ? left : right;
                }
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < n_classes; c++) if (votes[c] > votes[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```