```hypothesis
Family: label-tree (single CART decision tree). The Adult data has strong feature interactions (e.g., education × occupation × hours) that a linear model misses. A tuned-depth tree captures these non-linearities with far fewer operations than GBDT/MLP. We tune max_depth, min_samples_leaf, min_samples_split on the last 20% of train.in, then refit on all data. Failure: dev accuracy ≤ 0.8524 (label-linear) or inference cost ≥ label-linear.
```
```python
"""label-tree: single CART decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format: magic(0x54524545), d, n_nodes, C; then per node:
  int8 feature, uint8 threshold, int32 left, int32 right, uint8 leaf_class (left==-1).
Split: x[feature] <= threshold -> left else right. Leaf stores majority class.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Tuning grid
MAX_DEPTH_GRID = [5, 6, 7, 8, 9, 10]
MIN_LEAF_GRID = [1, 5, 10, 20]
MIN_SPLIT_GRID = [2, 5, 10]

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
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, idx, min_split, min_leaf):
    n = len(idx)
    if n < min_split:
        return None, None, None
    best_gini = 1.0
    best_feat = best_thr = None
    for f in range(X.shape[1]):
        vals = X[idx, f]
        # thresholds are midpoints between sorted unique values
        uniq = np.unique(vals)
        if len(uniq) == 1:
            continue
        thrs = (uniq[:-1] + uniq[1:]) // 2  # integer thresholds
        for thr in thrs:
            left = idx[vals <= thr]
            right = idx[vals > thr]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            g = (len(left) * gini(y[left]) + len(right) * gini(y[right])) / n
            if g < best_gini:
                best_gini, best_feat, best_thr = g, f, thr
    return best_feat, best_thr, best_gini

def build_tree(X, y, max_depth, min_leaf, min_split):
    n = X.shape[0]
    nodes = []  # each node: (feat, thr, left, right, leaf_class)
    # queue of (node_index, sample_indices, depth)
    queue = [(0, np.arange(n), 0)]
    while queue:
        node_idx, idx, depth = queue.pop(0)
        # compute leaf class (majority)
        if len(idx) == 0:
            leaf_cls = 0
        else:
            vals, cnt = np.unique(y[idx], return_counts=True)
            leaf_cls = int(vals[np.argmax(cnt)])
        # stopping criteria
        if depth >= max_depth or len(idx) < min_split or gini(y[idx]) == 0.0:
            nodes.append((-1, 0, -1, -1, leaf_cls))
            continue
        feat, thr, _ = best_split(X, y, idx, min_split, min_leaf)
        if feat is None:
            nodes.append((-1, 0, -1, -1, leaf_cls))
            continue
        left_idx = idx[X[idx, feat] <= thr]
        right_idx = idx[X[idx, feat] > thr]
        # create internal node, children will be appended later
        left_node = len(nodes) + 1
        right_node = len(nodes) + 2
        nodes.append((feat, thr, left_node, right_node, 0))
        queue.append((left_node, left_idx, depth + 1))
        queue.append((right_node, right_idx, depth + 1))
    # convert to flat arrays
    feat_arr = np.array([n[0] for n in nodes], dtype=np.int8)
    thr_arr = np.array([n[1] for n in nodes], dtype=np.uint8)
    left_arr = np.array([n[2] for n in nodes], dtype=np.int32)
    right_arr = np.array([n[3] for n in nodes], dtype=np.int32)
    leaf_arr = np.array([n[4] for n in nodes], dtype=np.uint8)
    return feat_arr, thr_arr, left_arr, right_arr, leaf_arr

def evaluate(X, y, feat_arr, thr_arr, left_arr, right_arr, leaf_arr):
    n = X.shape[0]
    pred = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        node = 0
        while left_arr[node] != -1:
            f = feat_arr[node]
            t = thr_arr[node]
            node = left_arr[node] if X[i, f] <= t else right_arr[node]
        pred[i] = leaf_arr[node]
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = X.shape[0]
    # tuning split: last 20%
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]
    best_acc = -1.0
    best_params = None
    for md in MAX_DEPTH_GRID:
        for ml in MIN_LEAF_GRID:
            for ms in MIN_SPLIT_GRID:
                feat, thr, left, right, leaf = build_tree(X_tr, y_tr, md, ml, ms)
                acc = evaluate(X_va, y_va, feat, thr, left, right, leaf)
                if acc > best_acc:
                    best_acc, best_params = acc, (md, ml, ms)
    # retrain on all data with best params
    md, ml, ms = best_params
    feat, thr, left, right, leaf = build_tree(X, y, md, ml, ms)
    # write weights.bin
    magic = 0x54524545  # "EERT" little-endian
    d = X.shape[1]
    n_nodes = len(feat)
    C = card["classes"]
    out = struct.pack("<4i", magic, d, n_nodes, C)
    out += feat.tobytes() + thr.tobytes() + left.tobytes() + right.tobytes() + leaf.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class index.
 * Tree stored as flat arrays: feature[i], threshold[i], left[i], right[i], leaf[i].
 * Traverse from root (0): while left[i]!=-1, go left if x[feat]<=thr else right. */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x54524545) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], n_nodes = hd[2], C = hd[3];
    if (C != 2 || n_nodes <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16L + (long)n_nodes * (1 + 1 + 4 + 4 + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const int8_t *feat = (const int8_t *)(hd + 4);
    const uint8_t *thr = (const uint8_t *)(feat + n_nodes);
    const int32_t *left = (const int32_t *)(thr + n_nodes);
    const int32_t *right = (const int32_t *)(left + n_nodes);
    const uint8_t *leaf = (const uint8_t *)(right + n_nodes);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (left[node] != -1) {
            int f = feat[node];
            uint8_t t = thr[node];
            node = (xi[f] <= t) ? left[node] : right[node];
        }
        unsigned char cls = leaf[node];
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```