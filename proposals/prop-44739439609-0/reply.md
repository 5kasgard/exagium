```hypothesis
Propose "label-tree": a single deterministic decision tree (max depth 6) trained with Gini impurity on the raw 14-byte features. It captures non-linear interactions missed by linear models while inference requires only a few branch predictions per record. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds label-linear (719 Ir/item), the family fails.
```

```python
"""label-tree: single decision tree for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seed), single thread, stdlib + numpy only.
Exports: magic 0x54524545, d, max_depth, n_nodes; then n_nodes of:
  uint8 feature (0..d-1, 255=leaf), uint8 threshold, int32 left, int32 right, uint8 leaf_class, uint8 pad[3]."""
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH_CANDIDATES = [1, 2, 3, 4, 5, 6]
TUNE_FRACTION = 0.2

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

def best_split(X, y, feature_idx, parent_gini):
    """Return (threshold, weighted_gini, left_mask) for the best split on one feature."""
    vals = X[:, feature_idx]
    best_thr = -1
    best_gini = parent_gini
    best_mask = None
    # thresholds 0..254 (split: <= thr vs > thr)
    for thr in range(255):
        mask = vals <= thr
        if mask.all() or (~mask).all():
            continue
        n = len(y)
        nL, nR = mask.sum(), (~mask).sum()
        gL = gini(y[mask])
        gR = gini(y[~mask])
        wg = (nL * gL + nR * gR) / n
        if wg < best_gini - 1e-12:
            best_gini = wg
            best_thr = thr
            best_mask = mask.copy()
    return best_thr, best_gini, best_mask

def build_tree(X, y, max_depth, depth=0, node_list=None, node_idx=0):
    """Recursively build tree, appending nodes to node_list in BFS order.
    Returns the index of the created node."""
    if node_list is None:
        node_list = []
    # stopping criteria
    if depth >= max_depth or len(np.unique(y)) == 1:
        leaf_class = int(np.bincount(y).argmax())
        node_list.append((255, 0, -1, -1, leaf_class))
        return len(node_list) - 1

    parent_gini = gini(y)
    best_feat = -1
    best_thr = -1
    best_gini = parent_gini
    best_mask = None

    for f in range(X.shape[1]):
        thr, wg, mask = best_split(X, y, f, parent_gini)
        if wg < best_gini - 1e-12:
            best_gini = wg
            best_feat = f
            best_thr = thr
            best_mask = mask

    if best_feat == -1:
        leaf_class = int(np.bincount(y).argmax())
        node_list.append((255, 0, -1, -1, leaf_class))
        return len(node_list) - 1

    # create internal node placeholder, will fill children after recursion
    my_idx = len(node_list)
    node_list.append((best_feat, best_thr, -1, -1, 0))  # children to be filled
    left_idx = build_tree(X[best_mask], y[best_mask], max_depth, depth+1, node_list, my_idx)
    right_idx = build_tree(X[~best_mask], y[~best_mask], max_depth, depth+1, node_list, my_idx)
    # update children indices (node_list is mutable)
    feat, thr, _, _, _ = node_list[my_idx]
    node_list[my_idx] = (feat, thr, left_idx, right_idx, 0)
    return my_idx

def tree_to_bytes(node_list, d, max_depth):
    """Serialize tree to bytes."""
    n_nodes = len(node_list)
    header = struct.pack("<4I", 0x54524545, d, max_depth, n_nodes)
    body = bytearray()
    for (feat, thr, left, right, leaf_class) in node_list:
        body += struct.pack("<BBiiiB3x", feat, thr, left, right, leaf_class)
    return header + bytes(body)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(y)

    # split tuning set (last 20% in file order)
    tune_start = int(n * (1 - TUNE_FRACTION))
    X_train, X_tune = X[:tune_start], X[tune_start:]
    y_train, y_tune = y[:tune_start], y[tune_start:]

    best_depth = MAX_DEPTH_CANDIDATES[0]
    best_acc = -1.0
    rng = np.random.default_rng(SEED)

    for depth in MAX_DEPTH_CANDIDATES:
        nodes = []
        build_tree(X_train, y_train, depth, node_list=nodes)
        # evaluate on tune
        correct = 0
        for xi, yi in zip(X_tune, y_tune):
            idx = 0
            while True:
                feat, thr, left, right, leaf = nodes[idx]
                if feat == 255:
                    pred = leaf
                    break
                if xi[feat] <= thr:
                    idx = left
                else:
                    idx = right
            if pred == yi:
                correct += 1
        acc = correct / len(y_tune)
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # retrain on full data with best_depth
    nodes = []
    build_tree(X, y, best_depth, node_list=nodes)
    out_bytes = tree_to_bytes(nodes, d, best_depth)
    open(sys.argv[4], "wb").write(out_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class index. */
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

typedef struct { uint8_t feat, thr; int32_t left, right; uint8_t leaf; } Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], max_depth = hd[2], n_nodes = hd[3];
    size_t expected = 16 + (size_t)n_nodes * 12;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int idx = 0;
        while (1) {
            Node nd = nodes[idx];
            if (nd.feat == 255) { put_rec(stdout, &nd.leaf, 1); break; }
            if (xi[nd.feat] <= nd.thr) idx = nd.left;
            else idx = nd.right;
        }
    }
    return 0;
}
```