```hypothesis
Family: label-rf (small deterministic Random Forest).  
Why: Tabular data with mixed ordinal/categorical features often benefits from non-linear interactions that a shallow forest captures better than a linear model, while remaining extremely fast at inference (a few branch predictions per tree).  
Risk signal: If dev accuracy ≤ 0.8524 or inference cost ≥ label-linear, the family does not justify its existence.
```
```python
"""label-rf: small deterministic Random Forest for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Trains an ensemble of shallow decision trees (max_depth=4, n_trees=5 by default).
Hyper-parameters (n_trees, max_depth) are chosen on the last 20% of train.in (tuning split),
then the final model is trained on the full train.in.  Exports a compact binary format."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyper-parameter grid searched on the tuning split (last 20% of train.in)
GRID = {"n_trees": [3, 5, 7], "max_depth": [3, 4, 5]}
MAGIC = 0x464F5246  # 'FORF' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class Node:
    __slots__ = ("feat", "thresh", "left", "right", "value")

    def __init__(self, feat=0, thresh=0, left=-1, right=-1, value=0):
        self.feat = feat
        self.thresh = thresh
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.left == -1 and self.right == -1

    def pack(self):
        # feat:uint8, thresh:uint8, left:int16, right:int16, value:int8, pad:uint8
        return struct.pack("<BBhhbB", self.feat, self.thresh, self.left, self.right, self.value, 0)


def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)


def best_split(X, y, rng, max_features=None):
    n, d = X.shape
    if n < 2:
        return None, None, 0.0
    parent_gini = gini(y)
    best_gain = 0.0
    best_feat = best_thresh = None
    features = rng.permutation(d)[:max_features] if max_features else range(d)
    for f in features:
        vals = np.unique(X[:, f])
        if len(vals) == 1:
            continue
        # try midpoints between consecutive unique values
        for t in (vals[:-1] + vals[1:]) // 2:
            mask = X[:, f] <= t
            if mask.all() or (~mask).all():
                continue
            n_left, n_right = mask.sum(), (~mask).sum()
            gain = parent_gini - (n_left * gini(y[mask]) + n_right * gini(y[~mask])) / n
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, f, t
    return best_feat, best_thresh, best_gain


def build_tree(X, y, rng, depth, max_depth, max_features):
    node = Node()
    if depth == max_depth or len(np.unique(y)) == 1:
        node.value = int(np.bincount(y).argmax())
        return [node]
    feat, thresh, gain = best_split(X, y, rng, max_features)
    if gain == 0.0:
        node.value = int(np.bincount(y).argmax())
        return [node]
    mask = X[:, feat] <= thresh
    node.feat, node.thresh = int(feat), int(thresh)
    left_nodes = build_tree(X[mask], y[mask], rng, depth + 1, max_depth, max_features)
    right_nodes = build_tree(X[~mask], y[~mask], rng, depth + 1, max_depth, max_features)
    node.left = 1
    node.right = 1 + len(left_nodes)
    return [node] + left_nodes + right_nodes


def predict_tree(nodes, x):
    idx = 0
    while True:
        n = nodes[idx]
        if n.is_leaf():
            return n.value
        idx = n.left if x[n.feat] <= n.thresh else n.right


def evaluate_forest(forest, X, y):
    preds = np.zeros(len(y), dtype=np.int64)
    for nodes in forest:
        for i, xi in enumerate(X):
            preds[i] += predict_tree(nodes, xi)
    preds = (preds > len(forest) // 2).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    # tuning split = last 20%
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_score = -1.0
    best_params = None
    for n_trees in GRID["n_trees"]:
        for max_depth in GRID["max_depth"]:
            # train on 80%
            forest = []
            for t in range(n_trees):
                # bootstrap sample
                idx = rng.integers(0, len(X_train), len(X_train))
                nodes = build_tree(X_train[idx], y_train[idx], rng, 0, max_depth, max_features=8)
                forest.append(nodes)
            score = evaluate_forest(forest, X_tune, y_tune)
            if score > best_score:
                best_score, best_params = score, (n_trees, max_depth)

    # retrain on full data with best params
    n_trees, max_depth = best_params
    rng = np.random.default_rng(SEED)  # reset for reproducibility
    forest = []
    for t in range(n_trees):
        idx = rng.integers(0, n, n)
        nodes = build_tree(X[idx], y[idx], rng, 0, max_depth, max_features=8)
        forest.append(nodes)

    # serialize
    d = X.shape[1]
    C = card["classes"]
    out = struct.pack("<4i", MAGIC, n_trees, d, C)
    for nodes in forest:
        out += struct.pack("<i", len(nodes))
        for nd in nodes:
            out += nd.pack()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-rf inference: ./infer weights.bin in.rec N
 * Random Forest with binary splits on byte features.
 * Weights format (little-endian):
 *   uint32 magic (0x464F5246), int32 n_trees, int32 n_features, int32 n_classes
 *   per tree: int32 n_nodes, then n_nodes * { uint8 feat, uint8 thresh, int16 left, int16 right, int8 value, uint8 pad } */
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

struct Node {
    uint8_t feat;
    uint8_t thresh;
    int16_t left;
    int16_t right;
    int8_t value;
};

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = *(uint32_t *)wb;
    if (magic != 0x464F5246) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_trees = *(int32_t *)(wb + 4);
    int32_t d = *(int32_t *)(wb + 8);
    int32_t C = *(int32_t *)(wb + 12);
    if (C != 2 || n_trees <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *wptr = wb + 16;
    struct Node **trees = malloc(n_trees * sizeof(struct Node *));
    int *tree_nodes = malloc(n_trees * sizeof(int));

    for (int t = 0; t < n_trees; t++) {
        if (wptr + 4 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        int n_nodes = *(int32_t *)wptr;
        wptr += 4;
        if (n_nodes <= 0 || wptr + (long)n_nodes * 8 > wb + nw) { fprintf(stderr, "bad tree size\n"); return 3; }
        tree_nodes[t] = n_nodes;
        trees[t] = (struct Node *)wptr;
        wptr += n_nodes * 8;
    }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int votes = 0;
        for (int t = 0; t < n_trees; t++) {
            int idx = 0;
            while (1) {
                struct Node *n = &trees[t][idx];
                if (n->left == -1 && n->right == -1) {
                    votes += n->value;
                    break;
                }
                idx = (xi[n->feat] <= n->thresh) ? n->left : n->right;
            }
        }
        unsigned char pred = (votes > n_trees / 2) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```