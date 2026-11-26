```hypothesis
A small random forest (10–30 trees, depth 5–7) on raw byte features captures non-linear interactions missed by the linear model. With ~14 features and 26k rows, shallow trees are fast to evaluate (a few hundred instructions/item) and deterministic via fixed bootstrap/feature seeds. If dev accuracy ≤ 0.855 (not clearly above label-linear’s 0.8524), the family does not justify its complexity.
```
```python
"""label-rf: small random forest for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Chooses n_trees, max_depth, min_leaf on last 20% of train,
then trains on all data. Weights format: magic 0x52463131, d, T, total_nodes; then T*(start, n_nodes);
then total_nodes*(feature, threshold, left, right) as int32 (leaf: feature=-1, threshold=class)."""
import json, math, struct, sys

import numpy as np

SEED = 1
TUNE_FRAC = 0.2
GRID = {
    "n_trees": [10, 20, 30],
    "max_depth": [5, 6, 7],
    "min_leaf": [1, 5, 10],
}


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


def best_split(X, y, feat_idx, rng):
    """Return (feature, threshold, gain, left_mask) for best split on given features."""
    n, d = X.shape
    best_gain = -1.0
    best_thr = 0
    best_f = -1
    best_mask = None
    parent_gini = gini(y)
    for f in feat_idx:
        vals = X[:, f]
        # thresholds from 0 to 254 (255 is special for unseen categorical)
        for thr in range(255):
            mask = vals <= thr
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            gain = parent_gini - (mask.sum() * gini(y[mask]) + (~mask).sum() * gini(y[~mask])) / n
            if gain > best_gain:
                best_gain = gain
                best_thr = thr
                best_f = f
                best_mask = mask
    return best_f, best_thr, best_gain, best_mask


def build_tree(X, y, max_depth, min_leaf, depth, rng, feat_subset):
    n = len(y)
    if depth >= max_depth or n < 2 * min_leaf or len(np.unique(y)) == 1:
        # leaf: majority class
        cls = int(np.bincount(y).argmax())
        return [(-1, cls, -1, -1)], 0  # nodes, root_index
    # choose random subset of features
    d = X.shape[1]
    n_feats = max(1, int(math.sqrt(d)))
    feats = rng.choice(d, n_feats, replace=False)
    f, thr, gain, mask = best_split(X, y, feats, rng)
    if gain <= 0 or f == -1:
        cls = int(np.bincount(y).argmax())
        return [(-1, cls, -1, -1)], 0
    left_nodes, left_root = build_tree(X[mask], y[mask], max_depth, min_leaf, depth + 1, rng, feat_subset)
    right_nodes, right_root = build_tree(X[~mask], y[~mask], max_depth, min_leaf, depth + 1, rng, feat_subset)
    # combine: current node at index 0, then left, then right
    nodes = [(f, thr, 1, 1 + len(left_nodes))]
    nodes.extend(left_nodes)
    nodes.extend(right_nodes)
    return nodes, 0


def train_forest(X, y, n_trees, max_depth, min_leaf, rng):
    n = len(y)
    trees = []
    for _ in range(n_trees):
        idx = rng.choice(n, n, replace=True)
        Xb, yb = X[idx], y[idx]
        nodes, root = build_tree(Xb, yb, max_depth, min_leaf, 0, rng, None)
        trees.append(nodes)
    return trees


def predict_forest(trees, X):
    n = X.shape[0]
    votes = np.zeros(n, dtype=np.int32)
    for nodes in trees:
        # iterative traversal
        node_idx = np.zeros(n, dtype=np.int32)
        active = np.ones(n, dtype=bool)
        while active.any():
            for i in np.where(active)[0]:
                f, thr, left, right = nodes[node_idx[i]]
                if f == -1:
                    votes[i] += thr
                    active[i] = False
                else:
                    if X[i, f] <= thr:
                        node_idx[i] = left
                    else:
                        node_idx[i] = right
    return (votes > len(trees) // 2).astype(np.int64)


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]

    # split tuning (last 20%) and train (first 80%)
    n = len(y)
    split = int(n * (1 - TUNE_FRAC))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    for n_trees in GRID["n_trees"]:
        for max_depth in GRID["max_depth"]:
            for min_leaf in GRID["min_leaf"]:
                trees = train_forest(X_tr, y_tr, n_trees, max_depth, min_leaf, rng)
                pred = predict_forest(trees, X_te)
                acc = accuracy(y_te, pred)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_trees, max_depth, min_leaf)
    print(f"tuning: best {best_params} acc={best_acc:.4f}", file=sys.stderr)

    # retrain on all data
    rng = np.random.default_rng(SEED)
    trees = train_forest(X, y, *best_params, rng)

    # serialize
    # count total nodes and compute offsets
    tree_infos = []
    total_nodes = 0
    for nodes in trees:
        tree_infos.append((total_nodes, len(nodes)))
        total_nodes += len(nodes)

    magic = 0x52463131
    header = struct.pack("<4i", magic, d, best_params[0], total_nodes)
    tree_info_bytes = b"".join(struct.pack("<2i", start, n_nodes) for start, n_nodes in tree_infos)
    node_bytes = b""
    for nodes in trees:
        for f, thr, left, right in nodes:
            node_bytes += struct.pack("<4i", f, thr, left, right)

    expected = 16 + 8 * best_params[0] + 16 * total_nodes
    out = header + tree_info_bytes + node_bytes
    assert len(out) == expected
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-rf inference: ./infer weights.bin in.rec N
 * Weights format: magic 0x52463131, d, T, total_nodes; T*(start, n_nodes); total_nodes*(feature, threshold, left, right).
 * Each input record is d bytes; output one byte (class 0 or 1) per record. */
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
    if (nw < 16 || hd[0] != 0x52463131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], T = hd[2], total_nodes = hd[3];
    if (d < 1 || d > 256 || T < 1 || total_nodes < T) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 8L * T + 16L * total_nodes;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    int32_t *tree_info = hd + 4;                 // T * 2 int32
    int32_t *nodes = tree_info + 2 * T;          // total_nodes * 4 int32

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int votes = 0;
        for (int t = 0; t < T; t++) {
            int start = tree_info[2 * t];
            int n_nodes = tree_info[2 * t + 1];
            int node_idx = 0;
            while (1) {
                int32_t *nd = nodes + 4 * (start + node_idx);
                int f = nd[0], thr = nd[1], left = nd[2], right = nd[3];
                if (f == -1) { votes += thr; break; }
                if (xi[f] <= (unsigned char)thr) node_idx = left;
                else node_idx = right;
            }
        }
        unsigned char cls = (votes > T / 2) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```