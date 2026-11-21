```hypothesis
Family: label-rf (random forest on the same one-hot/z-scored features as label-linear).
Why: Tabular data with mixed ordinal/categorical features often benefits from non-linear interactions that trees capture; a small forest (10-20 trees, depth 8-12) can exceed linear accuracy while inference remains a few hundred operations per item (far below GBDT cost).
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ 2000 (label-mlp16 territory), the family does not justify its existence.
```

```python
"""label-rf: Random forest for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-score, categorical as one-hot (unseen=all-zero).
Trains a random forest (bagged decision trees) with Gini splits, sqrt(D) features per node.
Hyperparameters (n_trees, max_depth, min_leaf) chosen on last 20% of train.in (tuning split),
then final model trained on all train.in. Exports flat node arrays + encoding params."""
import json, math, struct, sys
from array import array

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
N_TREES_GRID = [10, 20, 30]
MAX_DEPTH_GRID = [6, 8, 10, 12]
MIN_LEAF_GRID = [1, 5, 10]


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


def best_split(X, y, idx, feat_indices, rng):
    """Return (feature, threshold, left_idx, right_idx, gain) or None."""
    best_gain = -1.0
    best_f = best_t = None
    best_l = best_r = None
    parent_gini = gini(y[idx])
    n = len(idx)
    for f in feat_indices:
        vals = X[idx, f]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        # try midpoints between consecutive unique values
        thresholds = (uniq[:-1] + uniq[1:]) * 0.5
        for t in thresholds:
            left = idx[vals <= t]
            right = idx[vals > t]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = parent_gini - (len(left) * gini(y[left]) + len(right) * gini(y[right])) / n
            if gain > best_gain:
                best_gain = gain
                best_f, best_t = f, t
                best_l, best_r = left, right
    if best_gain <= 1e-9:
        return None
    return best_f, best_t, best_l, best_r, best_gain


def build_tree(X, y, idx, depth, max_depth, min_leaf, n_features, rng):
    """Returns list of nodes (flat arrays). Node: [feat, thresh, left, right, prob1]."""
    nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_prob = [], [], [], [], []
    def recurse(idxs, d):
        node_idx = len(nodes_feat)
        # leaf?
        if d >= max_depth or len(idxs) < min_leaf or gini(y[idxs]) == 0.0:
            nodes_feat.append(-1)
            nodes_thresh.append(0.0)
            nodes_left.append(-1)
            nodes_right.append(-1)
            nodes_prob.append(float(np.mean(y[idxs])))
            return node_idx
        # random feature subset
        feat_subset = rng.choice(n_features, size=int(math.sqrt(n_features)), replace=False)
        split = best_split(X, y, idxs, feat_subset, rng)
        if split is None:
            nodes_feat.append(-1)
            nodes_thresh.append(0.0)
            nodes_left.append(-1)
            nodes_right.append(-1)
            nodes_prob.append(float(np.mean(y[idxs])))
            return node_idx
        f, t, left_idx, right_idx, _ = split
        nodes_feat.append(f)
        nodes_thresh.append(t)
        # placeholders for children
        nodes_left.append(0)
        nodes_right.append(0)
        nodes_prob.append(0.0)
        left_node = recurse(left_idx, d + 1)
        right_node = recurse(right_idx, d + 1)
        nodes_left[node_idx] = left_node
        nodes_right[node_idx] = right_node
        return node_idx
    recurse(idx, 0)
    return nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_prob


def train_forest(X, y, n_trees, max_depth, min_leaf, rng):
    n, n_features = X.shape
    trees = []
    for t in range(n_trees):
        # bootstrap sample
        boot_idx = rng.choice(n, size=n, replace=True)
        tree_nodes = build_tree(X, y, boot_idx, 0, max_depth, min_leaf, n_features, rng)
        trees.append(tree_nodes)
    return trees


def encode_data(card, X_raw):
    """Return encoded matrix Z (n x D) and encoding metadata for inference."""
    n, d = X_raw.shape
    enc_meta = []  # list of dict per column
    Z_parts = []
    for j, typ in enumerate(card["input"]["types"]):
        col = X_raw[:, j].astype(np.float64)
        if typ == "ordinal":
            mu, sd = col.mean(), col.std() or 1.0
            Z_parts.append(((col - mu) / sd).astype(np.float32)[:, None])
            enc_meta.append({"type": "ordinal", "mean": float(mu), "std": float(sd)})
        else:
            vals = np.unique(col)
            K = len(vals)
            # map byte value -> one-hot index (0..K-1), 255 (unseen) -> -1
            mapping = np.full(256, -1, dtype=np.int32)
            for idx, v in enumerate(vals):
                mapping[int(v)] = idx
            # build one-hot for this column
            onehot = np.zeros((n, K), dtype=np.float32)
            for i in range(n):
                v = int(col[i])
                if mapping[v] != -1:
                    onehot[i, mapping[v]] = 1.0
            Z_parts.append(onehot)
            enc_meta.append({"type": "categorical", "K": K, "map": mapping})
    Z = np.concatenate(Z_parts, axis=1)
    return Z, enc_meta


def eval_forest(trees, X):
    """Return predicted class probabilities (n x 1) for forest."""
    n = X.shape[0]
    probs = np.zeros(n, dtype=np.float32)
    for nodes in trees:
        feat, thresh, left, right, prob = nodes
        for i in range(n):
            node = 0
            while feat[node] != -1:
                f = feat[node]
                if X[i, f] <= thresh[node]:
                    node = left[node]
                else:
                    node = right[node]
            probs[i] += prob[node]
    probs /= len(trees)
    return probs


def accuracy(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(np.int64)
    return np.mean(y_pred == y_true)


def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = len(y)
    # tuning split = last 20%
    n_tune = n_total // 5
    tune_idx = slice(n_total - n_tune, n_total)
    train_idx = slice(0, n_total - n_tune)

    # Encode on FULL training data (as per card: choose settings on tuning, then train on all)
    Z_full, enc_meta = encode_data(card, X_raw)
    D = Z_full.shape[1]

    # Hyperparameter search on tuning split
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    for n_trees in N_TREES_GRID:
        for max_depth in MAX_DEPTH_GRID:
            for min_leaf in MIN_LEAF_GRID:
                # train on train_idx only
                trees = train_forest(Z_full[train_idx], y[train_idx], n_trees, max_depth, min_leaf, rng)
                probs = eval_forest(trees, Z_full[tune_idx])
                acc = accuracy(y[tune_idx], probs)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_trees, max_depth, min_leaf)
    print(f"Tuning best: {best_params} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on ALL data with best params
    n_trees, max_depth, min_leaf = best_params
    final_trees = train_forest(Z_full, y, n_trees, max_depth, min_leaf, rng)

    # Serialize weights
    # Header: magic, n_trees, max_depth, D, C, d
    # Then per tree: n_nodes, then flat arrays (int32 feat, float32 thresh, int32 left, int32 right, float32 prob)
    # Then encoding meta: for each of d columns: type (0 ord, 1 cat), then if ord: mean, std; if cat: K, map[256]
    out = bytearray()
    out += struct.pack("<6i", 0x52464C31, n_trees, max_depth, D, 2, card["input"]["bytes"])
    for nodes in final_trees:
        feat, thresh, left, right, prob = nodes
        n_nodes = len(feat)
        out += struct.pack("<i", n_nodes)
        out += array('i', feat).tobytes()
        out += array('f', thresh).tobytes()
        out += array('i', left).tobytes()
        out += array('i', right).tobytes()
        out += array('f', prob).tobytes()
    for meta in enc_meta:
        if meta["type"] == "ordinal":
            out += struct.pack("<bff", 0, meta["mean"], meta["std"])
        else:
            K = meta["K"]
            out += struct.pack("<bi", 1, K)
            out += meta["map"].astype(np.int32).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rf inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output record: 1 byte (class 0 or 1). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* Node layout matches train.py: feat(int32), thresh(float32), left(int32), right(int32), prob(float32) */
typedef struct { int32_t feat; float thresh; int32_t left; int32_t right; float prob; } Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hdr = (int32_t *)wb;
    if (nw < 24 || hdr[0] != 0x52464C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int n_trees = hdr[1], max_depth = hdr[2], D = hdr[3], C = hdr[4], d = hdr[5];
    if (C != 2 || d != 14) { fprintf(stderr, "bad header dims\n"); return 3; }

    unsigned char *p = wb + 24;
    unsigned char *w_end = wb + nw;

    /* Load trees */
    Node **trees = malloc(n_trees * sizeof(Node *));
    int32_t *tree_nnodes = malloc(n_trees * sizeof(int32_t));
    for (int t = 0; t < n_trees; t++) {
        if (p + 4 > w_end) { fprintf(stderr, "truncated tree count\n"); return 3; }
        int32_t nn = *(int32_t *)p; p += 4;
        tree_nnodes[t] = nn;
        if (p + (long)nn * 20 > w_end) { fprintf(stderr, "truncated tree nodes\n"); return 3; }
        trees[t] = (Node *)p;
        p += nn * 20;
    }

    /* Load encoding metadata: 14 columns */
    float *ord_mean = malloc(14 * sizeof(float));
    float *ord_std = malloc(14 * sizeof(float));
    int *cat_K = malloc(14 * sizeof(int));
    int32_t **cat_map = malloc(14 * sizeof(int32_t *));
    int col_ordinal[14];
    for (int j = 0; j < 14; j++) {
        if (p + 1 > w_end) { fprintf(stderr, "truncated encoding type\n"); return 3; }
        int type = *p++;  // 0 ordinal, 1 categorical
        col_ordinal[j] = (type == 0);
        if (type == 0) {
            if (p + 8 > w_end) { fprintf(stderr, "truncated ordinal params\n"); return 3; }
            ord_mean[j] = *(float *)p; p += 4;
            ord_std[j] = *(float *)p; p += 4;
            cat_map[j] = NULL;
            cat_K[j] = 0;
        } else {
            if (p + 4 > w_end) { fprintf(stderr, "truncated cat K\n"); return 3; }
            cat_K[j] = *(int32_t *)p; p += 4;
            if (p + 256 * 4 > w_end) { fprintf(stderr, "truncated cat map\n"); return 3; }
            cat_map[j] = (int32_t *)p; p += 256 * 4;
            ord_mean[j] = ord_std[j] = 0.0f;
        }
    }
    if (p != w_end) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    /* Feature buffer */
    float *feat_vec = malloc(D * sizeof(float));
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Build feature vector */
        int feat_idx = 0;
        for (int j = 0; j < d; j++) {
            if (col_ordinal[j]) {
                feat_vec[feat_idx++] = (xi[j] - ord_mean[j]) / ord_std[j];
            } else {
                int idx = cat_map[j][xi[j]];
                for (int c = 0; c < cat_K[j]; c++)
                    feat_vec[feat_idx++] = (c == idx) ? 1.0f : 0.0f;
            }
        }

        /* Forest inference */
        float prob_sum = 0.0f;
        for (int t = 0; t < n_trees; t++) {
            Node *nodes = trees[t];
            int node = 0;
            while (nodes[node].feat != -1) {
                int f = nodes[node].feat;
                if (feat_vec[f] <= nodes[node].thresh)
                    node = nodes[node].left;
                else
                    node = nodes[node].right;
            }
            prob_sum += nodes[node].prob;
        }
        unsigned char pred = (prob_sum / n_trees > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```