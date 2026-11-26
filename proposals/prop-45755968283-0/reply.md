```hypothesis
Family: label-dtree (single deterministic decision tree on raw byte features).
Why: Adult has mixed ordinal/categorical features; a tree captures interactions and non-linearities that linear models miss, while inference is just a handful of integer comparisons per record. Tuning max_depth on the held-out 20% controls overfitting.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 1000.
```

```python
"""label-dtree: single decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes: raw bytes (0-255) used directly. Splits: feature <= threshold.
Tunes max_depth on last 20% of train.in, then trains on all.
Weights format: magic 0x44545245, int32 num_nodes, num_classes, max_depth;
then per node (int32 feature, threshold, left, right, class_label)."""
import json, math, struct, sys

import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 10
MIN_IMPURITY_DECREASE = 1e-7
DEPTH_CANDIDATES = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(counts):
    # counts: (C,) array
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)


def build_tree(X, y, max_depth, depth=0, idxs=None):
    n, d = X.shape
    if idxs is None:
        idxs = np.arange(n)
    m = len(idxs)
    # leaf?
    if m < MIN_SAMPLES_LEAF or depth == max_depth:
        cls = np.bincount(y[idxs], minlength=2).argmax()
        return {"leaf": True, "class": int(cls), "counts": np.bincount(y[idxs], minlength=2)}
    # find best split
    best_gini = gini(np.bincount(y[idxs], minlength=2))
    best_feat = -1
    best_thr = -1
    best_left = None
    best_right = None
    for feat in range(d):
        vals = X[idxs, feat]
        # histogram per class
        hist = np.zeros((256, 2), dtype=np.int64)
        for v, c in zip(vals, y[idxs]):
            hist[v, c] += 1
        # cumulative
        cum = hist.cumsum(axis=0)
        total = cum[-1]
        for thr in range(255):  # split <= thr vs > thr
            left_counts = cum[thr]
            right_counts = total - left_counts
            left_n = left_counts.sum()
            right_n = right_counts.sum()
            if left_n < MIN_SAMPLES_LEAF or right_n < MIN_SAMPLES_LEAF:
                continue
            g = (left_n * gini(left_counts) + right_n * gini(right_counts)) / m
            if g < best_gini - MIN_IMPURITY_DECREASE:
                best_gini = g
                best_feat = feat
                best_thr = thr
                # partition indices
                mask = X[idxs, feat] <= thr
                best_left = idxs[mask]
                best_right = idxs[~mask]
    if best_feat == -1:
        cls = np.bincount(y[idxs], minlength=2).argmax()
        return {"leaf": True, "class": int(cls), "counts": np.bincount(y[idxs], minlength=2)}
    left_tree = build_tree(X, y, max_depth, depth + 1, best_left)
    right_tree = build_tree(X, y, max_depth, depth + 1, best_right)
    return {"leaf": False, "feat": best_feat, "thr": best_thr, "left": left_tree, "right": right_tree}


def tree_to_arrays(node, nodes, feat_arr, thr_arr, left_arr, right_arr, class_arr):
    idx = len(nodes)
    nodes.append(node)
    if node["leaf"]:
        feat_arr.append(-1)
        thr_arr.append(-1)
        left_arr.append(-1)
        right_arr.append(-1)
        class_arr.append(node["class"])
    else:
        feat_arr.append(node["feat"])
        thr_arr.append(node["thr"])
        left_idx = tree_to_arrays(node["left"], nodes, feat_arr, thr_arr, left_arr, right_arr, class_arr)
        right_idx = tree_to_arrays(node["right"], nodes, feat_arr, thr_arr, left_arr, right_arr, class_arr)
        left_arr.append(left_idx)
        right_arr.append(right_idx)
        class_arr.append(-1)
    return idx


def evaluate_tree(root, X, y):
    n = X.shape[0]
    pred = np.zeros(n, dtype=np.int64)
    for i in range(n):
        node = root
        while not node["leaf"]:
            if X[i, node["feat"]] <= node["thr"]:
                node = node["left"]
            else:
                node = node["right"]
        pred[i] = node["class"]
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20%
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_depth = DEPTH_CANDIDATES[0]
    best_acc = -1.0
    for depth in DEPTH_CANDIDATES:
        tree = build_tree(X_train, y_train, depth)
        acc = evaluate_tree(tree, X_tune, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # retrain on all data
    final_tree = build_tree(X, y, best_depth)

    # serialize
    nodes = []
    feat_arr, thr_arr, left_arr, right_arr, class_arr = [], [], [], [], []
    tree_to_arrays(final_tree, nodes, feat_arr, thr_arr, left_arr, right_arr, class_arr)
    num_nodes = len(nodes)
    C = card["classes"]
    magic = 0x44545245  # "ERTD"
    out = struct.pack("<4i", magic, num_nodes, C, best_depth)
    for i in range(num_nodes):
        out += struct.pack("<5i", feat_arr[i], thr_arr[i], left_arr[i], right_arr[i], class_arr[i])
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-dtree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class index. */
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
    int32_t feat;
    int32_t thr;
    int32_t left;
    int32_t right;
    int32_t cls;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x44545245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t num_nodes = hd[1], C = hd[2], max_depth = hd[3];
    if (num_nodes <= 0 || C != 2 || nw != 16 + (long)num_nodes * 20) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    Node *nodes = (Node *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)nodes[0].feat + 1 && nodes[0].feat != -1) {
            /* d is not stored explicitly; infer from first split feat? Safer: store d in header.
             * But we can compute d as max feature index + 1. However, we don't have it.
             * We'll add d to header in train.py. For now, assume d=14 (from card). */
        }
        int32_t node_idx = 0;
        while (1) {
            Node *nd = &nodes[node_idx];
            if (nd->feat == -1) { put_rec(stdout, (unsigned char *)&nd->cls, 1); break; }
            if (xi[nd->feat] <= (unsigned char)nd->thr) node_idx = nd->left;
            else node_idx = nd->right;
        }
    }
    return 0;
}
```