```hypothesis
Family: label-tree (single CART decision tree on raw binned features).
Why: The Adult features are already quantized (ordinal: 255 bins; categorical: category indices). A shallow tree can capture non-linear interactions without one-hot explosion, and inference is just a few integer comparisons per record. Tuning max_depth on the last 20% of train.in should yield a model that beats label-linear (0.8524) at a fraction of the CPU cost.
Failure: If dev accuracy is below 0.85, the tree is too weak for this task.
```

```python
"""label-tree: single CART decision tree on raw byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, struct, sys

import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 10  # fixed, not tuned


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini_impurity(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)


def best_split(X, y, n_classes):
    n, d = X.shape
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    for j in range(d):
        # values are 0..255
        cnt = np.zeros((256, n_classes), dtype=np.int64)
        np.add.at(cnt, (X[:, j], y), 1)
        left_cnt = np.zeros(n_classes, dtype=np.int64)
        right_cnt = cnt.sum(axis=0)
        for v in range(255):  # split <= v
            left_cnt += cnt[v]
            right_cnt -= cnt[v]
            left_total = left_cnt.sum()
            right_total = right_cnt.sum()
            if left_total < MIN_SAMPLES_LEAF or right_total < MIN_SAMPLES_LEAF:
                continue
            gini = (left_total * gini_impurity(left_cnt) +
                    right_total * gini_impurity(right_cnt)) / n
            if gini < best_gini:
                best_gini = gini
                best_feat = j
                best_thresh = v
    return best_feat, best_thresh, best_gini


def build_tree(X, y, depth, max_depth, n_classes, nodes):
    n = X.shape[0]
    # stopping criteria
    if depth == max_depth or n < 2 * MIN_SAMPLES_LEAF:
        leaf_val = np.bincount(y, minlength=n_classes).argmax()
        nodes.append((-1, 0, -1, -1, leaf_val))  # feature=-1 marks leaf
        return len(nodes) - 1

    feat, thresh, _ = best_split(X, y, n_classes)
    if feat == -1:
        leaf_val = np.bincount(y, minlength=n_classes).argmax()
        nodes.append((-1, 0, -1, -1, leaf_val))
        return len(nodes) - 1

    mask = X[:, feat] <= thresh
    left_idx = build_tree(X[mask], y[mask], depth + 1, max_depth, n_classes, nodes)
    right_idx = build_tree(X[~mask], y[~mask], depth + 1, max_depth, n_classes, nodes)
    nodes.append((feat, thresh, left_idx, right_idx, 0))
    return len(nodes) - 1


def predict_tree(X, nodes):
    n = X.shape[0]
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        node = 0
        while True:
            feat, thresh, left, right, val = nodes[node]
            if feat == -1:
                out[i] = val
                break
            if X[i, feat] <= thresh:
                node = left
            else:
                node = right
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]

    # validation split: last 20% in file order
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    # tune max_depth
    best_depth, best_acc = 3, 0.0
    for max_depth in range(3, 11):
        nodes = []
        build_tree(X_tr, y_tr, 0, max_depth, C, nodes)
        pred = predict_tree(X_val, nodes)
        acc = (pred == y_val).mean()
        if acc > best_acc:
            best_acc, best_depth = acc, max_depth

    # train on full data
    nodes = []
    build_tree(X, y, 0, best_depth, C, nodes)

    # serialize: magic, n_nodes, d, C, then arrays
    n_nodes = len(nodes)
    feat_arr = np.array([n[0] for n in nodes], dtype=np.int8)
    thresh_arr = np.array([n[1] for n in nodes], dtype=np.uint8)
    left_arr = np.array([n[2] for n in nodes], dtype=np.int32)
    right_arr = np.array([n[3] for n in nodes], dtype=np.int32)
    val_arr = np.array([n[4] for n in nodes], dtype=np.uint8)

    magic = 0x54524545  # "TREE"
    header = struct.pack("<4i", magic, n_nodes, d, C)
    body = (feat_arr.tobytes() + thresh_arr.tobytes() +
            left_arr.tobytes() + right_arr.tobytes() +
            val_arr.tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    np.random.seed(SEED)
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t n_nodes = hd[1], d = hd[2], C = hd[3];
    if (n_nodes <= 0 || d != 14 || C != 2) { fprintf(stderr, "bad header values\n"); return 3; }
    size_t expected = 16 + (size_t)n_nodes * (1 + 1 + 4 + 4 + 1);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const int8_t *feat = (const int8_t *)(hd + 4);
    const uint8_t *thresh = (const uint8_t *)(feat + n_nodes);
    const int32_t *left = (const int32_t *)(thresh + n_nodes);
    const int32_t *right = (const int32_t *)(left + n_nodes);
    const uint8_t *val = (const uint8_t *)(right + n_nodes);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int32_t node = 0;
        while (1) {
            int8_t f = feat[node];
            if (f == -1) { put_rec(stdout, &val[node], 1); break; }
            uint8_t t = thresh[node];
            if (xi[f] <= t) node = left[node];
            else node = right[node];
        }
    }
    return 0;
}
```