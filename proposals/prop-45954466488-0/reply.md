```hypothesis
Family: label-dtree (single deterministic decision tree). Adult has only 14 byte features; a depth-limited CART with histogram-based splits trains in seconds, uses no randomness, and yields a tiny model. A well-tuned tree should match or beat linear (0.8524) at far lower inference cost (≈100 Ir/item). If dev accuracy stays ≤0.852 or inference cost exceeds label-linear, the idea fails.
```
```python
"""Deterministic decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Builds a CART (Gini) with histogram splits on raw bytes. Hyper-parameters chosen on last 20% of train.in."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyper-parameter grid searched on the tuning split (last 20% of train.in)
GRID = {
    "max_depth": [5, 6, 7, 8, 9, 10],
    "min_samples_leaf": [1, 5, 10, 20],
    "min_samples_split": [2, 10, 20],
}

MAGIC = 0x44545245  # 'DTRE' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(counts):
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return 1.0 - np.dot(p, p)


def best_split(X, y, idx, min_samples_leaf):
    """Return (feature, threshold, gain) or (None, None, 0)."""
    n = len(idx)
    if n < 2:
        return None, None, 0.0
    parent = gini(np.bincount(y[idx], minlength=2))
    best_gain = 0.0
    best_feat = -1
    best_thr = -1
    # For each feature, build 256x2 histogram
    for j in range(X.shape[1]):
        vals = X[idx, j]
        # histogram per class
        hist = np.zeros((256, 2), dtype=np.int64)
        for c in (0, 1):
            mask = y[idx] == c
            if mask.any():
                hist[:, c] = np.bincount(vals[mask], minlength=256)
        cum = hist.cumsum(axis=0)          # left counts up to threshold t
        tot_left = cum.sum(axis=1)
        tot_right = n - tot_left
        # avoid empty leaves
        valid = (tot_left >= min_samples_leaf) & (tot_right >= min_samples_leaf)
        if not valid.any():
            continue
        # Gini for left and right at each threshold
        left_gini = 1.0 - (cum[:, 0] / tot_left) ** 2 - (cum[:, 1] / tot_left) ** 2
        right_counts = np.array([y[idx].sum(), n - y[idx].sum()]) - cum
        right_gini = 1.0 - (right_counts[:, 0] / tot_right) ** 2 - (right_counts[:, 1] / tot_right) ** 2
        gain = parent - (tot_left / n) * left_gini - (tot_right / n) * right_gini
        gain[~valid] = -1.0
        t = int(gain.argmax())
        if gain[t] > best_gain + 1e-12:
            best_gain = gain[t]
            best_feat = j
            best_thr = t
    if best_gain <= 1e-12:
        return None, None, 0.0
    return best_feat, best_thr, best_gain


def build_tree(X, y, idx, depth, max_depth, min_samples_leaf, min_samples_split,
               feat_arr, thr_arr, left_arr, right_arr):
    """Recursive builder, returns node index."""
    n = len(idx)
    # stopping criteria
    if depth == max_depth or n < min_samples_split or len(np.unique(y[idx])) == 1:
        leaf_class = int(np.bincount(y[idx], minlength=2).argmax())
        node = len(feat_arr)
        feat_arr.append(-1)
        thr_arr.append(leaf_class)
        left_arr.append(-1)
        right_arr.append(-1)
        return node
    feat, thr, gain = best_split(X, y, idx, min_samples_leaf)
    if feat is None:
        leaf_class = int(np.bincount(y[idx], minlength=2).argmax())
        node = len(feat_arr)
        feat_arr.append(-1)
        thr_arr.append(leaf_class)
        left_arr.append(-1)
        right_arr.append(-1)
        return node
    left_idx = idx[X[idx, feat] <= thr]
    right_idx = idx[X[idx, feat] > thr]
    node = len(feat_arr)
    feat_arr.append(feat)
    thr_arr.append(thr)
    left_arr.append(0)   # placeholder
    right_arr.append(0)
    left_child = build_tree(X, y, left_idx, depth + 1, max_depth, min_samples_leaf,
                            min_samples_split, feat_arr, thr_arr, left_arr, right_arr)
    right_child = build_tree(X, y, right_idx, depth + 1, max_depth, min_samples_leaf,
                             min_samples_split, feat_arr, thr_arr, left_arr, right_arr)
    left_arr[node] = left_child
    right_arr[node] = right_child
    return node


def train_one(X, y, max_depth, min_samples_leaf, min_samples_split):
    feat_arr, thr_arr, left_arr, right_arr = [], [], [], []
    idx = np.arange(len(y))
    build_tree(X, y, idx, 0, max_depth, min_samples_leaf, min_samples_split,
               feat_arr, thr_arr, left_arr, right_arr)
    return np.array(feat_arr, np.int8), np.array(thr_arr, np.uint8), \
           np.array(left_arr, np.int32), np.array(right_arr, np.int32)


def accuracy(X, y, feat, thr, left, right):
    n = len(y)
    correct = 0
    for i in range(n):
        node = 0
        while feat[node] != -1:
            f = feat[node]
            t = thr[node]
            node = left[node] if X[i, f] <= t else right[node]
        if thr[node] == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    n = len(X)
    tune_sz = n // 5
    tr_X, tr_y = X[:-tune_sz], y[:-tune_sz]
    tu_X, tu_y = X[-tune_sz:], y[-tune_sz:]

    best_acc = -1.0
    best_params = None
    best_arrays = None

    for md in GRID["max_depth"]:
        for msl in GRID["min_samples_leaf"]:
            for mss in GRID["min_samples_split"]:
                feat, thr, left, right = train_one(tr_X, tr_y, md, msl, mss)
                acc = accuracy(tu_X, tu_y, feat, thr, left, right)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (md, msl, mss)
                    best_arrays = (feat, thr, left, right)

    # Retrain on full data with best params
    md, msl, mss = best_params
    feat, thr, left, right = train_one(X, y, md, msl, mss)

    # Serialize
    num_nodes = len(feat)
    header = struct.pack("<4i", MAGIC, num_nodes, X.shape[1], 2)
    body = feat.tobytes() + thr.tobytes() + left.tobytes() + right.tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
```
```c
/* Decision tree inference for label-dtree: ./infer weights.bin in.rec N */
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
    if (nw < 16 || hd[0] != 0x44545245) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t num_nodes = hd[1], d = hd[2], C = hd[3];
    if (C != 2 || d != 14) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + (long)num_nodes * (1 + 1 + 4 + 4);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    int8_t *feat = (int8_t *)(hd + 4);
    uint8_t *thr = (uint8_t *)(feat + num_nodes);
    int32_t *left = (int32_t *)(thr + num_nodes);
    int32_t *right = (int32_t *)((uint8_t *)(thr + num_nodes) + num_nodes * 4); /* thr is 1 byte each, but left starts after thr + padding? */
    /* Actually layout: feat[num_nodes] (int8), thr[num_nodes] (uint8), left[num_nodes] (int32), right[num_nodes] (int32) */
    /* We must compute offsets carefully. */
    /* Let's recompute with char pointer. */
    unsigned char *base = (unsigned char *)(hd + 4);
    feat = (int8_t *)base;
    thr = base + num_nodes;
    left = (int32_t *)(thr + num_nodes);
    right = (int32_t *)((unsigned char *)left + num_nodes * 4);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (feat[node] != -1) {
            int f = feat[node];
            uint8_t t = thr[node];
            node = (xi[f] <= t) ? left[node] : right[node];
        }
        unsigned char cls = (unsigned char)thr[node];
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```