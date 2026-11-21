```hypothesis
Family: label-dtree (single deterministic decision tree with histogram-based splits).
Why: Adult has mixed ordinal/categorical features; a shallow tree captures non-linear interactions that linear models miss, while inference is a few branch predictions per item (extremely low CPU cost). Hyperparameters (max_depth, min_samples_leaf) are chosen on the last 20% of train.in.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear (719 Ir/item).
```

```python
"""label-dtree: single decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: fixed seeds, single thread, histogram splits."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split (last 20% of train.in)
MAX_DEPTH_GRID = [3, 4, 5, 6, 7, 8]
MIN_SAMPLES_LEAF_GRID = [1, 5, 10, 20]
MIN_SAMPLES_SPLIT_GRID = [2, 5, 10]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini_from_counts(counts, total):
    if total == 0:
        return 0.0
    p0 = counts[0] / total
    p1 = counts[1] / total
    return 1.0 - p0 * p0 - p1 * p1

def find_best_split(indices, X, y, feature_types, min_samples_leaf):
    n = len(indices)
    if n < 2:
        return -1, -1, -1, -1.0
    y_sub = y[indices]
    total_counts = np.bincount(y_sub, minlength=2)
    parent_gini = gini_from_counts(total_counts, n)
    best_gain = -1.0
    best_feature = -1
    best_split_type = -1
    best_threshold = -1

    for f in range(X.shape[1]):
        vals = X[indices, f]
        if feature_types[f] == "ordinal":
            # histogram of class counts per bin (0-255)
            hist = np.zeros((256, 2), dtype=np.int64)
            for c in (0, 1):
                mask = (y_sub == c)
                if mask.any():
                    hist[:, c] = np.bincount(vals[mask], minlength=256)
            # cumulative sums from left
            cum = hist.cumsum(axis=0)
            # evaluate thresholds 0..254 (split: <= t vs > t)
            for t in range(255):
                left_n = cum[t, 0] + cum[t, 1]
                right_n = n - left_n
                if left_n < min_samples_leaf or right_n < min_samples_leaf:
                    continue
                left_counts = cum[t]
                right_counts = total_counts - left_counts
                left_gini = gini_from_counts(left_counts, left_n)
                right_gini = gini_from_counts(right_counts, right_n)
                weighted_gini = (left_n * left_gini + right_n * right_gini) / n
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_split_type = 0   # <=
                    best_threshold = t
        else:  # categorical
            # histogram per category
            hist = np.zeros((256, 2), dtype=np.int64)
            for c in (0, 1):
                mask = (y_sub == c)
                if mask.any():
                    hist[:, c] = np.bincount(vals[mask], minlength=256)
            # only categories present in this node
            present = (hist[:, 0] + hist[:, 1]) > 0
            for cat in np.where(present)[0]:
                left_counts = hist[cat]
                left_n = left_counts[0] + left_counts[1]
                right_n = n - left_n
                if left_n < min_samples_leaf or right_n < min_samples_leaf:
                    continue
                right_counts = total_counts - left_counts
                left_gini = gini_from_counts(left_counts, left_n)
                right_gini = gini_from_counts(right_counts, right_n)
                weighted_gini = (left_n * left_gini + right_n * right_gini) / n
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_split_type = 1   # ==
                    best_threshold = cat
    return best_feature, best_split_type, best_threshold, best_gain

def build_tree(X, y, feature_types, max_depth, min_samples_leaf, min_samples_split):
    nodes = []  # each node: (feature_idx, split_type, threshold, left, right, leaf_class)
    # recursive closure
    def build(indices, depth):
        n = len(indices)
        # stopping criteria
        if n < min_samples_split or depth == max_depth:
            leaf_class = int(np.bincount(y[indices]).argmax())
            nodes.append((-1, 0, 0, -1, -1, leaf_class))
            return len(nodes) - 1
        # find best split
        feat, spt, thr, gain = find_best_split(indices, X, y, feature_types, min_samples_leaf)
        if gain <= 0.0 or feat == -1:
            leaf_class = int(np.bincount(y[indices]).argmax())
            nodes.append((-1, 0, 0, -1, -1, leaf_class))
            return len(nodes) - 1
        # split indices
        vals = X[indices, feat]
        if spt == 0:
            left_mask = vals <= thr
        else:
            left_mask = vals == thr
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        # build children first (post-order)
        left_child = build(left_indices, depth + 1)
        right_child = build(right_indices, depth + 1)
        nodes.append((feat, spt, thr, left_child, right_child, 0))
        return len(nodes) - 1

    root = build(np.arange(len(y)), 0)
    return nodes

def evaluate(nodes, X, y):
    n = len(y)
    correct = 0
    for i in range(n):
        xi = X[i]
        node = 0
        while True:
            feat, spt, thr, left, right, leaf = nodes[node]
            if feat == -1:
                pred = leaf
                break
            val = xi[feat]
            if spt == 0:
                node = left if val <= thr else right
            else:
                node = left if val == thr else right
        if pred == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    feature_types = card["input"]["types"]
    n = len(X)
    tune_start = int(0.8 * n)
    train_idx = np.arange(tune_start)
    tune_idx = np.arange(tune_start, n)

    best_acc = -1.0
    best_params = None
    best_nodes = None

    for max_depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_SAMPLES_LEAF_GRID:
            for min_split in MIN_SAMPLES_SPLIT_GRID:
                nodes = build_tree(X[train_idx], y[train_idx], feature_types, max_depth, min_leaf, min_split)
                acc = evaluate(nodes, X[tune_idx], y[tune_idx])
                if acc > best_acc:
                    best_acc = acc
                    best_params = (max_depth, min_leaf, min_split)
                    best_nodes = nodes

    # retrain on full train with best params
    max_depth, min_leaf, min_split = best_params
    final_nodes = build_tree(X, y, feature_types, max_depth, min_leaf, min_split)

    # serialize: magic, num_nodes, then per node: int8 feat, int8 split_type, uint8 thr, int32 left, int32 right, uint8 leaf
    magic = 0x44545245  # 'DTRE'
    num_nodes = len(final_nodes)
    out = struct.pack("<II", magic, num_nodes)
    for feat, spt, thr, left, right, leaf in final_nodes:
        out += struct.pack("<bbBiiB", feat, spt, thr, left, right, leaf)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-dtree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class per record. */
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
    int8_t feat;
    int8_t split_type;
    uint8_t thr;
    int32_t left;
    int32_t right;
    uint8_t leaf;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t*)wb;
    uint32_t num_nodes = *(uint32_t*)(wb + 4);
    if (magic != 0x44545245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    size_t expected = 8 + num_nodes * (1 + 1 + 1 + 4 + 4 + 1);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = malloc(num_nodes * sizeof(Node));
    const unsigned char *p = wb + 8;
    for (uint32_t i = 0; i < num_nodes; i++) {
        nodes[i].feat = *(int8_t*)p; p += 1;
        nodes[i].split_type = *(int8_t*)p; p += 1;
        nodes[i].thr = *p; p += 1;
        nodes[i].left = *(int32_t*)p; p += 4;
        nodes[i].right = *(int32_t*)p; p += 4;
        nodes[i].leaf = *p; p += 1;
    }
    free(wb);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    int d = -1;
    if (N > 0) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        d = (int)k;
        pos = 0;  // reset to re-read first record
    }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (d >= 0 && (int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node_idx = 0;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.feat == -1) {
                put_rec(stdout, &nd.leaf, 1);
                break;
            }
            uint8_t val = xi[nd.feat];
            if (nd.split_type == 0) {
                node_idx = (val <= nd.thr) ? nd.left : nd.right;
            } else {
                node_idx = (val == nd.thr) ? nd.left : nd.right;
            }
        }
    }
    free(nodes);
    if (x) free(x);
    return 0;
}
```