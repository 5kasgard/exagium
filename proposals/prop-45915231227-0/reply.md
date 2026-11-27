```hypothesis
A single decision tree (max depth 10, min leaf 20) with greedy Gini splits on the raw byte features.
The Adult data has strong non-linear interactions (e.g., education × occupation) that a linear model cannot capture.
Tuning depth and min_samples_leaf on the last 20% of train.in should yield a model that beats label-linear (0.8524).
If dev accuracy ≤ 0.8524 or inference cost exceeds label-linear, the family does not improve on the baseline.
```
```python
"""Decision tree for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports a binary tree: magic 0x44545245 ('DTRE'), int32 n_nodes,
then n_nodes * 5 int32: feature, threshold, left, right, leaf_class (leaf_class=-1 for internal)."""
import json, struct, sys
import numpy as np

SEED = 1
MAX_DEPTH_GRID = [5, 10, 15]
MIN_LEAF_GRID = [10, 20, 50]
TUNING_FRACTION = 0.2


def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
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


def best_split(X, y, min_leaf):
    n, d = X.shape
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1
    parent_gini = gini(y)
    for feat in range(d):
        vals = X[:, feat]
        # thresholds are midpoints between sorted unique values, but bytes are 0-255.
        # We can try all 256 possible thresholds (<= thr goes left).
        # To speed up, we can compute histogram.
        hist = np.zeros((256, 2), dtype=np.int32)
        for v, c in zip(vals, y):
            hist[v, c] += 1
        left_counts = np.zeros(2, dtype=np.int32)
        right_counts = hist.sum(axis=0).copy()
        for thr in range(256):
            left_counts += hist[thr]
            right_counts -= hist[thr]
            nl = left_counts.sum()
            nr = right_counts.sum()
            if nl < min_leaf or nr < min_leaf:
                continue
            gain = parent_gini - (nl / n) * gini_from_counts(left_counts) - (nr / n) * gini_from_counts(right_counts)
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thr
    return best_feat, best_thresh, best_gain


def gini_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)


def build_tree(X, y, depth, max_depth, min_leaf, nodes):
    # returns node index
    if depth >= max_depth or len(y) < 2 * min_leaf or gini(y) == 0.0:
        leaf_class = int(np.bincount(y).argmax())
        nodes.append((-1, -1, -1, -1, leaf_class))
        return len(nodes) - 1
    feat, thresh, gain = best_split(X, y, min_leaf)
    if gain <= 0.0 or feat == -1:
        leaf_class = int(np.bincount(y).argmax())
        nodes.append((-1, -1, -1, -1, leaf_class))
        return len(nodes) - 1
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    left_idx = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_leaf, nodes)
    right_idx = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_leaf, nodes)
    nodes.append((feat, thresh, left_idx, right_idx, -1))
    return len(nodes) - 1


def train_tree(X, y, max_depth, min_leaf):
    nodes = []
    build_tree(X, y, 0, max_depth, min_leaf, nodes)
    return nodes


def evaluate(nodes, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        node_idx = 0
        while True:
            feat, thresh, left, right, leaf = nodes[node_idx]
            if feat == -1:
                if leaf == y[i]:
                    correct += 1
                break
            if X[i, feat] <= thresh:
                node_idx = left
            else:
                node_idx = right
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20% in file order
    tune_start = int(n * (1 - TUNING_FRACTION))
    X_train, y_train = X[:tune_start], y[:tune_start]
    X_tune, y_tune = X[tune_start:], y[tune_start:]

    best_acc = -1.0
    best_params = (MAX_DEPTH_GRID[0], MIN_LEAF_GRID[0])
    for max_depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_LEAF_GRID:
            nodes = train_tree(X_train, y_train, max_depth, min_leaf)
            acc = evaluate(nodes, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)

    # retrain on all data
    final_nodes = train_tree(X, y, best_params[0], best_params[1])

    # serialize
    magic = 0x44545245  # 'DTRE'
    n_nodes = len(final_nodes)
    out = struct.pack("<2i", magic, n_nodes)
    for feat, thresh, left, right, leaf in final_nodes:
        out += struct.pack("<5i", feat, thresh, left, right, leaf)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Decision tree inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class index.
 * Weights format: magic 0x44545245, int32 n_nodes, then n_nodes * (5 int32):
 *   feature, threshold, left, right, leaf_class (leaf_class=-1 for internal). */
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
    int32_t feature;
    int32_t threshold;
    int32_t left;
    int32_t right;
    int32_t leaf_class;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x44545245) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_nodes = hdr[1];
    long expected = 8 + (long)n_nodes * 5 * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(hdr + 2);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        int d = nodes[0].feature >= 0 ? 14 : 1; /* d is fixed by card, but we don't store it; we know it's 14 */
        if ((int)k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }
        int32_t idx = 0;
        while (1) {
            Node nd = nodes[idx];
            if (nd.feature == -1) {
                unsigned char out = (unsigned char)nd.leaf_class;
                put_rec(stdout, &out, 1);
                break;
            }
            if (xi[nd.feature] <= (unsigned char)nd.threshold)
                idx = nd.left;
            else
                idx = nd.right;
        }
    }
    return 0;
}
```