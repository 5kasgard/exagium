```hypothesis
Propose label-tree: a single deterministic decision tree (CART, Gini impurity) with max depth tuned on the last 20% of train.in. The 8x8 pixel grid has local spatial structure that axis-aligned splits can exploit non-linearly, while a depth-5..10 tree stays tiny (a few hundred nodes) for fast inference. If dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 5059 (label-linear cost), the family does not justify its existence.
```

```python
import json, math, struct, sys

import numpy as np

# deterministic, single-threaded
SEED = 1

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
    probs = counts / total
    return 1.0 - np.sum(probs * probs)

def best_split(X, y, n_classes):
    n_samples, n_features = X.shape
    best_gini = 1.0
    best_feature = -1
    best_threshold = -1
    # features are uint8 0..16 (card says 0..16)
    for f in range(n_features):
        # histogram: 17 possible values x n_classes
        hist = np.zeros((17, n_classes), dtype=np.int32)
        for v in range(17):
            mask = (X[:, f] == v)
            if np.any(mask):
                for c in range(n_classes):
                    hist[v, c] = np.sum(y[mask] == c)
        total_counts = hist.sum(axis=0)
        left_counts = np.zeros(n_classes, dtype=np.int32)
        for v in range(16):  # split between v and v+1
            left_counts += hist[v]
            right_counts = total_counts - left_counts
            left_total = left_counts.sum()
            right_total = right_counts.sum()
            if left_total == 0 or right_total == 0:
                continue
            gini = (left_total * gini_impurity(left_counts) +
                    right_total * gini_impurity(right_counts)) / n_samples
            if gini < best_gini:
                best_gini = gini
                best_feature = f
                best_threshold = v  # <= v go left
    return best_feature, best_threshold, best_gini

def build_tree(X, y, depth, max_depth, n_classes, nodes):
    # stopping criteria
    if depth >= max_depth or len(np.unique(y)) == 1 or X.shape[0] < 2:
        # leaf: majority class (ties: smallest class)
        counts = np.bincount(y, minlength=n_classes)
        leaf_class = int(np.argmax(counts))
        nodes.append((0xFFFFFFFF, leaf_class, -1, -1))
        return len(nodes) - 1

    feature, threshold, _ = best_split(X, y, n_classes)
    if feature == -1:
        counts = np.bincount(y, minlength=n_classes)
        leaf_class = int(np.argmax(counts))
        nodes.append((0xFFFFFFFF, leaf_class, -1, -1))
        return len(nodes) - 1

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    # if one side empty (should not happen because best_split checks), make leaf
    if not np.any(left_mask) or not np.any(right_mask):
        counts = np.bincount(y, minlength=n_classes)
        leaf_class = int(np.argmax(counts))
        nodes.append((0xFFFFFFFF, leaf_class, -1, -1))
        return len(nodes) - 1

    # create split node, but we need children indices first
    node_idx = len(nodes)
    nodes.append(None)  # placeholder
    left_idx = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, n_classes, nodes)
    right_idx = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, n_classes, nodes)
    nodes[node_idx] = (feature, threshold, left_idx, right_idx)
    return node_idx

def evaluate(nodes, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        xi = X[i]
        node = 0
        while True:
            feature, threshold, left, right = nodes[node]
            if feature == 0xFFFFFFFF:
                if threshold == y[i]:
                    correct += 1
                break
            if xi[feature] <= threshold:
                node = left
            else:
                node = right
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_classes = card["classes"]
    n = X.shape[0]

    # tuning split: last 20% in file order
    tune_size = n // 5
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    best_depth = 1
    best_acc = -1.0
    for max_depth in range(1, 11):
        nodes = []
        build_tree(train_X, train_y, 0, max_depth, n_classes, nodes)
        acc = evaluate(nodes, tune_X, tune_y)
        if acc > best_acc:
            best_acc = acc
            best_depth = max_depth

    # retrain on full data
    nodes = []
    build_tree(X, y, 0, best_depth, n_classes, nodes)

    # write weights.bin
    # magic: 0x45455254 ("TREE" little-endian)
    # uint32 num_nodes
    # each node: int32 feature, int32 threshold, int32 left, int32 right
    out = struct.pack("<II", 0x45455254, len(nodes))
    for node in nodes:
        out += struct.pack("<iiii", *node)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t *)wb;
    uint32_t num_nodes = *(uint32_t *)(wb + 4);
    if (magic != 0x45455254) { fprintf(stderr, "bad weights magic\n"); return 3; }
    if (nw != 8 + (long)num_nodes * 16) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 8);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int node_idx = 0;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.feature == -1) {  // leaf
                unsigned char cls = (unsigned char)nd.threshold;
                put_rec(stdout, &cls, 1);
                break;
            }
            if (xi[nd.feature] <= nd.threshold)
                node_idx = nd.left;
            else
                node_idx = nd.right;
        }
    }
    return 0;
}
```