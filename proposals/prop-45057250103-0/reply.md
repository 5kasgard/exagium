```hypothesis
Family: label-tree (single CART on raw byte features)
Why: The 14 input bytes are already binned ordinal (ordered) and categorical indices; a shallow decision tree can capture non-linear interactions and threshold effects that a linear model misses, while inference is just a handful of integer compares per item (extremely fast). If dev accuracy does not exceed label-linear's 0.8524, the task is essentially linear and trees add no value.
```

```python
"""label-tree: single CART decision tree on raw byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 20
MAX_DEPTH_CANDIDATES = range(1, 11)  # 1..10
TUNING_FRACTION = 0.2


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


def best_split(X, y, idxs):
    """Return (feature, threshold, gain, left_idxs, right_idxs) or None."""
    n = len(idxs)
    if n < 2 * MIN_SAMPLES_LEAF:
        return None
    parent_gini = gini(y[idxs])
    best_gain = -1.0
    best_feat = best_thr = None
    best_left = best_right = None

    for f in range(X.shape[1]):
        vals = X[idxs, f]
        uniq = np.unique(vals)
        if len(uniq) == 1:
            continue
        # thresholds are the unique values (split <= t vs > t)
        for t in uniq[:-1]:  # last value would put all in left
            left_mask = vals <= t
            right_mask = ~left_mask
            if left_mask.sum() < MIN_SAMPLES_LEAF or right_mask.sum() < MIN_SAMPLES_LEAF:
                continue
            left_idxs = idxs[left_mask]
            right_idxs = idxs[right_mask]
            gain = parent_gini - (len(left_idxs) * gini(y[left_idxs]) +
                                  len(right_idxs) * gini(y[right_idxs])) / n
            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thr = t
                best_left = left_idxs
                best_right = right_idxs
    if best_gain <= 0:
        return None
    return best_feat, best_thr, best_gain, best_left, best_right


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "label")
    def __init__(self, feature=-1, threshold=-1, left=-1, right=-1, label=-1):
        self.feature = feature      # -1 for leaf
        self.threshold = threshold  # -1 for leaf
        self.left = left            # index in nodes array
        self.right = right
        self.label = label          # 0 or 1 for leaf


def build_tree(X, y, max_depth):
    nodes = []
    root_idx = 0
    nodes.append(Node())  # placeholder
    stack = [(root_idx, np.arange(len(y)), 0)]  # (node_idx, sample_idxs, depth)

    while stack:
        node_idx, idxs, depth = stack.pop()
        if depth == max_depth or len(idxs) < 2 * MIN_SAMPLES_LEAF:
            # make leaf
            label = int(np.bincount(y[idxs]).argmax())
            nodes[node_idx] = Node(label=label)
            continue

        split = best_split(X, y, idxs)
        if split is None:
            label = int(np.bincount(y[idxs]).argmax())
            nodes[node_idx] = Node(label=label)
            continue

        feat, thr, _, left_idxs, right_idxs = split
        left_idx = len(nodes)
        nodes.append(Node())
        right_idx = len(nodes)
        nodes.append(Node())
        nodes[node_idx] = Node(feature=feat, threshold=thr, left=left_idx, right=right_idx)
        # push right then left so left is processed first (order doesn't matter for correctness)
        stack.append((right_idx, right_idxs, depth + 1))
        stack.append((left_idx, left_idxs, depth + 1))
    return nodes


def evaluate(nodes, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        xi = X[i]
        node = nodes[0]
        while node.feature != -1:
            if xi[node.feature] <= node.threshold:
                node = nodes[node.left]
            else:
                node = nodes[node.right]
        if node.label == y[i]:
            correct += 1
    return correct / n


def serialize(nodes, d, C):
    # magic 0x54524545 ("TREE" little-endian)
    magic = 0x54524545
    n_nodes = len(nodes)
    # header: magic, d, n_nodes, C, max_depth (not needed), min_samples_leaf
    hdr = struct.pack("<5i", magic, d, n_nodes, C, MIN_SAMPLES_LEAF)
    body = bytearray()
    for nd in nodes:
        body += struct.pack("<5i", nd.feature, nd.threshold, nd.left, nd.right, nd.label)
    return hdr + body


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    n = len(y)
    split = int(n * (1 - TUNING_FRACTION))
    idx = np.arange(n)
    train_idx, tune_idx = idx[:split], idx[split:]

    best_depth = 1
    best_acc = -1.0
    rng = np.random.default_rng(SEED)

    for depth in MAX_DEPTH_CANDIDATES:
        # deterministic: no randomness in tree building
        nodes = build_tree(X[train_idx], y[train_idx], depth)
        acc = evaluate(nodes, X[tune_idx], y[tune_idx])
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # retrain on all data
    final_nodes = build_tree(X, y, best_depth)
    blob = serialize(final_nodes, d, C)
    open(sys.argv[4], "wb").write(blob)


if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1). */
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
    int32_t label;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], n_nodes = hd[2], C = hd[3], min_leaf = hd[4];
    if (d != 14 || C != 2 || n_nodes <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 20L + (long)n_nodes * 5 * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    Node *nodes = (Node *)(hd + 5);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node_idx = 0;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.feature == -1) {          // leaf
                unsigned char cls = (unsigned char)nd.label;
                put_rec(stdout, &cls, 1);
                break;
            }
            if (xi[nd.feature] <= (unsigned char)nd.threshold)
                node_idx = nd.left;
            else
                node_idx = nd.right;
        }
    }
    return 0;
}
```