```hypothesis
Family: label-tree (single deterministic CART decision tree).
Why: Optdigits has 64 ordinal pixel-count features (0–16) with local spatial structure; a shallow tree can capture axis-aligned decision boundaries that approximate digit shapes, outperforming linear models while remaining extremely fast (≈10 comparisons per item).
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds label-linear by >2×, the family does not meet its goal.
```

```python
"""label-tree: deterministic CART decision tree for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Tunes max_depth on last 20% of train.in (file order), then trains on all data.
Exports a flat BFS array of nodes: magic 0x54524545, num_nodes, max_depth, d, C;
each node 16 bytes: left(int32), right(int32), feature(uint8), is_leaf(uint8),
class(uint8), pad(uint8), threshold(float32)."""
import json, struct, sys

import numpy as np

SEED = 1
MIN_SAMPLES_SPLIT = 2
MAX_DEPTH_CANDIDATES = range(1, 31)


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
    probs = counts / len(y)
    return 1.0 - np.sum(probs * probs)


def find_best_split(X, y):
    n, d = X.shape
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1.0
    for f in range(d):
        vals = np.unique(X[:, f])
        for v in vals:
            left = X[:, f] <= v
            right = ~left
            if left.sum() < MIN_SAMPLES_SPLIT or right.sum() < MIN_SAMPLES_SPLIT:
                continue
            g = (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / n
            if g < best_gini or (g == best_gini and (f < best_feat or (f == best_feat and v < best_thresh))):
                best_gini = g
                best_feat = f
                best_thresh = float(v)
    return best_feat, best_thresh, best_gini


class Node:
    __slots__ = ("is_leaf", "cls", "feat", "thresh", "left", "right")
    def __init__(self, is_leaf, cls=0, feat=0, thresh=0.0, left=None, right=None):
        self.is_leaf = is_leaf
        self.cls = cls
        self.feat = feat
        self.thresh = thresh
        self.left = left
        self.right = right


def build_tree(X, y, depth, max_depth):
    if depth == max_depth or len(y) < MIN_SAMPLES_SPLIT or gini(y) == 0.0:
        cls = int(np.bincount(y).argmax())
        return Node(True, cls=cls)
    feat, thresh, _ = find_best_split(X, y)
    if feat == -1:
        cls = int(np.bincount(y).argmax())
        return Node(True, cls=cls)
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
    return Node(False, feat=feat, thresh=thresh, left=left, right=right)


def tree_to_bfs(root):
    nodes = []
    queue = [root]
    for node in queue:
        if not node.is_leaf:
            queue.append(node.left)
            queue.append(node.right)
    for node in queue:
        nodes.append(node)
    # assign indices
    for i, node in enumerate(nodes):
        if not node.is_leaf:
            node.left = nodes.index(node.left)
            node.right = nodes.index(node.right)
    return nodes


def evaluate_tree(nodes, X):
    preds = np.zeros(len(X), dtype=np.int64)
    for i, x in enumerate(X):
        idx = 0
        while not nodes[idx].is_leaf:
            if x[nodes[idx].feat] <= nodes[idx].thresh:
                idx = nodes[idx].left
            else:
                idx = nodes[idx].right
        preds[i] = nodes[idx].cls
    return preds


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    val_size = n // 5
    tr_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    best_depth = 1
    best_acc = -1.0
    for d in MAX_DEPTH_CANDIDATES:
        root = build_tree(X_tr, y_tr, 0, d)
        nodes = tree_to_bfs(root)
        preds = evaluate_tree(nodes, X_val)
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_depth = d

    # final training on all data
    root = build_tree(X, y, 0, best_depth)
    nodes = tree_to_bfs(root)

    # export
    num_nodes = len(nodes)
    d = card["input"]["bytes"]
    C = card["classes"]
    magic = 0x54524545  # "TREE"
    out = struct.pack("<5i", magic, num_nodes, best_depth, d, C)
    for node in nodes:
        left = node.left if not node.is_leaf else -1
        right = node.right if not node.is_leaf else -1
        feat = node.feat if not node.is_leaf else 0
        is_leaf = 1 if node.is_leaf else 0
        cls = node.cls if node.is_leaf else 0
        thresh = node.thresh if not node.is_leaf else 0.0
        out += struct.pack("<iiBBBBf", left, right, feat, is_leaf, cls, 0, thresh)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the predicted class.
 * Weights format: magic 0x54524545, num_nodes, max_depth, d, C (all int32).
 * Then num_nodes * 16 bytes per node: left(int32), right(int32),
 * feature(uint8), is_leaf(uint8), class(uint8), pad(uint8), threshold(float32). */
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
    int32_t left, right;
    uint8_t feature, is_leaf, cls, pad;
    float threshold;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t num_nodes = hd[1], max_depth = hd[2], d = hd[3], C = hd[4];
    if (d != 64 || C != 10 || num_nodes <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + (long)num_nodes * 16;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 20);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int idx = 0;
        while (!nodes[idx].is_leaf) {
            if (xi[nodes[idx].feature] <= nodes[idx].threshold)
                idx = nodes[idx].left;
            else
                idx = nodes[idx].right;
        }
        unsigned char out = nodes[idx].cls;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```