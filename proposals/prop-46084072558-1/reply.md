```hypothesis
Family: label-dtree (single decision tree with tuned max depth).
Why: Optdigits pixels have clear local patterns (strokes) that axis-aligned splits capture well. A shallow tree (depth 5–7) gives strong non-linearity with only a few comparisons per item, so it should beat label-linear (0.9613) at lower CPU cost.
Failure: If dev accuracy ≤ 0.9613 or instructions/item ≥ label-linear's 5058, the hypothesis is wrong.
```

```python
"""label-dtree: single decision tree for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Tunes max_depth on last 20% of train.in,
then trains on all train. Exports a flat array of nodes:
  int32 magic=0x44545231, n_nodes, n_features, n_classes
  per node: int32 feature_idx (or -1 for leaf), float32 threshold,
            int32 left, int32 right, int32 class_label (leaf only)."""
import json, math, struct, sys

import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 10
MAX_DEPTH_CANDIDATES = [3, 4, 5, 6, 7, 8]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "label")
    def __init__(self, feature=-1, threshold=0.0, left=-1, right=-1, label=-1):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self):
        return self.feature == -1


def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)


def best_split(X, y, feature_indices, rng):
    best_gain = -1.0
    best_feat = -1
    best_thresh = 0.0
    parent_gini = gini(y)
    n = len(y)
    for f in feature_indices:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        for t in thresholds:
            left_mask = vals <= t
            right_mask = ~left_mask
            if left_mask.sum() < MIN_SAMPLES_LEAF or right_mask.sum() < MIN_SAMPLES_LEAF:
                continue
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            gain = parent_gini - (n_left / n) * gini(y[left_mask]) - (n_right / n) * gini(y[right_mask])
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feat = f
                best_thresh = t
    return best_feat, best_thresh, best_gain


def build_tree(X, y, depth, max_depth, feature_indices, rng, nodes):
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2 * MIN_SAMPLES_LEAF:
        label = int(np.bincount(y).argmax())
        nodes.append(Node(label=label))
        return len(nodes) - 1

    feat, thresh, gain = best_split(X, y, feature_indices, rng)
    if feat == -1 or gain <= 1e-12:
        label = int(np.bincount(y).argmax())
        nodes.append(Node(label=label))
        return len(nodes) - 1

    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    node_idx = len(nodes)
    nodes.append(Node())  # placeholder
    left_idx = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, feature_indices, rng, nodes)
    right_idx = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, feature_indices, rng, nodes)
    nodes[node_idx] = Node(feature=feat, threshold=thresh, left=left_idx, right=right_idx)
    return node_idx


def predict_one(tree, x):
    idx = 0
    while True:
        node = tree[idx]
        if node.is_leaf():
            return node.label
        if x[node.feature] <= node.threshold:
            idx = node.left
        else:
            idx = node.right


def evaluate(tree, X, y):
    correct = 0
    for i in range(len(y)):
        if predict_one(tree, X[i]) == y[i]:
            correct += 1
    return correct / len(y)


def serialize_tree(tree, n_features, n_classes):
    n_nodes = len(tree)
    out = struct.pack("<4i", 0x44545231, n_nodes, n_features, n_classes)
    for node in tree:
        out += struct.pack("<iffii", node.feature, node.threshold, node.left, node.right, node.label)
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"]).astype(np.float32)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_features = X.shape[1]
    n_classes = card["classes"]
    n = len(y)

    # split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    feature_indices = np.arange(n_features)

    best_depth = MAX_DEPTH_CANDIDATES[0]
    best_acc = -1.0
    for d in MAX_DEPTH_CANDIDATES:
        nodes = []
        build_tree(X_train, y_train, 0, d, feature_indices, rng, nodes)
        acc = evaluate(nodes, X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_depth = d

    # retrain on all data
    nodes = []
    build_tree(X, y, 0, best_depth, feature_indices, rng, nodes)

    open(sys.argv[4], "wb").write(serialize_tree(nodes, n_features, n_classes))


if __name__ == "__main__":
    main()
```

```c
/* label-dtree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class.
 * Weights format: magic 0x44545231, n_nodes, d, C; then n_nodes x (int32 feature, float32 thresh, int32 left, int32 right, int32 label). */
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
    float threshold;
    int32_t left;
    int32_t right;
    int32_t label;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x44545231) { fprintf(stderr, "bad magic\n"); return 3; }
    int n_nodes = hd[1], d = hd[2], C = hd[3];
    if (n_nodes <= 0 || d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + (long)n_nodes * (4 + 4 + 4 + 4 + 4);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int idx = 0;
        while (1) {
            Node nd = nodes[idx];
            if (nd.feature == -1) { unsigned char out = (unsigned char)nd.label; put_rec(stdout, &out, 1); break; }
            float val = xi[nd.feature];
            idx = (val <= nd.threshold) ? nd.left : nd.right;
        }
    }
    return 0;
}
```