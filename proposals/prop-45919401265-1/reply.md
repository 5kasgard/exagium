```hypothesis
Family: label-rf (random forest with shallow trees). The 8x8 optdigits pixels have strong local interactions (strokes) that linear models miss; an ensemble of small decision trees captures these non-linearities with few parameters. Trees use ordinal splits on raw 0..16 pixel counts, so no per-byte scaling tables are needed, keeping weights tiny and inference fast. If dev accuracy does not exceed 0.965, the family does not justify its complexity over label-linear.
```
```python
"""Random forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Chooses n_trees and max_depth on the last 20% of train.in,
then trains on all data. Exports a flat array of nodes: int32 magic 0x52463130, n_trees, n_features,
n_classes; then per tree: root_index, n_nodes; then global node array.
Node (52 bytes): uint8 is_leaf, feature, threshold, pad; int32 left, right; float32 probs[10]."""
import json, math, struct, sys

import numpy as np

SEED = 1
GRID_TREES = [10, 20, 30]
GRID_DEPTH = [10, 15, 20]
MIN_SAMPLES_SPLIT = 5
N_FEATURES_SUBSET = 8  # sqrt(64)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)


def best_split(X, y, rng, features):
    n, d = X.shape
    best_gini = 1.0
    best_feat = -1
    best_thr = -1
    for f in features:
        vals = np.unique(X[:, f])
        if len(vals) <= 1:
            continue
        thresholds = (vals[:-1] + vals[1:]) / 2.0
        for thr in thresholds:
            left = X[:, f] <= thr
            right = ~left
            if left.sum() < MIN_SAMPLES_SPLIT or right.sum() < MIN_SAMPLES_SPLIT:
                continue
            g = (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / n
            if g < best_gini:
                best_gini = g
                best_feat = f
                best_thr = thr
    return best_feat, best_thr, best_gini


class Node:
    __slots__ = ("is_leaf", "feature", "threshold", "left", "right", "probs")
    def __init__(self):
        self.is_leaf = 0
        self.feature = 0
        self.threshold = 0
        self.left = -1
        self.right = -1
        self.probs = None


def build_tree(X, y, depth, max_depth, rng, n_classes):
    node = Node()
    if depth >= max_depth or len(y) < MIN_SAMPLES_SPLIT or gini(y) == 0:
        node.is_leaf = 1
        counts = np.bincount(y, minlength=n_classes)
        node.probs = (counts + 1.0) / (len(y) + n_classes)
        return node
    features = rng.choice(X.shape[1], N_FEATURES_SUBSET, replace=False)
    feat, thr, _ = best_split(X, y, rng, features)
    if feat == -1:
        node.is_leaf = 1
        counts = np.bincount(y, minlength=n_classes)
        node.probs = (counts + 1.0) / (len(y) + n_classes)
        return node
    node.is_leaf = 0
    node.feature = feat
    node.threshold = thr
    left_mask = X[:, feat] <= thr
    right_mask = ~left_mask
    node.left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, rng, n_classes)
    node.right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, rng, n_classes)
    return node


def flatten_tree(root):
    nodes = []
    def dfs(node):
        idx = len(nodes)
        nodes.append(node)
        if not node.is_leaf:
            node.left = dfs(node.left)
            node.right = dfs(node.right)
        return idx
    root_idx = dfs(root)
    return root_idx, nodes


def predict_tree(nodes, root_idx, x):
    idx = root_idx
    while True:
        node = nodes[idx]
        if node.is_leaf:
            return node.probs
        if x[node.feature] <= node.threshold:
            idx = node.left
        else:
            idx = node.right


def evaluate(forest, X, y):
    n = len(y)
    correct = 0
    for i in range(n):
        probs = np.zeros(10, dtype=np.float32)
        for root_idx, nodes in forest:
            probs += predict_tree(nodes, root_idx, X[i])
        pred = np.argmax(probs)
        if pred == y[i]:
            correct += 1
    return correct / n


def train_forest(X, y, n_trees, max_depth, rng, n_classes):
    n = len(y)
    forest = []
    for _ in range(n_trees):
        idx = rng.choice(n, n, replace=True)
        root = build_tree(X[idx], y[idx], 0, max_depth, rng, n_classes)
        root_idx, nodes = flatten_tree(root)
        forest.append((root_idx, nodes))
    return forest


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # tuning split: last 20%
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_acc = -1.0
    best_params = None
    for n_trees in GRID_TREES:
        for max_depth in GRID_DEPTH:
            forest = train_forest(X_train, y_train, n_trees, max_depth, rng, C)
            acc = evaluate(forest, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (n_trees, max_depth)

    n_trees, max_depth = best_params
    rng = np.random.default_rng(SEED)  # reset for final training
    forest = train_forest(X, y, n_trees, max_depth, rng, C)

    # flatten all nodes into a single array
    all_nodes = []
    tree_info = []  # (root_idx, n_nodes)
    for root_idx, nodes in forest:
        tree_info.append((len(all_nodes) + root_idx, len(nodes)))
        all_nodes.extend(nodes)

    # write weights.bin
    magic = 0x52463130  # "RF10"
    out = struct.pack("<4i", magic, n_trees, d, C)
    for root_idx, n_nodes in tree_info:
        out += struct.pack("<2i", root_idx, n_nodes)
    for node in all_nodes:
        out += struct.pack("<4B2i", node.is_leaf, node.feature, int(node.threshold), 0, node.left, node.right)
        out += node.probs.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Random forest inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output: one byte class (argmax of averaged probs). */
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

#pragma pack(push, 1)
typedef struct {
    uint8_t is_leaf;
    uint8_t feature;
    uint8_t threshold;
    uint8_t pad;
    int32_t left;
    int32_t right;
    float probs[10];
} Node;
#pragma pack(pop)

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x52463130) { fprintf(stderr, "bad magic\n"); return 3; }
    int n_trees = hd[1], d = hd[2], C = hd[3];
    if (C != 10 || d != 64) { fprintf(stderr, "bad dims\n"); return 3; }
    int32_t *tree_info = (int32_t *)(wb + 16);
    Node *nodes = (Node *)(wb + 16 + n_trees * 8);
    long expected_nodes = (nw - 16 - n_trees * 8) / sizeof(Node);
    if ((nw - 16 - n_trees * 8) % sizeof(Node) != 0) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float acc_probs[10];
    unsigned char xi[64];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *rec = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(xi, rec, d);
        for (int c = 0; c < C; c++) acc_probs[c] = 0.0f;
        for (int t = 0; t < n_trees; t++) {
            int32_t root = tree_info[2 * t];
            int32_t n_nodes = tree_info[2 * t + 1];
            int idx = root;
            while (1) {
                Node *nd = &nodes[idx];
                if (nd->is_leaf) {
                    for (int c = 0; c < C; c++) acc_probs[c] += nd->probs[c];
                    break;
                }
                if (xi[nd->feature] <= nd->threshold) idx = nd->left;
                else idx = nd->right;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc_probs[c] > acc_probs[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```