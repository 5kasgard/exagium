```hypothesis
Family: label-tree (single decision tree with tuned max depth).
Why: Optdigits pixels are ordinal 0-16; a shallow tree can capture local intensity thresholds and spatial interactions that a linear model misses, while inference is just a handful of comparisons per sample (extremely low CPU cost). We tune max_depth on the last 20% of train.in, then retrain on all train.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds label-linear, the hypothesis is wrong.
```

```python
"""label-tree: single decision tree for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Tunes max_depth on last 20% of train.in, then trains on all train.
Exports weights.bin: magic 0x44544C31, int32 num_nodes, max_depth, classes, features;
then per-node (pre-order): uint8 is_leaf, feature, threshold, pad; int32 left, right; uint8 class, pad[3]."""
import json, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 5
MIN_SAMPLES_SPLIT = 10
MAX_DEPTH_CANDIDATES = list(range(1, 16))  # 1..15

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ("is_leaf", "feature", "threshold", "left", "right", "class_label")
    def __init__(self, is_leaf, feature=0, threshold=0, left=-1, right=-1, class_label=0):
        self.is_leaf = is_leaf
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.class_label = class_label

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, rng):
    n, d = X.shape
    best_gain = -1.0
    best_feat = 0
    best_thresh = 0
    parent_gini = gini(y)
    for feat in range(d):
        vals = X[:, feat]
        # thresholds between 0..15 (since values 0..16)
        for thresh in range(16):
            left_mask = vals <= thresh
            right_mask = ~left_mask
            nl, nr = left_mask.sum(), right_mask.sum()
            if nl < MIN_SAMPLES_LEAF or nr < MIN_SAMPLES_LEAF:
                continue
            gain = parent_gini - (nl/n)*gini(y[left_mask]) - (nr/n)*gini(y[right_mask])
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh, best_gain

def build_tree(X, y, depth, max_depth, rng):
    # stopping criteria
    if depth >= max_depth or len(y) < MIN_SAMPLES_SPLIT:
        cls = np.bincount(y).argmax()
        return Node(True, class_label=int(cls))
    if len(np.unique(y)) == 1:
        return Node(True, class_label=int(y[0]))
    feat, thresh, gain = best_split(X, y, rng)
    if gain <= 1e-12:
        cls = np.bincount(y).argmax()
        return Node(True, class_label=int(cls))
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    left_node = build_tree(X[left_mask], y[left_mask], depth+1, max_depth, rng)
    right_node = build_tree(X[right_mask], y[right_mask], depth+1, max_depth, rng)
    return Node(False, feature=feat, threshold=thresh, left=left_node, right=right_node)

def flatten_tree(root):
    """Pre-order traversal, returns list of Nodes with left/right as indices."""
    nodes = []
    def dfs(node):
        idx = len(nodes)
        nodes.append(node)
        if not node.is_leaf:
            node.left = dfs(node.left)
            node.right = dfs(node.right)
        return idx
    dfs(root)
    # convert child Node objects to indices
    for i, node in enumerate(nodes):
        if not node.is_leaf:
            # left and right are already indices because dfs returns index
            pass
    return nodes

def evaluate_tree(nodes, X):
    preds = np.zeros(X.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        idx = 0
        while True:
            node = nodes[idx]
            if node.is_leaf:
                preds[i] = node.class_label
                break
            if X[i, node.feature] <= node.threshold:
                idx = node.left
            else:
                idx = node.right
    return preds

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_depth = 1
    best_acc = -1.0
    for depth in MAX_DEPTH_CANDIDATES:
        root = build_tree(X_train, y_train, 0, depth, rng)
        nodes = flatten_tree(root)
        preds = evaluate_tree(nodes, X_tune)
        acc = (preds == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # retrain on all data
    root = build_tree(X, y, 0, best_depth, rng)
    nodes = flatten_tree(root)

    # serialize
    magic = 0x44544C31  # "DTL1"
    num_nodes = len(nodes)
    num_classes = card["classes"]
    num_features = card["input"]["bytes"]
    header = struct.pack("<5i", magic, num_nodes, best_depth, num_classes, num_features)
    body = bytearray()
    for node in nodes:
        body.append(1 if node.is_leaf else 0)
        body.append(node.feature)
        body.append(node.threshold)
        body.append(0)  # pad
        body.extend(struct.pack("<ii", node.left, node.right))
        body.append(node.class_label)
        body.extend(b"\x00\x00\x00")  # pad
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes. Output record: 1 byte class index. */
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
    uint8_t pad0;
    int32_t left;
    int32_t right;
    uint8_t class_label;
    uint8_t pad1[3];
} Node;
#pragma pack(pop)

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x44544C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int num_nodes = hd[1], max_depth = hd[2], C = hd[3], d = hd[4];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + (long)num_nodes * 16;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    Node *nodes = (Node *)(wb + 20);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int idx = 0;
        while (1) {
            Node *nd = &nodes[idx];
            if (nd->is_leaf) {
                put_rec(stdout, &nd->class_label, 1);
                break;
            }
            if (xi[nd->feature] <= nd->threshold) idx = nd->left;
            else idx = nd->right;
        }
    }
    return 0;
}
```