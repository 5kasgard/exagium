```hypothesis
Family: label-tree (single deterministic CART decision tree with tuned max depth).
Why: The 8x8 optdigits images have strong local pixel interactions that a linear model cannot capture, but a shallow tree can model with few splits. A single tree is extremely fast at inference (only integer comparisons) and should exceed label-linear's 0.9613 accuracy while keeping CPU cost very low.
Failure signal: If dev accuracy stays ≤ 0.9613 or inference cost exceeds label-linear's ~5k Ir/item, the hypothesis is wrong.
```

```python
"""label-tree: single CART decision tree for fixed-size byte items (ordinal only).
train.py card.json train.in train.key weights.bin
Deterministic: fixed seed, tie-breaking by feature then threshold.
Tunes max_depth on last 20% of train, then retrains on all train."""
import json, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 5


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


def best_split(X, y, feature_vals):
    """Return (feature, threshold, gain) or (None, None, 0) if no good split."""
    n = len(y)
    parent_gini = gini(y)
    best_gain = 0.0
    best_feat = None
    best_thresh = None
    for feat in range(X.shape[1]):
        vals = X[:, feat]
        # thresholds from min to max-1 (values are 0..16)
        for thresh in range(16):
            left = vals <= thresh
            right = ~left
            nl, nr = left.sum(), right.sum()
            if nl < MIN_SAMPLES_LEAF or nr < MIN_SAMPLES_LEAF:
                continue
            gain = parent_gini - (nl * gini(y[left]) + nr * gini(y[right])) / n
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh, best_gain


class Node:
    __slots__ = ("is_leaf", "class_label", "feature", "threshold", "left", "right")
    def __init__(self, is_leaf, class_label=0, feature=0, threshold=0, left=None, right=None):
        self.is_leaf = is_leaf
        self.class_label = class_label
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


def build_tree(X, y, depth, max_depth):
    # stopping criteria
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2 * MIN_SAMPLES_LEAF:
        # majority class (ties: smallest label)
        vals, counts = np.unique(y, return_counts=True)
        return Node(True, class_label=int(vals[np.argmax(counts)]))

    feat, thresh, gain = best_split(X, y, None)
    if feat is None or gain <= 1e-12:
        vals, counts = np.unique(y, return_counts=True)
        return Node(True, class_label=int(vals[np.argmax(counts)]))

    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
    return Node(False, feature=feat, threshold=thresh, left=left, right=right)


def flatten_tree(root):
    """Pre-order traversal to flat list of (is_leaf, class/feature, threshold, left_idx, right_idx)."""
    nodes = []
    def dfs(node):
        idx = len(nodes)
        if node.is_leaf:
            nodes.append((1, node.class_label, 0, -1, -1))
        else:
            nodes.append((0, node.feature, node.threshold, -1, -1))  # placeholders
            left_idx = dfs(node.left)
            right_idx = dfs(node.right)
            nodes[idx] = (0, node.feature, node.threshold, left_idx, right_idx)
        return idx
    dfs(root)
    return nodes


def evaluate(tree_root, X, y):
    pred = np.zeros(len(y), dtype=np.int64)
    for i, x in enumerate(X):
        node = tree_root
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        pred[i] = node.class_label
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    # tuning split: last 20%
    split = int(n * 0.8)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_depth = 1
    best_acc = 0.0
    for max_depth in range(1, 21):
        tree = build_tree(X_train, y_train, 0, max_depth)
        acc = evaluate(tree, X_tune, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_depth = max_depth

    # final training on all data
    final_tree = build_tree(X, y, 0, best_depth)
    flat = flatten_tree(final_tree)

    # write weights.bin
    # magic: 0x54524545 ("TREE")
    # num_nodes: int32
    # each node: uint8 is_leaf, uint8 class_or_feature, uint8 threshold, int32 left, int32 right
    out = struct.pack("<II", 0x54524545, len(flat))
    for is_leaf, a, b, left, right in flat:
        out += struct.pack("<BBBii", is_leaf, a, b, left, right)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels 0..16). Output: one byte class.
 * Weights format: magic 0x54524545, num_nodes (int32), then nodes:
 *   uint8 is_leaf, uint8 class_or_feature, uint8 threshold, int32 left, int32 right.
 * Tree is stored in pre-order; left/right are indices into node array.
 * Inference walks the tree for each input record. */
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
    uint8_t is_leaf;
    uint8_t class_or_feature;
    uint8_t threshold;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t num_nodes = (int32_t)(wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24);
    if (num_nodes <= 0) { fprintf(stderr, "bad node count\n"); return 3; }
    long expected = 8 + (long)num_nodes * 10;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 8);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int idx = 0;
        while (1) {
            Node *n = &nodes[idx];
            if (n->is_leaf) {
                unsigned char cls = n->class_or_feature;
                put_rec(stdout, &cls, 1);
                break;
            }
            uint8_t feat = n->class_or_feature;
            uint8_t thresh = n->threshold;
            if (xi[feat] <= thresh) idx = n->left;
            else idx = n->right;
        }
    }
    return 0;
}
```