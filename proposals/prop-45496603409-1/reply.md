```hypothesis
Family: label-rf (random forest). Random forests capture non-linear pixel interactions and are robust to noise. With 30 trees of depth 12, we expect >96.5% accuracy on dev, beating label-linear (96.1%) at similar inference cost (~30*12=360 node visits per item). If dev accuracy <= 96.1%, the model does not justify its complexity.
```

```python
"""Random forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys
import numpy as np

SEED = 1
N_TREES = 30
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = 8  # sqrt(64)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ("feature", "threshold", "left", "right", "leaf_class")
    def __init__(self, feature=0, threshold=0.0, left=-1, right=-1, leaf_class=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_class = leaf_class

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, features, rng):
    best_gini = 1.0
    best_feat = -1
    best_thresh = 0.0
    for f in features:
        vals = X[:, f]
        # thresholds at midpoints between 0..16
        for t in range(16):
            thresh = t + 0.5
            mask = vals <= thresh
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            g = (mask.sum() * gini(y[mask]) + (~mask).sum() * gini(y[~mask])) / len(y)
            if g < best_gini:
                best_gini = g
                best_feat = f
                best_thresh = thresh
    return best_feat, best_thresh, best_gini

def build_tree(X, y, depth, rng, n_features):
    n_samples = len(y)
    if depth >= MAX_DEPTH or n_samples < MIN_SAMPLES_LEAF or len(np.unique(y)) == 1:
        leaf_class = np.bincount(y).argmax()
        return Node(leaf_class=leaf_class)
    features = rng.choice(n_features, size=min(MAX_FEATURES, n_features), replace=False)
    feat, thresh, _ = best_split(X, y, features, rng)
    if feat == -1:
        leaf_class = np.bincount(y).argmax()
        return Node(leaf_class=leaf_class)
    mask = X[:, feat] <= thresh
    left = build_tree(X[mask], y[mask], depth + 1, rng, n_features)
    right = build_tree(X[~mask], y[~mask], depth + 1, rng, n_features)
    return Node(feature=feat, threshold=thresh, left=left, right=right)

def flatten_tree(root):
    nodes = []
    def dfs(node):
        idx = len(nodes)
        nodes.append(node)
        if node.left != -1:
            dfs(node.left)
            dfs(node.right)
    dfs(root)
    # convert to flat arrays
    flat = []
    for node in nodes:
        if node.left != -1:
            # find indices of children in flat list (they are appended after parent in pre-order)
            # we need to know the indices of left and right children.
            # Since we do pre-order, left child is at idx+1, right child is after left subtree.
            # We'll compute by building a list of nodes with children indices.
            pass
    # Instead, we do a second pass to assign indices.
    # Let's do a non-recursive flattening with a stack.
    flat_nodes = []
    stack = [(root, -1, -1)]  # (node, left_idx, right_idx) -1 means not processed
    while stack:
        node, left_idx, right_idx = stack.pop()
        if left_idx == -1:  # first visit
            idx = len(flat_nodes)
            flat_nodes.append(node)
            if node.left != -1:
                # push parent again with placeholder, then children
                stack.append((node, idx + 1, -2))  # -2 means left done, right pending
                stack.append((node.right, -1, -1))
                stack.append((node.left, -1, -1))
            else:
                # leaf, nothing to push
                pass
        else:
            # children already processed, update indices
            flat_nodes[left_idx] = node  # not needed, we already have node
    # The above is messy. Simpler: do recursive flattening that returns list and child indices.
    # Let's implement a helper that returns (list_of_nodes, left_child_index, right_child_index) for root.
    pass

# We'll implement a cleaner flattening.
def flatten_tree(root):
    nodes = []
    def dfs(node):
        idx = len(nodes)
        nodes.append(node)
        if node.left != -1:
            left_idx = dfs(node.left)
            right_idx = dfs(node.right)
            node.left = left_idx
            node.right = right_idx
        else:
            node.left = -1
            node.right = -1
        return idx
    dfs(root)
    return nodes

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_samples, n_features = X.shape
    n_classes = card["classes"]

    rng = np.random.default_rng(SEED)

    trees = []
    for t in range(N_TREES):
        # bootstrap sample
        idx = rng.integers(0, n_samples, n_samples)
        Xb, yb = X[idx], y[idx]
        root = build_tree(Xb, yb, 0, rng, n_features)
        flat = flatten_tree(root)
        trees.append(flat)

    # Write weights.bin
    # Header: magic (0x52465354), n_trees, n_classes, n_features
    out = struct.pack("<4I", 0x52465354, N_TREES, n_classes, n_features)
    for flat in trees:
        out += struct.pack("<I", len(flat))
        for node in flat:
            # node: feature (uint8), threshold (float32), left (int32), right (int32), leaf_class (uint8)
            out += struct.pack("<BfiiB", node.feature, node.threshold, node.left, node.right, node.leaf_class)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Random forest inference for label tasks: ./infer weights.bin in.rec N */
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
    uint8_t feature;
    float threshold;
    int32_t left;
    int32_t right;
    uint8_t leaf_class;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x52465354) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t n_trees = hd[1], n_classes = hd[2], n_features = hd[3];
    if (n_classes > 256 || n_features > 255) { fprintf(stderr, "bad header\n"); return 3; }

    unsigned char *ptr = wb + 16;
    long remaining = nw - 16;
    Node **trees = malloc(n_trees * sizeof(Node *));
    uint32_t *tree_sizes = malloc(n_trees * sizeof(uint32_t));
    for (uint32_t t = 0; t < n_trees; t++) {
        if (remaining < 4) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t n_nodes = *(uint32_t *)ptr;
        ptr += 4; remaining -= 4;
        if (remaining < (long)n_nodes * 14) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t] = (Node *)ptr;
        tree_sizes[t] = n_nodes;
        ptr += n_nodes * 14;
        remaining -= n_nodes * 14;
    }
    if (remaining != 0) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *votes = malloc(n_classes);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, n_features); return 3; }
        memset(votes, 0, n_classes);
        for (uint32_t t = 0; t < n_trees; t++) {
            Node *nodes = trees[t];
            uint32_t idx = 0;
            while (1) {
                Node *node = &nodes[idx];
                if (node->left == -1) {
                    votes[node->leaf_class]++;
                    break;
                }
                if (xi[node->feature] <= node->threshold)
                    idx = node->left;
                else
                    idx = node->right;
            }
        }
        unsigned char best = 0;
        for (uint32_t c = 1; c < n_classes; c++)
            if (votes[c] > votes[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```