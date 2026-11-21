```hypothesis
Family: label-tree (single CART decision tree). The 8x8 optdigits pixels are ordinal 0..16, ideal for axis-aligned splits. A tuned tree should exceed label-linear's 96.1% dev accuracy while keeping inference to a few dozen branches per item. If dev accuracy stays ≤ 96.1%, the family does not justify its addition.
```

```python
"""label-tree: CART decision tree for fixed-size byte items.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split (last 20% of train)
MAX_DEPTH_GRID = [5, 10, 15, 20, 25, 30]
MIN_LEAF_GRID = [1, 2, 5, 10]

MAGIC = 0x54524545  # 'TREE'

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

def build_tree(X, y, max_depth, min_leaf, depth=0, node_list=None):
    """Recursively build tree, appending nodes to node_list.
    Returns index of created node."""
    if node_list is None:
        node_list = []
    n = len(y)
    # Stopping criteria
    if depth >= max_depth or n < min_leaf or gini(y) == 0.0:
        # leaf: majority class (smallest class on tie)
        cls = int(np.bincount(y).argmax())
        node_list.append((-1, -1, 0, 0, cls))
        return len(node_list) - 1

    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    best_left_idx = best_right_idx = None

    # Try all features and thresholds (0..15 since values 0..16)
    for feat in range(X.shape[1]):
        vals = X[:, feat]
        unique_vals = np.unique(vals)
        if len(unique_vals) <= 1:
            continue
        for thresh in unique_vals[:-1]:  # split <= thresh vs > thresh
            left_mask = vals <= thresh
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            w_left = left_mask.sum() / n
            w_right = right_mask.sum() / n
            weighted_gini = w_left * gini(y[left_mask]) + w_right * gini(y[right_mask])
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feat = feat
                best_thresh = thresh
                best_left_idx = left_mask
                best_right_idx = right_mask

    if best_feat == -1:  # no valid split
        cls = int(np.bincount(y).argmax())
        node_list.append((-1, -1, 0, 0, cls))
        return len(node_list) - 1

    # Build children first (post-order so indices are stable)
    left_child = build_tree(X[best_left_idx], y[best_left_idx], max_depth, min_leaf, depth + 1, node_list)
    right_child = build_tree(X[best_right_idx], y[best_right_idx], max_depth, min_leaf, depth + 1, node_list)

    # Current node
    node_list.append((left_child, right_child, best_feat, int(best_thresh), 0))
    return len(node_list) - 1

def predict_tree(X, nodes):
    """Vectorized prediction for a batch (not used in train, but for tuning eval)."""
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        node_idx = 0
        while True:
            left, right, feat, thresh, cls = nodes[node_idx]
            if left == -1:  # leaf
                preds[i] = cls
                break
            if X[i, feat] <= thresh:
                node_idx = left
            else:
                node_idx = right
    return preds

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = len(y)

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    np.random.seed(SEED)
    best_acc = -1.0
    best_params = None
    best_nodes = None

    for max_depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_LEAF_GRID:
            nodes = []
            build_tree(X_train, y_train, max_depth, min_leaf, node_list=nodes)
            preds = predict_tree(X_tune, nodes)
            acc = (preds == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_nodes = nodes

    # Retrain on full train with best params
    max_depth, min_leaf = best_params
    final_nodes = []
    build_tree(X, y, max_depth, min_leaf, node_list=final_nodes)

    # Write weights.bin
    num_nodes = len(final_nodes)
    header = struct.pack("<Iiii", MAGIC, d, num_nodes, C)
    node_bytes = bytearray()
    for left, right, feat, thresh, cls in final_nodes:
        # left, right: int32; feat, thresh, cls: uint8; pad: uint8
        node_bytes += struct.pack("<iiBBBB", left, right, feat, thresh, cls, 0)
    with open(sys.argv[4], "wb") as f:
        f.write(header + node_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels). Output record: 1 byte (class 0..9). */
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
    int32_t left;
    int32_t right;
    uint8_t feature;
    uint8_t threshold;
    uint8_t cls;
    uint8_t pad;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wbuf[0] | wbuf[1] << 8 | wbuf[2] << 16 | (uint32_t)wbuf[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int32_t d = *(int32_t*)(wbuf + 4);
    int32_t num_nodes = *(int32_t*)(wbuf + 8);
    int32_t C = *(int32_t*)(wbuf + 12);
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights: invalid header\n"); return 3; }
    size_t expected = 16 + (size_t)num_nodes * sizeof(Node);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expected %zu\n", nw, expected); return 3; }
    Node *nodes = (Node*)(wbuf + 16);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node_idx = 0;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.left == -1) {  // leaf
                unsigned char out = nd.cls;
                put_rec(stdout, &out, 1);
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