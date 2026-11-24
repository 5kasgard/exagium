```hypothesis
label-tree: a single decision tree (max depth 6, min leaf 5) trained greedily with Gini impurity on raw 0..16 pixel values. Captures non-linear pixel interactions that linear models miss, while inference needs only ~6 comparisons per sample. Should beat label-linear's 0.9613 dev accuracy at lower CPU cost. Fails if dev accuracy <= 0.96 or Ir/item > 5000.
```
```python
"""label-tree: single decision tree for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seed), single thread, stdlib + numpy only."""
import json, struct, sys
import numpy as np

MAX_DEPTH = 6
MIN_LEAF = 5
SEED = 1
MAGIC = 0x54524531  # "TRE1" little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ("feat", "thresh", "left", "right", "cls", "idx")
    def __init__(self):
        self.feat = 255   # uint8, 255 = leaf
        self.thresh = 0.0 # float32
        self.left = -1    # int32
        self.right = -1   # int32
        self.cls = 0      # uint8
        self.idx = -1     # assigned during flattening

def gini(y):
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, feat_idx):
    """Return (threshold, gini_gain) for best split on one feature, or (None, 0) if no split."""
    vals = X[:, feat_idx]
    uniq = np.unique(vals)
    if len(uniq) <= 1:
        return None, 0.0
    # candidate thresholds midpoints between sorted unique values
    cand = (uniq[:-1] + uniq[1:]) * 0.5
    parent_gini = gini(y)
    best_gain = 0.0
    best_thr = None
    for thr in cand:
        mask = vals <= thr
        if mask.sum() < MIN_LEAF or (~mask).sum() < MIN_LEAF:
            continue
        n = len(y)
        gain = parent_gini - (mask.sum() * gini(y[mask]) + (~mask).sum() * gini(y[~mask])) / n
        if gain > best_gain:
            best_gain = gain
            best_thr = thr
    return best_thr, best_gain

def build_tree(X, y, depth, nodes, rng):
    node = Node()
    nodes.append(node)
    # stopping criteria
    if depth >= MAX_DEPTH or len(y) < 2 * MIN_LEAF or gini(y) == 0.0:
        node.feat = 255
        node.cls = np.bincount(y).argmax()
        return node
    # find best feature and threshold
    best_feat = -1
    best_thr = None
    best_gain = 0.0
    for f in range(X.shape[1]):
        thr, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feat = f
            best_thr = thr
    if best_gain <= 0.0:
        node.feat = 255
        node.cls = np.bincount(y).argmax()
        return node
    # split
    node.feat = best_feat
    node.thresh = float(best_thr)
    mask = X[:, best_feat] <= best_thr
    node.left = build_tree(X[mask], y[mask], depth + 1, nodes, rng)
    node.right = build_tree(X[~mask], y[~mask], depth + 1, nodes, rng)
    return node

def flatten_tree(root, nodes_flat):
    """DFS to assign indices and produce flat list."""
    stack = [root]
    while stack:
        node = stack.pop()
        node.idx = len(nodes_flat)
        nodes_flat.append(node)
        if node.feat != 255:
            # push right then left so left is processed first (lower index)
            stack.append(node.right)
            stack.append(node.left)
    # now fix left/right to indices
    for node in nodes_flat:
        if node.feat != 255:
            node.left = node.left.idx
            node.right = node.right.idx
        else:
            node.left = -1
            node.right = -1

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)
    # shuffle for deterministic tie-breaking in build_tree (though we use first-best)
    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    nodes = []
    root = build_tree(X, y, 0, nodes, rng)
    flat = []
    flatten_tree(root, flat)
    # serialize
    # header: magic, n_nodes, n_features, n_classes
    hdr = struct.pack("<4i", MAGIC, len(flat), d, C)
    body = bytearray()
    for node in flat:
        body.append(node.feat)                 # uint8
        body.extend(struct.pack("<f", node.thresh))  # float32
        body.extend(struct.pack("<i", node.left))    # int32
        body.extend(struct.pack("<i", node.right))   # int32
        body.append(node.cls)                  # uint8
        body.extend(b"\x00\x00\x00")           # padding to 16 bytes per node
    open(sys.argv[4], "wb").write(hdr + bytes(body))

if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (uint8 pixels). Output record: 1 byte class. */
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

typedef struct { uint8_t feat; float thresh; int32_t left, right; uint8_t cls; } Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x54524531) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_nodes = *(int32_t*)(wb+4);
    int32_t d = *(int32_t*)(wb+8);
    int32_t C = *(int32_t*)(wb+12);
    if (d != 64 || C != 10 || n_nodes <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + (size_t)n_nodes * 16;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }
    Node *nodes = (Node*)(wb + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int idx = 0;
        while (1) {
            Node *nd = &nodes[idx];
            if (nd->feat == 255) { put_rec(stdout, &nd->cls, 1); break; }
            uint8_t val = xi[nd->feat];
            idx = (val <= nd->thresh) ? nd->left : nd->right;
            if (idx < 0 || idx >= n_nodes) { fprintf(stderr, "bad child index\n"); return 3; }
        }
    }
    return 0;
}
```