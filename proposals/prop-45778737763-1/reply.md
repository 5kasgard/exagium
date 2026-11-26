```hypothesis
Family: label-tree (single CART decision tree with tuned depth).  
Why: 8x8 pixels are spatially structured; axis-aligned splits on pixel counts capture local ink patterns. A single tree is faster than GBRT and more expressive than linear.  
Failure signal: dev accuracy ≤ label-linear (0.9613) or inference > 2× label-linear cycles.
```

```python
import json, struct, sys, math
import numpy as np

SEED = 1
# Hyper-parameter grid searched on the tuning split (last 20% of train)
MAX_DEPTH_GRID = [5, 8, 10, 12, 15]
MIN_LEAF_GRID  = [1, 2, 5, 10]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

class Node:
    __slots__ = ("feat", "thresh", "left", "right", "cls")
    def __init__(self, feat=-1, thresh=0, left=None, right=None, cls=-1):
        self.feat = feat      # feature index (0..63) or -1 for leaf
        self.thresh = thresh  # threshold (0..255)
        self.left = left      # left child Node
        self.right = right    # right child Node
        self.cls = cls        # class label for leaf, -1 for internal

def gini(y):
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / len(y)
    return 1.0 - np.sum(p*p)

def best_split(X, y, feat_idx):
    """Return (threshold, gini_gain) for a single feature."""
    vals = X[:, feat_idx]
    uniq = np.unique(vals)
    if len(uniq) <= 1:
        return None, -1.0
    parent = gini(y)
    best_gain, best_thr = -1.0, uniq[0]
    for thr in uniq[:-1]:
        mask = vals <= thr
        if mask.sum() == 0 or (~mask).sum() == 0:
            continue
        gain = parent - (mask.sum()*gini(y[mask]) + (~mask).sum()*gini(y[~mask])) / len(y)
        if gain > best_gain:
            best_gain, best_thr = gain, thr
    return best_thr, best_gain

def build_tree(X, y, depth, max_depth, min_leaf):
    # Stopping criteria
    if depth >= max_depth or len(y) <= min_leaf or gini(y) == 0.0:
        cls = int(np.bincount(y).argmax())
        return Node(cls=cls)
    # Find best feature and threshold
    best_feat, best_thr, best_gain = -1, 0, -1.0
    for f in range(X.shape[1]):
        thr, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain, best_feat, best_thr = gain, f, thr
    if best_gain <= 0:
        cls = int(np.bincount(y).argmax())
        return Node(cls=cls)
    mask = X[:, best_feat] <= best_thr
    left  = build_tree(X[mask], y[mask], depth+1, max_depth, min_leaf)
    right = build_tree(X[~mask], y[~mask], depth+1, max_depth, min_leaf)
    return Node(feat=best_feat, thresh=best_thr, left=left, right=right)

def flatten_tree(root):
    """BFS flatten to arrays for serialization."""
    nodes = []
    queue = [root]
    for nd in queue:
        if nd.feat == -1:
            nodes.append((nd.feat, nd.thresh, -1, -1, nd.cls))
        else:
            left_idx = len(queue)
            queue.append(nd.left)
            right_idx = len(queue)
            queue.append(nd.right)
            nodes.append((nd.feat, nd.thresh, left_idx, right_idx, -1))
    return nodes

def eval_tree(nodes, x):
    """Iterative traversal on flattened nodes."""
    idx = 0
    while True:
        feat, thr, left, right, cls = nodes[idx]
        if feat == -1:
            return cls
        if x[feat] <= thr:
            idx = left
        else:
            idx = right

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    best_acc, best_md, best_ml = -1, 8, 1
    for md in MAX_DEPTH_GRID:
        for ml in MIN_LEAF_GRID:
            root = build_tree(X_tr, y_tr, 0, md, ml)
            nodes = flatten_tree(root)
            pred = np.array([eval_tree(nodes, x) for x in X_va])
            acc = (pred == y_va).mean()
            if acc > best_acc:
                best_acc, best_md, best_ml = acc, md, ml

    # Retrain on full data
    root = build_tree(X, y, 0, best_md, best_ml)
    nodes = flatten_tree(root)

    # Serialize: magic, version, num_nodes, num_classes, then per node: int16 feat, uint8 thresh, int32 left, int32 right, int8 cls
    magic = 0x54524545  # "TREE"
    version = 1
    num_nodes = len(nodes)
    num_classes = card["classes"]
    out = struct.pack("<IIII", magic, version, num_nodes, num_classes)
    for feat, thr, left, right, cls in nodes:
        out += struct.pack("<hBii b", feat, thr, left, right, cls)
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t*)wb;
    if (hdr[0] != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t version = hdr[1], num_nodes = hdr[2], num_classes = hdr[3];
    if (version != 1 || num_classes > 256) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + num_nodes * (2 + 1 + 4 + 4 + 1);  // hBii b (with padding)
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    // Node arrays
    int16_t  *feat    = malloc(num_nodes * sizeof(int16_t));
    uint8_t  *thresh  = malloc(num_nodes * sizeof(uint8_t));
    int32_t  *left    = malloc(num_nodes * sizeof(int32_t));
    int32_t  *right   = malloc(num_nodes * sizeof(int32_t));
    int8_t   *cls     = malloc(num_nodes * sizeof(int8_t));
    unsigned char *p = wb + 16;
    for (uint32_t i = 0; i < num_nodes; i++) {
        feat[i]   = *(int16_t*)p; p += 2;
        thresh[i] = *p; p += 1;
        p += 1;  // padding to align int32
        left[i]   = *(int32_t*)p; p += 4;
        right[i]  = *(int32_t*)p; p += 4;
        cls[i]    = *(int8_t*)p; p += 1;
        p += 3;  // padding to 16-byte per node
    }
    free(wb);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        uint32_t idx = 0;
        while (1) {
            if (feat[idx] == -1) { put_rec(stdout, (unsigned char*)&cls[idx], 1); break; }
            if (xi[feat[idx]] <= thresh[idx]) idx = left[idx];
            else idx = right[idx];
        }
    }
    return 0;
}
```