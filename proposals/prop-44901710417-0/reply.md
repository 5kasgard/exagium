```hypothesis
Family: label-tree (single deterministic CART on raw byte features).
Why: The 14 input bytes are already quantized (ordinals as 255 quantile bins, categoricals as category indices). A shallow CART (max_depth=5, min_leaf=10) splits directly on these bytes, avoiding the one-hot expansion of label-linear. Inference is a handful of integer comparisons per record, so it should be faster than label-linear's dense dot products. If dev accuracy does not exceed label-linear's 0.8524 ± 0.0044, the hypothesis fails.
```

```python
"""label-tree: deterministic CART on raw byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Weights format: magic 0x54524545, int32 d, int32 num_nodes, then nodes array of
  dtype=[('feat','u1'),('thresh','u1'),('left','i4'),('right','i4'),('leaf','u1')].
Leaf nodes have left==right==-1 and leaf=class (0 or 1)."""
import json, struct, sys
import numpy as np

SEED = 1
MAX_DEPTH = 5
MIN_LEAF = 10

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(y):
    if len(y) == 0: return 0.0
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, feat_idx, rng):
    """Return (threshold, gain, left_idx, right_idx) for best split on one feature."""
    vals = X[:, feat_idx]
    uniq = np.unique(vals)
    if len(uniq) <= 1:
        return None, -1.0, None, None
    best_gain, best_thresh = -1.0, None
    parent_gini = gini(y)
    n = len(y)
    for t in uniq[:-1]:  # split: <= t vs > t
        left = y[vals <= t]
        right = y[vals > t]
        if len(left) < MIN_LEAF or len(right) < MIN_LEAF:
            continue
        gain = parent_gini - (len(left)/n)*gini(left) - (len(right)/n)*gini(right)
        if gain > best_gain + 1e-12:
            best_gain, best_thresh = gain, t
        elif abs(gain - best_gain) < 1e-12 and t < best_thresh:
            best_thresh = t
    if best_thresh is None:
        return None, -1.0, None, None
    left_idx = vals <= best_thresh
    right_idx = vals > best_thresh
    return best_thresh, best_gain, left_idx, right_idx

def build_tree(X, y, depth, rng):
    """Returns list of nodes in preorder; each node is dict with keys feat, thresh, left, right, leaf."""
    # stopping criteria
    if depth >= MAX_DEPTH or len(y) < 2*MIN_LEAF or gini(y) == 0.0:
        leaf_class = int(np.bincount(y).argmax())
        return [{'feat': 0, 'thresh': 0, 'left': -1, 'right': -1, 'leaf': leaf_class}]
    # find best feature and threshold
    best_feat, best_thresh, best_left_idx, best_right_idx = -1, None, None, None
    best_gain = -1.0
    for f in range(X.shape[1]):
        thresh, gain, left_idx, right_idx = best_split(X, y, f, rng)
        if gain > best_gain + 1e-12:
            best_gain, best_feat, best_thresh = gain, f, thresh
            best_left_idx, best_right_idx = left_idx, right_idx
        elif abs(gain - best_gain) < 1e-12 and f < best_feat:
            best_feat, best_thresh = f, thresh
            best_left_idx, best_right_idx = left_idx, right_idx
    if best_gain <= 0.0:
        leaf_class = int(np.bincount(y).argmax())
        return [{'feat': 0, 'thresh': 0, 'left': -1, 'right': -1, 'leaf': leaf_class}]
    # recurse
    left_nodes = build_tree(X[best_left_idx], y[best_left_idx], depth+1, rng)
    right_nodes = build_tree(X[best_right_idx], y[best_right_idx], depth+1, rng)
    # preorder: root, then left subtree, then right subtree
    root = {'feat': best_feat, 'thresh': int(best_thresh), 'left': 1, 'right': 1 + len(left_nodes), 'leaf': 0}
    return [root] + left_nodes + right_nodes

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # tuning split: last 20% in file order
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # grid search over (max_depth, min_leaf) on validation set
    best_acc, best_params = -1.0, (MAX_DEPTH, MIN_LEAF)
    for md in [3, 4, 5, 6]:
        for ml in [5, 10, 20]:
            rng = np.random.default_rng(SEED)
            nodes = build_tree(X_train, y_train, 0, rng)
            # evaluate on validation
            correct = 0
            for xi, yi in zip(X_val, y_val):
                idx = 0
                while nodes[idx]['left'] != -1:
                    if xi[nodes[idx]['feat']] <= nodes[idx]['thresh']:
                        idx = nodes[idx]['left']
                    else:
                        idx = nodes[idx]['right']
                if nodes[idx]['leaf'] == yi:
                    correct += 1
            acc = correct / len(y_val)
            if acc > best_acc:
                best_acc, best_params = acc, (md, ml)

    # retrain on full data with best params
    global MAX_DEPTH, MIN_LEAF
    MAX_DEPTH, MIN_LEAF = best_params
    rng = np.random.default_rng(SEED)
    nodes = build_tree(X, y, 0, rng)

    # write weights.bin
    magic = 0x54524545  # "TREE"
    d = X.shape[1]
    num_nodes = len(nodes)
    dt = np.dtype([('feat','u1'),('thresh','u1'),('left','i4'),('right','i4'),('leaf','u1')])
    arr = np.zeros(num_nodes, dtype=dt)
    for i, nd in enumerate(nodes):
        arr[i] = (nd['feat'], nd['thresh'], nd['left'], nd['right'], nd['leaf'])
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<Iii", magic, d, num_nodes))
        f.write(arr.tobytes())

if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output record: 1 byte class (0 or 1). */
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
    uint8_t feat;
    uint8_t thresh;
    int32_t left;
    int32_t right;
    uint8_t leaf;
} Node;
#pragma pack(pop)

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t*)wb;
    int32_t d = *(int32_t*)(wb + 4);
    int32_t num_nodes = *(int32_t*)(wb + 8);
    if (magic != 0x54524545 || d != 14 || num_nodes <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 12 + (long)num_nodes * sizeof(Node);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node*)(wb + 12);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int idx = 0;
        while (nodes[idx].left != -1) {
            if (xi[nodes[idx].feat] <= nodes[idx].thresh)
                idx = nodes[idx].left;
            else
                idx = nodes[idx].right;
        }
        unsigned char out = nodes[idx].leaf;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```