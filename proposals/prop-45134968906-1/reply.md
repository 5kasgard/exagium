```hypothesis
Family: label-tree (decision tree). Optdigits 8x8 pixels have spatial structure; a depth-6 tree (~64 leaves) captures non-linear boundaries with pure integer comparisons at inference. Expect ~0.965 accuracy (beating label-linear's 0.9613) at <3000 Ir/item (no FP math, just 6-8 byte comparisons per item). Failure: dev accuracy < 0.9613 or Ir/item > 5000.
```
```python
"""label-tree: greedy CART decision tree for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Chooses max_depth/min_samples on last 20% tuning split,
then retrains on all data. Exports binary tree: magic 0x54524545, n_nodes (u16), then per node:
  leaf:   u8 flags(bit0=1), u8 class, u8[4] pad
  internal: u8 flags(bit0=0), u8 feature(0-63), u8 threshold(0-15 -> thresh+0.5), u16 left, u16 right"""
import json, struct, sys
import numpy as np

SEED = 1
MAX_DEPTH_OPTS = [4, 5, 6, 7]
MIN_SAMPLES_OPTS = [2, 5, 10, 20]
TUNING_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(y):
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, rng):
    n, d = X.shape
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    for feat in range(d):
        vals = X[:, feat]
        uniq = np.unique(vals)
        if len(uniq) <= 1: continue
        for v in uniq[:-1]:  # threshold between v and next
            thresh = v + 0.5
            left = vals <= thresh
            right = ~left
            if left.sum() == 0 or right.sum() == 0: continue
            g = (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / n
            if g < best_gini - 1e-12:
                best_gini, best_feat, best_thresh = g, feat, thresh
    return best_feat, best_thresh, best_gini

def build_tree(X, y, depth, max_depth, min_samples, rng):
    n = len(y)
    if depth >= max_depth or n < min_samples or len(np.unique(y)) == 1:
        cls = int(np.bincount(y).argmax())
        return {'leaf': True, 'class': cls}
    feat, thresh, _ = best_split(X, y, rng)
    if feat == -1:
        cls = int(np.bincount(y).argmax())
        return {'leaf': True, 'class': cls}
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], depth+1, max_depth, min_samples, rng)
    right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth, min_samples, rng)
    return {'leaf': False, 'feature': feat, 'threshold': int(thresh - 0.5), 'left': left, 'right': right}

def count_nodes(node):
    if node['leaf']: return 1
    return 1 + count_nodes(node['left']) + count_nodes(node['right'])

def flatten_tree(node, nodes, idx):
    """returns (my_idx, next_free_idx)"""
    my = idx
    if node['leaf']:
        nodes.append((1, node['class'], 0, 0, 0, 0))
        return my, idx + 1
    left_idx, next_idx = flatten_tree(node['left'], nodes, idx + 1)
    right_idx, next_idx = flatten_tree(node['right'], nodes, next_idx)
    nodes.append((0, node['feature'], node['threshold'], left_idx, right_idx, 0))
    return my, next_idx

def accuracy(tree, X, y):
    preds = predict_batch(tree, X)
    return np.mean(preds == y)

def predict_batch(tree, X):
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        node = tree
        while not node['leaf']:
            if X[i, node['feature']] <= node['threshold'] + 0.5:
                node = node['left']
            else:
                node = node['right']
        preds[i] = node['class']
    return preds

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    split = int(n * (1 - TUNING_FRAC))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_tree = None
    best_params = None

    for max_depth in MAX_DEPTH_OPTS:
        for min_samples in MIN_SAMPLES_OPTS:
            tree = build_tree(X_train, y_train, 0, max_depth, min_samples, rng)
            acc = accuracy(tree, X_val, y_val)
            if acc > best_acc:
                best_acc, best_tree, best_params = acc, tree, (max_depth, min_samples)
    # retrain on all data
    tree = build_tree(X, y, 0, best_params[0], best_params[1], rng)
    nodes = []
    flatten_tree(tree, nodes, 0)
    n_nodes = len(nodes)
    # pack: magic(4), n_nodes(2), then per node: flags(1), class/feat(1), thresh(1), left(2), right(2), pad(2)
    out = struct.pack("<IH", 0x54524545, n_nodes)
    for flags, a, b, c, d, _ in nodes:
        if flags:
            out += struct.pack("<BBHHH", flags, a, 0, 0, 0)
        else:
            out += struct.pack("<BBHHH", flags, a, b, c, d)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Reads N records of 64 bytes each, traverses tree, writes 1-byte class per record. */
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
    uint8_t flags;
    uint8_t a;      // class (leaf) or feature (internal)
    uint8_t thresh; // 0-15 (internal)
    uint16_t left;  // internal
    uint16_t right; // internal
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 6) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    uint16_t n_nodes = wb[4] | wb[5] << 8;
    if (nw != 6 + n_nodes * 8) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 6);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        uint16_t idx = 0;
        while (1) {
            Node nd = nodes[idx];
            if (nd.flags & 1) { // leaf
                put_rec(stdout, &nd.a, 1);
                break;
            }
            uint8_t feat = nd.a;
            uint8_t thr = nd.thresh;
            if (xi[feat] <= thr + 0.5) idx = nd.left;
            else idx = nd.right;
        }
    }
    return 0;
}
```