```hypothesis
Family: label-tree (single CART decision tree). Optdigits has 64 ordinal pixel features (0–16) that interact locally; a shallow tree captures piecewise-constant boundaries better than a linear model. Max depth is tuned on the last 20% of train.in (file order), then retrained on all data. If dev accuracy ≤ label-linear (0.9613) or inference cost > 2× label-linear, the family does not improve.
```
```python
"""label-tree: single CART decision tree for fixed-size byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Features used raw (ordinal). Max depth tuned on last 20% of train.in.
Weights format: magic 0x54524545, int32 num_nodes, then per node: int32 feature, float32 threshold,
int32 left, int32 right, int32 class (leaf: feature=-1, left=right=-1)."""
import json, math, struct, sys
import numpy as np

SEED = 1
MIN_GAIN = 1e-7
MIN_SAMPLES_LEAF = 1
MAX_DEPTH_CANDIDATES = list(range(1, 21))

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
    best_gain = -1.0
    best_feat = -1
    best_thresh = 0.0
    parent_gini = gini(y)
    for feat in range(d):
        vals = X[:, feat]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        threshs = (uniq[:-1] + uniq[1:]) * 0.5
        for thresh in threshs:
            left = vals <= thresh
            right = ~left
            if left.sum() < MIN_SAMPLES_LEAF or right.sum() < MIN_SAMPLES_LEAF:
                continue
            gain = parent_gini - (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / n
            if gain > best_gain + 1e-12:
                best_gain, best_feat, best_thresh = gain, feat, thresh
            elif abs(gain - best_gain) < 1e-12:
                # deterministic tie-break: smaller feature, then smaller threshold
                if feat < best_feat or (feat == best_feat and thresh < best_thresh):
                    best_feat, best_thresh = feat, thresh
    return best_feat, best_thresh, best_gain

def build_tree(X, y, depth, max_depth, rng, nodes):
    # nodes: lists feat, thresh, left, right, cls
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2 * MIN_SAMPLES_LEAF:
        cls = int(np.bincount(y).argmax())
        nodes['feat'].append(-1)
        nodes['thresh'].append(0.0)
        nodes['left'].append(-1)
        nodes['right'].append(-1)
        nodes['cls'].append(cls)
        return len(nodes['feat']) - 1
    feat, thresh, gain = best_split(X, y, rng)
    if feat == -1 or gain < MIN_GAIN:
        cls = int(np.bincount(y).argmax())
        nodes['feat'].append(-1)
        nodes['thresh'].append(0.0)
        nodes['left'].append(-1)
        nodes['right'].append(-1)
        nodes['cls'].append(cls)
        return len(nodes['feat']) - 1
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    node_idx = len(nodes['feat'])
    nodes['feat'].append(feat)
    nodes['thresh'].append(float(thresh))
    nodes['left'].append(-1)   # placeholder
    nodes['right'].append(-1)  # placeholder
    nodes['cls'].append(-1)
    left_idx = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, rng, nodes)
    right_idx = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, rng, nodes)
    nodes['left'][node_idx] = left_idx
    nodes['right'][node_idx] = right_idx
    return node_idx

def train_tree(X, y, max_depth, rng):
    nodes = {'feat': [], 'thresh': [], 'left': [], 'right': [], 'cls': []}
    build_tree(X, y, 0, max_depth, rng, nodes)
    return nodes

def evaluate(nodes, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        node = 0
        while nodes['feat'][node] != -1:
            feat = nodes['feat'][node]
            if X[i, feat] <= nodes['thresh'][node]:
                node = nodes['left'][node]
            else:
                node = nodes['right'][node]
        if nodes['cls'][node] == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_depth = 1
    best_acc = -1.0
    for d in MAX_DEPTH_CANDIDATES:
        nodes = train_tree(X_train, y_train, d, rng)
        acc = evaluate(nodes, X_tune, y_tune)
        if acc > best_acc + 1e-12:
            best_acc, best_depth = acc, d
    # retrain on all data
    nodes = train_tree(X, y, best_depth, rng)
    # write weights
    num_nodes = len(nodes['feat'])
    magic = 0x54524545  # 'TREE' little-endian
    out = struct.pack("<2i", magic, num_nodes)
    for i in range(num_nodes):
        out += struct.pack("<ifiii", nodes['feat'][i], nodes['thresh'][i],
                           nodes['left'][i], nodes['right'][i], nodes['cls'][i])
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=64). Output one byte: class index.
 * Weights: magic 0x54524545, int32 num_nodes, then per node:
 *   int32 feature; float32 threshold; int32 left; int32 right; int32 class.
 * Leaf: feature==-1, left==right==-1, class in 0..9. */
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
    int32_t cls;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t num_nodes = hdr[1];
    long expected = 8L + (long)num_nodes * 20L;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 8);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int node = 0;
        while (nodes[node].feature != -1) {
            int feat = nodes[node].feature;
            float val = (float)xi[feat];
            if (val <= nodes[node].threshold) node = nodes[node].left;
            else node = nodes[node].right;
        }
        unsigned char best = (unsigned char)nodes[node].cls;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```