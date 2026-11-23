```hypothesis
Family: label-tree (single CART decision tree with tuned max_depth).
Why: Optdigits has strong feature interactions (strokes) that a shallow tree captures; inference is just integer comparisons (O(depth)), far cheaper than MLP/GBT.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or instructions/item > 2× label-linear.
```
```python
"""label-tree: single CART decision tree for fixed-size byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread. Tunes max_depth on last 20% of train,
then retrains on all data. Exports tree as flat arrays for fast C inference.
Format: magic(0x45525444), d, num_nodes, max_depth; then per node (BFS order):
  uint8 is_leaf, uint8 feature, uint8 threshold, int32 left, int32 right, uint8 leaf_class.
"""
import json, math, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 10
MIN_GAIN = 1e-7
DEPTH_CANDIDATES = list(range(3, 18))  # 3..17


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


def best_split(X, y, rng):
    n, d = X.shape
    parent_gini = gini(y)
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1
    for j in range(d):
        vals = np.unique(X[:, j])
        if len(vals) <= 1:
            continue
        # thresholds are midpoints between successive values (0..16 -> 0.5..15.5)
        for v in vals[:-1]:
            thresh = v + 0.5
            left = X[:, j] <= thresh
            nL, nR = left.sum(), (~left).sum()
            if nL < MIN_SAMPLES_LEAF or nR < MIN_SAMPLES_LEAF:
                continue
            gain = parent_gini - (nL / n) * gini(y[left]) - (nR / n) * gini(y[~left])
            if gain > best_gain + 1e-12:
                best_gain, best_feat, best_thresh = gain, j, thresh
    if best_gain < MIN_GAIN:
        return -1, -1, -1.0
    return best_feat, best_thresh, best_gain


class Node:
    __slots__ = ("is_leaf", "feature", "threshold", "left", "right", "class_idx", "idx")
    def __init__(self):
        self.is_leaf = True
        self.feature = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.class_idx = 0
        self.idx = -1


def build_tree(X, y, depth, max_depth, rng):
    node = Node()
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2 * MIN_SAMPLES_LEAF:
        node.is_leaf = True
        node.class_idx = int(np.bincount(y).argmax())
        return node
    feat, thresh, gain = best_split(X, y, rng)
    if feat == -1:
        node.is_leaf = True
        node.class_idx = int(np.bincount(y).argmax())
        return node
    node.is_leaf = False
    node.feature = feat
    node.threshold = int(thresh)  # threshold is v+0.5, so int(thresh) = v; compare X[:,feat] <= v
    left_mask = X[:, feat] <= node.threshold
    node.left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, rng)
    node.right = build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth, rng)
    return node


def assign_indices(root):
    nodes = []
    q = [root]
    while q:
        n = q.pop(0)
        n.idx = len(nodes)
        nodes.append(n)
        if not n.is_leaf:
            q.append(n.left)
            q.append(n.right)
    return nodes


def predict_tree(nodes, x):
    i = 0
    while True:
        n = nodes[i]
        if n.is_leaf:
            return n.class_idx
        if x[n.feature] <= n.threshold:
            i = n.left.idx
        else:
            i = n.right.idx


def evaluate(nodes, X, y):
    correct = 0
    for i in range(len(X)):
        if predict_tree(nodes, X[i]) == y[i]:
            correct += 1
    return correct / len(X)


def export_tree(nodes, d, max_depth):
    num_nodes = len(nodes)
    # magic "DTRE" little-endian = 0x45525444
    hdr = struct.pack("<4i", 0x45525444, d, num_nodes, max_depth)
    body = bytearray()
    for n in nodes:
        body.append(1 if n.is_leaf else 0)
        body.append(n.feature if not n.is_leaf else 0)
        body.append(n.threshold if not n.is_leaf else 0)
        body.append(0)  # padding
        body.extend(struct.pack("<i", n.left.idx if not n.is_leaf else -1))
        body.extend(struct.pack("<i", n.right.idx if not n.is_leaf else -1))
        body.append(n.class_idx if n.is_leaf else 0)
        body.extend(b'\x00\x00\x00')  # padding to 16 bytes per node
    return hdr + bytes(body)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_depth = DEPTH_CANDIDATES[0]
    best_acc = -1.0
    for depth in DEPTH_CANDIDATES:
        root = build_tree(X_train, y_train, 0, depth, rng)
        nodes = assign_indices(root)
        acc = evaluate(nodes, X_tune, y_tune)
        if acc > best_acc:
            best_acc, best_depth = acc, depth

    # retrain on all data
    root = build_tree(X, y, 0, best_depth, rng)
    nodes = assign_indices(root)
    blob = export_tree(nodes, X.shape[1], best_depth)
    open(sys.argv[4], "wb").write(blob)


if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index.
 * Weights format: magic(0x45525444), d, num_nodes, max_depth; then per node (16 bytes):
 *   uint8 is_leaf, uint8 feature, uint8 threshold, uint8 pad,
 *   int32 left, int32 right, uint8 leaf_class, uint8 pad[3].
 */
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x45525444) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = wb[4] | wb[5]<<8 | wb[6]<<16 | wb[7]<<24;
    int num_nodes = wb[8] | wb[9]<<8 | wb[10]<<16 | wb[11]<<24;
    int max_depth = wb[12] | wb[13]<<8 | wb[14]<<16 | wb[15]<<24;
    if (d < 1 || d > 256 || num_nodes < 1 || max_depth < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 16L * num_nodes;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *is_leaf = malloc(num_nodes);
    unsigned char *feature = malloc(num_nodes);
    unsigned char *threshold = malloc(num_nodes);
    int32_t *left = malloc(num_nodes * sizeof(int32_t));
    int32_t *right = malloc(num_nodes * sizeof(int32_t));
    unsigned char *leaf_class = malloc(num_nodes);
    unsigned char *p = wb + 16;
    for (int i = 0; i < num_nodes; i++) {
        is_leaf[i] = p[0];
        feature[i] = p[1];
        threshold[i] = p[2];
        left[i] = p[4] | p[5]<<8 | p[6]<<16 | (int32_t)p[7]<<24;
        right[i] = p[8] | p[9]<<8 | p[10]<<16 | (int32_t)p[11]<<24;
        leaf_class[i] = p[12];
        p += 16;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (1) {
            if (is_leaf[node]) { put_rec(stdout, &leaf_class[node], 1); break; }
            if (xi[feature[node]] <= threshold[node]) node = left[node];
            else node = right[node];
        }
    }
    return 0;
}
```