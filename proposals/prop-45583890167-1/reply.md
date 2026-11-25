```hypothesis
Family: label-tree. A shallow decision tree (max depth 10, min leaf 5) captures non-linear pixel interactions in the 8x8 optdigits data. It should beat the linear model (0.9613) because digit shapes require spatial reasoning, while inference stays fast (only a few integer comparisons per item). If accuracy stays ≤0.9613, the task is largely linearly separable in the raw pixel space.
```
```python
"""label-tree: single decision tree for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Tunes max_depth and min_samples_leaf on last 20% of train,
then retrains on all data. Exports a flat binary tree: magic 0x54524545, n_nodes, n_features, n_classes,
then per node (12 bytes): uint8 is_leaf, uint8 class_or_feature, uint8 threshold, uint8 pad,
uint32 left, uint32 right (left/right=0 for leaves)."""
import json, struct, sys
import numpy as np

SEED = 1
DEPTH_GRID = [5, 8, 10, 12]
LEAF_GRID = [1, 5, 10]

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
    counts = np.bincount(y, minlength=10)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def most_common(y):
    return int(np.argmax(np.bincount(y, minlength=10)))

class TreeBuilder:
    def __init__(self, max_depth, min_leaf, n_features=64, n_classes=10):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_features = n_features
        self.n_classes = n_classes
        self.nodes = []  # each node will be a dict

    def build(self, X, y, depth=0):
        node_idx = len(self.nodes)
        self.nodes.append(None)  # placeholder

        # stopping criteria
        if depth >= self.max_depth or len(y) <= self.min_leaf or len(np.unique(y)) == 1:
            leaf_class = most_common(y)
            self.nodes[node_idx] = (0, leaf_class, 0, 0, 0, 0)  # is_leaf=0? we use 0 for leaf, 1 for internal
            return node_idx

        best_gain = -1.0
        best_f = -1
        best_t = -1
        parent_gini = gini(y)
        n = len(y)

        for f in range(self.n_features):
            # values are 0..16, thresholds 0..15
            for t in range(16):
                left_mask = X[:, f] <= t
                right_mask = ~left_mask
                nl = left_mask.sum()
                nr = right_mask.sum()
                if nl == 0 or nr == 0:
                    continue
                gain = parent_gini - (nl * gini(y[left_mask]) + nr * gini(y[right_mask])) / n
                if gain > best_gain or (gain == best_gain and (f < best_f or (f == best_f and t < best_t))):
                    best_gain = gain
                    best_f = f
                    best_t = t

        if best_gain <= 0:
            leaf_class = most_common(y)
            self.nodes[node_idx] = (0, leaf_class, 0, 0, 0, 0)
            return node_idx

        left_mask = X[:, best_f] <= best_t
        right_mask = ~left_mask
        left_idx = self.build(X[left_mask], y[left_mask], depth + 1)
        right_idx = self.build(X[right_mask], y[right_mask], depth + 1)

        # is_leaf=1 for internal
        self.nodes[node_idx] = (1, best_f, best_t, 0, left_idx, right_idx)
        return node_idx

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]

    best_acc = -1.0
    best_params = None
    best_tree = None

    for depth in DEPTH_GRID:
        for leaf in LEAF_GRID:
            builder = TreeBuilder(depth, leaf)
            builder.build(train_X, train_y)
            # evaluate on tune
            correct = 0
            for xi, yi in zip(tune_X, tune_y):
                node = 0
                while True:
                    is_leaf, a, b, _, left, right = builder.nodes[node]
                    if is_leaf == 0:
                        pred = a
                        break
                    if xi[a] <= b:
                        node = left
                    else:
                        node = right
                if pred == yi:
                    correct += 1
            acc = correct / len(tune_y)
            if acc > best_acc or (acc == best_acc and (depth < best_params[0] or (depth == best_params[0] and leaf < best_params[1]))):
                best_acc = acc
                best_params = (depth, leaf)
                best_tree = builder.nodes

    # retrain on all data with best params
    builder = TreeBuilder(best_params[0], best_params[1])
    builder.build(X, y)
    nodes = builder.nodes

    # serialize
    magic = 0x54524545  # 'TREE' little-endian
    n_nodes = len(nodes)
    n_features = card["input"]["bytes"]
    n_classes = card["classes"]
    header = struct.pack("<4i", magic, n_nodes, n_features, n_classes)
    body = bytearray()
    for node in nodes:
        is_leaf, a, b, pad, left, right = node
        body += struct.pack("<BBBBI I", is_leaf, a, b, pad, left, right)  # 1+1+1+1+4+4=12
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d from weights). Output: one byte class per record. */
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
    uint8_t pad;
    uint32_t left;
    uint32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    int n_nodes = hd[1], d = hd[2], C = hd[3];
    if (n_nodes <= 0 || d != 64 || C != 10) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + (long)n_nodes * 12;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (1) {
            Node nd = nodes[node];
            if (nd.is_leaf == 0) {
                put_rec(stdout, &nd.class_or_feature, 1);
                break;
            }
            if (xi[nd.class_or_feature] <= nd.threshold)
                node = nd.left;
            else
                node = nd.right;
        }
    }
    return 0;
}
```