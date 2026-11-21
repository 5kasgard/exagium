```hypothesis
Family: label-tree (single decision tree with ordinal splits). Why: Optdigits has 64 ordinal pixel features (0-16) where axis-aligned splits capture local stroke patterns better than a linear model. A depth-limited tree adds minimal inference cost (branch per level) while modeling non-linear interactions. Failure: If dev accuracy ≤ label-linear (0.9613) or cost per item exceeds label-mlp16 (8685 Ir/item).
```

```python
import json, struct, sys
import numpy as np

SEED = 1
MAX_DEPTH_CANDIDATES = [3, 4, 5, 6, 7]
MIN_LEAF_CANDIDATES = [1, 2, 5, 10]
TUNING_FRACTION = 0.2

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

class TreeBuilder:
    def __init__(self, max_depth, min_leaf):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.nodes = []  # each node: (feature, threshold, left, right)
        self.X = None
        self.y = None

    def build(self, X, y):
        self.X = X
        self.y = y
        self.nodes = []
        root = self._build_node(np.arange(len(y)), 0)
        assert root == 0
        return self.nodes

    def _build_node(self, idx, depth):
        node_idx = len(self.nodes)
        self.nodes.append(None)  # placeholder
        if depth >= self.max_depth or len(idx) < self.min_leaf:
            leaf_class = int(np.bincount(self.y[idx]).argmax())
            self.nodes[node_idx] = (-1, leaf_class, -1, -1)
            return node_idx

        parent_gini = gini(self.y[idx])
        best_gain = -1.0
        best_feature = -1
        best_threshold = -1
        best_left = None
        best_right = None

        for f in range(self.X.shape[1]):
            vals = self.X[idx, f]
            for t in range(16):  # thresholds 0..15
                left_mask = vals <= t
                right_mask = ~left_mask
                nl, nr = left_mask.sum(), right_mask.sum()
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                left_gini = gini(self.y[idx[left_mask]])
                right_gini = gini(self.y[idx[right_mask]])
                gain = parent_gini - (nl/len(idx))*left_gini - (nr/len(idx))*right_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = t
                    best_left = idx[left_mask]
                    best_right = idx[right_mask]

        if best_gain <= 0:
            leaf_class = int(np.bincount(self.y[idx]).argmax())
            self.nodes[node_idx] = (-1, leaf_class, -1, -1)
            return node_idx

        left_idx = self._build_node(best_left, depth + 1)
        right_idx = self._build_node(best_right, depth + 1)
        self.nodes[node_idx] = (best_feature, best_threshold, left_idx, right_idx)
        return node_idx

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    tune_size = int(n * TUNING_FRACTION)
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    best_acc = -1.0
    best_params = None
    for max_depth in MAX_DEPTH_CANDIDATES:
        for min_leaf in MIN_LEAF_CANDIDATES:
            builder = TreeBuilder(max_depth, min_leaf)
            nodes = builder.build(X[train_idx], y[train_idx])
            # evaluate on tuning split
            correct = 0
            for i in tune_idx:
                xi = X[i]
                node = 0
                while True:
                    feat, thresh, left, right = nodes[node]
                    if feat == -1:
                        pred = thresh
                        break
                    if xi[feat] <= thresh:
                        node = left
                    else:
                        node = right
                if pred == y[i]:
                    correct += 1
            acc = correct / len(tune_idx)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)

    # retrain on full data
    builder = TreeBuilder(*best_params)
    nodes = builder.build(X, y)

    # serialize: magic, num_nodes, then each node (int8, int8, int32, int32)
    out = struct.pack("<Ii", 0x54524545, len(nodes))  # 'TREE'
    for feat, thresh, left, right in nodes:
        out += struct.pack("<bbii", feat, thresh, left, right)
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
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

typedef struct { int8_t feature; int8_t threshold; int32_t left; int32_t right; } Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | (uint32_t)wb[3]<<24;
    int32_t num_nodes = *(int32_t*)(wb + 4);
    if (magic != 0x54524545 || num_nodes <= 0 || nw != 8 + (long)num_nodes * 12) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    Node *nodes = (Node*)(wb + 8);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int node = 0;
        while (1) {
            Node nd = nodes[node];
            if (nd.feature == -1) {
                unsigned char pred = (unsigned char)nd.threshold;
                put_rec(stdout, &pred, 1);
                break;
            }
            if (xi[nd.feature] <= nd.threshold) node = nd.left;
            else node = nd.right;
        }
    }
    return 0;
}
```