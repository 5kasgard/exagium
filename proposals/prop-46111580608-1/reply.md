```hypothesis
A single deterministic CART decision tree with tuned max depth captures non-linear pixel interactions in 8x8 digits.
It should beat label-linear (0.9613) because trees model feature conjunctions (e.g., strokes) that linear models miss.
Inference is a few integer comparisons per item, so CPU cost should stay near label-linear.
If dev accuracy ≤ 0.965, the tree does not justify its complexity over the linear baseline.
```
```python
"""Deterministic CART decision tree for fixed-size byte items (optdigits).
python3 -s -P train.py card.json train.in train.key weights.bin
Tunes max_depth on last 20% of train, then trains on all train.
Weights format: magic 0x44545245 ('DTRE'), uint32 node_count, then node_count * 12 bytes:
  uint8 feature, uint8 threshold, uint8 class, uint8 pad, int32 left, int32 right.
Leaf: left = -1, right = -1, class = majority class (smallest on tie)."""
import json, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
DEPTH_CANDIDATES = range(3, 11)  # 3..10


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)


def best_split(X, y, n_classes):
    n_samples, n_features = X.shape
    best_gini = 1.0
    best_feature = -1
    best_threshold = -1
    for f in range(n_features):
        counts = np.zeros((256, n_classes), dtype=np.int64)
        for i in range(n_samples):
            v = X[i, f]
            counts[v, y[i]] += 1
        cum = np.cumsum(counts, axis=0)
        total = cum[-1]
        for t in range(255):
            left_counts = cum[t]
            right_counts = total - left_counts
            left_total = left_counts.sum()
            right_total = right_counts.sum()
            if left_total < MIN_SAMPLES_LEAF or right_total < MIN_SAMPLES_LEAF:
                continue
            gini_left = gini(left_counts)
            gini_right = gini(right_counts)
            weighted = (left_total * gini_left + right_total * gini_right) / n_samples
            if weighted < best_gini:
                best_gini = weighted
                best_feature = f
                best_threshold = t
    return best_feature, best_threshold, best_gini


def majority_class(y):
    return int(np.argmax(np.bincount(y)))


class TreeBuilder:
    def __init__(self, max_depth, n_classes):
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.nodes = []  # list of (feature, threshold, left, right, class)

    def build(self, X, y, depth=0):
        n_samples = X.shape[0]
        # stopping criteria
        if depth >= self.max_depth or n_samples < MIN_SAMPLES_SPLIT or len(np.unique(y)) == 1:
            cls = majority_class(y)
            self.nodes.append((255, 0, -1, -1, cls))  # feature=255 marks leaf
            return len(self.nodes) - 1
        feat, thresh, _ = best_split(X, y, self.n_classes)
        if feat == -1:  # no valid split
            cls = majority_class(y)
            self.nodes.append((255, 0, -1, -1, cls))
            return len(self.nodes) - 1
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        left_idx = self.build(X[left_mask], y[left_mask], depth + 1)
        right_idx = self.build(X[right_mask], y[right_mask], depth + 1)
        self.nodes.append((feat, thresh, left_idx, right_idx, 0))
        return len(self.nodes) - 1


def evaluate(nodes, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        node_idx = 0
        while True:
            feat, thresh, left, right, cls = nodes[node_idx]
            if left == -1:
                if cls == yi:
                    correct += 1
                break
            if xi[feat] <= thresh:
                node_idx = left
            else:
                node_idx = right
    return correct / len(y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    best_depth = DEPTH_CANDIDATES[0]
    best_acc = -1.0
    for d in DEPTH_CANDIDATES:
        builder = TreeBuilder(d, card["classes"])
        builder.build(X[train_idx], y[train_idx])
        acc = evaluate(builder.nodes, X[tune_idx], y[tune_idx])
        if acc > best_acc:
            best_acc = acc
            best_depth = d

    # final training on all data
    builder = TreeBuilder(best_depth, card["classes"])
    builder.build(X, y)
    nodes = builder.nodes

    # write weights.bin
    magic = 0x44545245  # 'DTRE'
    node_count = len(nodes)
    out = struct.pack("<II", magic, node_count)
    for feat, thresh, left, right, cls in nodes:
        out += struct.pack("<BBBBIii", feat, thresh, cls, 0, left, right)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
```
```c
/* Decision tree inference for label-dtree family.
 * ./infer weights.bin in.rec N
 * Each input record: 64 bytes (pixels). Output: 1 byte class.
 * Weights format: magic 0x44545245, uint32 node_count, then node_count * 12 bytes:
 *   uint8 feature, uint8 threshold, uint8 class, uint8 pad, int32 left, int32 right.
 * Leaf: left = -1, right = -1. Internal: left/right >= 0. */
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

typedef struct __attribute__((packed)) {
    uint8_t feature;
    uint8_t threshold;
    uint8_t cls;
    uint8_t pad;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = *(uint32_t*)wb;
    if (magic != 0x44545245) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t node_count = *(uint32_t*)(wb + 4);
    if (nw != 8 + (long)node_count * 12) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node*)(wb + 8);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        uint32_t idx = 0;
        while (1) {
            Node *nd = &nodes[idx];
            if (nd->left == -1) { out_byte = nd->cls; break; }
            if (xi[nd->feature] <= nd->threshold) idx = (uint32_t)nd->left;
            else idx = (uint32_t)nd->right;
        }
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```