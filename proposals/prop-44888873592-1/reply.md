```hypothesis
Family: label-tree (single CART decision tree). The 8x8 digit pixels have strong local spatial interactions (strokes, loops) that a linear model cannot capture, but a shallow tree can model via axis-aligned splits on nearby pixels. A depth-8..10 tree needs only ~10-20 comparisons per item (<< label-linear's 5058 Ir/item) and should reach ~0.97 accuracy. If dev accuracy stays ≤0.962 or Ir/item exceeds 6000, the hypothesis fails.
```
```python
"""label-tree: single CART decision tree for ordinal byte features.
train.py card.json train.in train.key weights.bin
Deterministic (fixed tie-breaking), single thread, stdlib+numpy only."""
import json, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_SPLIT = 2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)


def build_tree(X, y, indices, depth, max_depth, nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class):
    """Recursively build tree, appending nodes to lists. Returns node index."""
    n = len(indices)
    classes, counts = np.unique(y[indices], return_counts=True)
    majority_class = classes[np.argmax(counts)]

    # Stopping criteria
    if len(classes) == 1 or depth >= max_depth or n < MIN_SAMPLES_SPLIT:
        nodes_feat.append(255)          # leaf marker
        nodes_thresh.append(0)
        nodes_left.append(-1)
        nodes_right.append(-1)
        nodes_class.append(int(majority_class))
        return len(nodes_feat) - 1

    total = n
    parent_gini = gini_from_counts(counts)
    best_gini = parent_gini
    best_feat = -1
    best_thresh = -1

    # Try all features and thresholds (values 0..16 -> thresholds 0..15 for <= split)
    for feat in range(X.shape[1]):
        vals = X[indices, feat]
        # Histogram: 17 values (0..16) x 10 classes
        hist = np.zeros((17, 10), dtype=np.int32)
        for idx in indices:
            v = X[idx, feat]
            c = y[idx]
            hist[v, c] += 1

        left_counts = np.zeros(10, dtype=np.int32)
        right_counts = hist.sum(axis=0).copy()

        for thresh in range(16):  # split: x <= thresh -> left
            left_counts += hist[thresh]
            right_counts -= hist[thresh]
            left_n = left_counts.sum()
            right_n = right_counts.sum()
            if left_n == 0 or right_n == 0:
                continue
            left_gini = gini_from_counts(left_counts)
            right_gini = gini_from_counts(right_counts)
            weighted = (left_n * left_gini + right_n * right_gini) / total
            # Deterministic tie-break: lower feature, then lower threshold
            if weighted < best_gini - 1e-12:
                best_gini = weighted
                best_feat = feat
                best_thresh = thresh

    if best_feat == -1:  # no improving split
        nodes_feat.append(255)
        nodes_thresh.append(0)
        nodes_left.append(-1)
        nodes_right.append(-1)
        nodes_class.append(int(majority_class))
        return len(nodes_feat) - 1

    # Split indices
    left_mask = X[indices, best_feat] <= best_thresh
    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]

    # Create internal node (placeholder, will fill children after recursion)
    node_idx = len(nodes_feat)
    nodes_feat.append(best_feat)
    nodes_thresh.append(best_thresh)
    nodes_left.append(-1)   # temporary
    nodes_right.append(-1)
    nodes_class.append(0)   # unused for internal

    left_child = build_tree(X, y, left_indices, depth + 1, max_depth,
                            nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class)
    right_child = build_tree(X, y, right_indices, depth + 1, max_depth,
                             nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class)

    nodes_left[node_idx] = left_child
    nodes_right[node_idx] = right_child
    return node_idx


def evaluate_tree(X, y, indices, nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class):
    correct = 0
    for idx in indices:
        node = 0
        while nodes_feat[node] != 255:
            feat = nodes_feat[node]
            thresh = nodes_thresh[node]
            if X[idx, feat] <= thresh:
                node = nodes_left[node]
            else:
                node = nodes_right[node]
        if nodes_class[node] == y[idx]:
            correct += 1
    return correct / len(indices)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    assert n == y.shape[0]

    # Tuning split: last 20% in file order
    val_size = n // 5
    train_idx = np.arange(n - val_size, dtype=np.int64)
    val_idx = np.arange(n - val_size, n, dtype=np.int64)

    best_depth = 1
    best_acc = 0.0

    # Try depths 1..15 (plenty for 2867 samples)
    for max_depth in range(1, 16):
        nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class = [], [], [], [], []
        build_tree(X, y, train_idx, 0, max_depth,
                   nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class)
        acc = evaluate_tree(X, y, val_idx,
                            nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class)
        if acc > best_acc:
            best_acc = acc
            best_depth = max_depth

    # Retrain on full data with best depth
    nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class = [], [], [], [], []
    build_tree(X, y, np.arange(n, dtype=np.int64), 0, best_depth,
               nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class)

    num_nodes = len(nodes_feat)
    # Pack weights: magic, num_nodes, max_depth, then nodes (feature u8, thresh u8, left i32, right i32, class u8, pad 3)
    magic = 0x54524545  # "TREE"
    header = struct.pack("<3I", magic, num_nodes, best_depth)
    node_data = bytearray()
    for i in range(num_nodes):
        node_data += struct.pack("<BBiiiB3x",
                                 nodes_feat[i],
                                 nodes_thresh[i],
                                 nodes_left[i],
                                 nodes_right[i],
                                 nodes_class[i])
    with open(sys.argv[4], "wb") as f:
        f.write(header + node_data)


if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (64 for optdigits); output is one byte class.
 * Weights format: uint32 magic(0x54524545), num_nodes, max_depth;
 * then num_nodes * {uint8 feat, uint8 thresh, int32 left, int32 right, uint8 class, uint8 pad[3]} */
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
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)w;
    if (hd[0] != 0x54524545) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    uint32_t num_nodes = hd[1];
    uint32_t max_depth = hd[2];  // unused at inference
    size_t expected = 12 + (size_t)num_nodes * 16;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    // Node arrays
    const unsigned char *nodes = w + 12;
    const uint8_t *feat = (const uint8_t *)nodes;
    const uint8_t *thresh = (const uint8_t *)(nodes + 1);
    const int32_t *left = (const int32_t *)(nodes + 2);  // misaligned but OK on x86; use memcpy for strict
    const int32_t *right = (const int32_t *)(nodes + 6);
    const uint8_t *cls = (const uint8_t *)(nodes + 10);
    // Better: parse per node with stride 16
    // We'll just use pointer arithmetic per node in the loop.

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        uint32_t node = 0;
        while (1) {
            const unsigned char *nd = nodes + node * 16;
            uint8_t f = nd[0];
            if (f == 255) {  // leaf
                unsigned char pred = nd[10];
                put_rec(stdout, &pred, 1);
                break;
            }
            uint8_t t = nd[1];
            int32_t l = *(const int32_t *)(nd + 2);
            int32_t r = *(const int32_t *)(nd + 6);
            if (xi[f] <= t) node = (uint32_t)l;
            else node = (uint32_t)r;
        }
    }
    return 0;
}
```