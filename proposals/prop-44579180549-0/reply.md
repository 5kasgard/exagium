```hypothesis
Family: label-rf (Random Forest with 10 trees, max depth 10). Random Forests are strong on tabular data and often outperform linear models on Adult. Deterministic training via fixed seed and bootstrap. Inference is fast (only byte comparisons). Failure: dev accuracy ≤ 0.8524 or Ir/item ≥ 719.
```
```python
"""Random Forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: seed 1, single thread. 10 trees, max depth 10, min_samples_split=10, min_samples_leaf=5.
Exported as flat node arrays: magic 0x52464C31, num_trees, num_features, total_nodes; root_indices[num_trees];
nodes[total_nodes] each: feature(uint8), threshold(uint8), is_leaf(uint8), class(uint8), left(int32), right(int32)."""
import json, struct, sys

import numpy as np

NUM_TREES = 10
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
SEED = 1
MAGIC = 0x52464C31  # '1LFR' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_tree(X, y, indices, depth, nodes, rng):
    n = len(indices)
    if n < MIN_SAMPLES_SPLIT or depth == MAX_DEPTH:
        cls = int(np.bincount(y[indices]).argmax())
        nodes.append((0, 0, 1, cls, -1, -1))
        return len(nodes) - 1
    if np.all(y[indices] == y[indices[0]]):
        cls = int(y[indices[0]])
        nodes.append((0, 0, 1, cls, -1, -1))
        return len(nodes) - 1

    best_gini = 1.0
    best_feature = -1
    best_threshold = -1

    for f in range(X.shape[1]):
        vals = X[indices, f]
        labs = y[indices]
        hist = np.zeros((256, 2), dtype=np.int32)
        np.add.at(hist, (vals, labs), 1)
        left_counts = np.cumsum(hist[:-1], axis=0)          # (255,2)
        right_counts = hist[1:].sum(axis=0) - left_counts   # (255,2)
        left_n = left_counts.sum(axis=1)
        right_n = right_counts.sum(axis=1)
        valid = (left_n >= MIN_SAMPLES_LEAF) & (right_n >= MIN_SAMPLES_LEAF)
        if not np.any(valid):
            continue
        left_p = left_counts / left_n[:, None]
        right_p = right_counts / right_n[:, None]
        left_gini = 1.0 - (left_p[:, 0] ** 2 + left_p[:, 1] ** 2)
        right_gini = 1.0 - (right_p[:, 0] ** 2 + right_p[:, 1] ** 2)
        weighted_gini = (left_n * left_gini + right_n * right_gini) / n
        idx = int(np.argmin(np.where(valid, weighted_gini, 2.0)))
        if weighted_gini[idx] < best_gini:
            best_gini = weighted_gini[idx]
            best_feature = f
            best_threshold = idx

    if best_feature == -1:
        cls = int(np.bincount(y[indices]).argmax())
        nodes.append((0, 0, 1, cls, -1, -1))
        return len(nodes) - 1

    left_mask = X[indices, best_feature] <= best_threshold
    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]

    left_child = build_tree(X, y, left_indices, depth + 1, nodes, rng)
    right_child = build_tree(X, y, right_indices, depth + 1, nodes, rng)
    nodes.append((best_feature, best_threshold, 0, 0, left_child, right_child))
    return len(nodes) - 1


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n, d = X.shape
    rng = np.random.default_rng(SEED)

    all_nodes = []
    root_indices = []
    for _ in range(NUM_TREES):
        boot_idx = rng.integers(0, n, size=n)
        tree_nodes = []
        root_local = build_tree(X, y, boot_idx, 0, tree_nodes, rng)
        offset = len(all_nodes)
        all_nodes.extend(tree_nodes)
        root_indices.append(offset + root_local)

    total_nodes = len(all_nodes)
    # header
    out = struct.pack("<IIII", MAGIC, NUM_TREES, d, total_nodes)
    # root indices
    out += struct.pack("<" + "I" * NUM_TREES, *root_indices)
    # nodes
    for node in all_nodes:
        f, th, leaf, cls, left, right = node
        out += struct.pack("<BBBBii", f, th, leaf, cls, left, right)

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Random Forest inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the majority class across trees.
 * Weights format: magic 0x52464C31, num_trees, num_features, total_nodes;
 * root_indices[num_trees] (uint32_t); nodes[total_nodes] each:
 *   feature(uint8), threshold(uint8), is_leaf(uint8), class(uint8), left(int32_t), right(int32_t). */
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
    uint8_t feature;
    uint8_t threshold;
    uint8_t is_leaf;
    uint8_t class_;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x52464C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t num_trees = hdr[1];
    uint32_t num_features = hdr[2];
    uint32_t total_nodes = hdr[3];
    size_t expected = 16 + 4ULL * num_trees + 12ULL * total_nodes;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    uint32_t *roots = (uint32_t *)(wb + 16);
    Node *nodes = (Node *)(wb + 16 + 4 * num_trees);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != num_features) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, num_features); return 3; }
        int votes[2] = {0, 0};
        for (uint32_t t = 0; t < num_trees; t++) {
            uint32_t idx = roots[t];
            while (1) {
                Node nd = nodes[idx];
                if (nd.is_leaf) {
                    votes[nd.class_]++;
                    break;
                }
                if (xi[nd.feature] <= nd.threshold) idx = (uint32_t)nd.left;
                else idx = (uint32_t)nd.right;
            }
        }
        unsigned char best = votes[1] > votes[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```