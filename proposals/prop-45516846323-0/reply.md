```hypothesis
A single CART decision tree (max depth 8, min_samples_leaf=20) with greedy information-gain splitting on the 14 quantized features. Trees naturally capture the pairwise interactions (e.g., education × occupation) that a linear model misses, so dev accuracy should exceed label-linear's 0.8524. Inference is a tight loop of ≤8 byte comparisons and branches, costing far less than label-linear's ~719 Ir/item. If accuracy ≤0.8524, the quantized Adult representation is effectively linearly separable.
```
```python
"""CART decision tree for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed tie-breaking), single-thread, stdlib+numpy only.
Exports: magic 0x54524545, n_nodes, max_depth, then per node:
  u8 is_leaf, u8 feat, u8 thresh, i32 left, i32 right, u8 leaf_class
"""
import json, struct, sys
import numpy as np

MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 20
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    nz = p > 0
    return -np.sum(p[nz] * np.log2(p[nz]))

def build_tree(X, y, depth, node_idx, nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class, nodes_is_leaf):
    n = len(y)
    if n < MIN_SAMPLES_LEAF or depth >= MAX_DEPTH or len(np.unique(y)) == 1:
        nodes_is_leaf[node_idx] = 1
        nodes_class[node_idx] = np.bincount(y, minlength=2).argmax()
        return
    best_gain = -1.0
    best_feat = 0
    best_thresh = 0
    parent_ent = entropy(np.bincount(y, minlength=2))
    for f in range(X.shape[1]):
        vals = X[:, f]
        # histogram per class
        hist = np.zeros((256, 2), dtype=np.int32)
        np.add.at(hist, (vals, y), 1)
        left_counts = np.zeros(2, dtype=np.int32)
        for t in range(255):
            left_counts += hist[t]
            right_counts = hist.sum(axis=0) - left_counts
            n_left = left_counts.sum()
            n_right = right_counts.sum()
            if n_left < MIN_SAMPLES_LEAF or n_right < MIN_SAMPLES_LEAF:
                continue
            gain = parent_ent - (n_left/n)*entropy(left_counts) - (n_right/n)*entropy(right_counts)
            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thresh = t
    if best_gain <= 0:
        nodes_is_leaf[node_idx] = 1
        nodes_class[node_idx] = np.bincount(y, minlength=2).argmax()
        return
    nodes_feat[node_idx] = best_feat
    nodes_thresh[node_idx] = best_thresh
    nodes_is_leaf[node_idx] = 0
    mask = X[:, best_feat] <= best_thresh
    left_idx = len(nodes_feat)
    nodes_feat.append(0); nodes_thresh.append(0); nodes_left.append(0); nodes_right.append(0); nodes_class.append(0); nodes_is_leaf.append(0)
    nodes_left[node_idx] = left_idx
    build_tree(X[mask], y[mask], depth+1, left_idx, nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class, nodes_is_leaf)
    right_idx = len(nodes_feat)
    nodes_feat.append(0); nodes_thresh.append(0); nodes_left.append(0); nodes_right.append(0); nodes_class.append(0); nodes_is_leaf.append(0)
    nodes_right[node_idx] = right_idx
    build_tree(X[~mask], y[~mask], depth+1, right_idx, nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class, nodes_is_leaf)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    nodes_feat = [0]; nodes_thresh = [0]; nodes_left = [0]; nodes_right = [0]; nodes_class = [0]; nodes_is_leaf = [0]
    build_tree(X, y, 0, 0, nodes_feat, nodes_thresh, nodes_left, nodes_right, nodes_class, nodes_is_leaf)
    n_nodes = len(nodes_feat)
    out = struct.pack("<3i", 0x54524545, n_nodes, MAX_DEPTH)
    for i in range(n_nodes):
        out += struct.pack("<BBBiiiB",
            nodes_is_leaf[i],
            nodes_feat[i],
            nodes_thresh[i],
            nodes_left[i] if not nodes_is_leaf[i] else -1,
            nodes_right[i] if not nodes_is_leaf[i] else -1,
            nodes_class[i]
        )
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* CART decision tree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14 for adult). Output: 1 byte class.
 * Weights format: magic 0x54524545, n_nodes, max_depth, then per node:
 *   u8 is_leaf, u8 feat, u8 thresh, i32 left, i32 right, u8 leaf_class
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
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t n_nodes = (int32_t)(wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24);
    int32_t max_depth = (int32_t)(wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24);
    (void)max_depth;
    if (n_nodes <= 0 || nw != 12 + (long)n_nodes * 16) { fprintf(stderr, "bad weights size\n"); return 3; }
    uint8_t *is_leaf = malloc(n_nodes);
    uint8_t *feat = malloc(n_nodes);
    uint8_t *thresh = malloc(n_nodes);
    int32_t *left = malloc(n_nodes * 4);
    int32_t *right = malloc(n_nodes * 4);
    uint8_t *leaf_class = malloc(n_nodes);
    const unsigned char *p = wb + 12;
    for (int i = 0; i < n_nodes; i++) {
        is_leaf[i] = p[0];
        feat[i] = p[1];
        thresh[i] = p[2];
        left[i] = (int32_t)(p[3] | p[4] << 8 | p[5] << 16 | p[6] << 24);
        right[i] = (int32_t)(p[7] | p[8] << 8 | p[9] << 16 | p[10] << 24);
        leaf_class[i] = p[11];
        p += 16;
    }
    free(wb);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }
        int node = 0;
        while (!is_leaf[node]) {
            if (xi[feat[node]] <= thresh[node]) node = left[node];
            else node = right[node];
        }
        unsigned char out = leaf_class[node];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```