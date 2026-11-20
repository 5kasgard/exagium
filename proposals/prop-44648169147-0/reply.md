```hypothesis
Family: label-tree (single decision tree on one-hot encoded features).
Why: The Adult dataset has strong non-linear interactions (e.g., education × occupation) that a linear model misses. A depth-5 tree on the same one-hot encoding as label-linear captures these with minimal overhead. Inference is a handful of byte comparisons and float checks per sample, so CPU cost should stay near label-linear.
Failure sign: Dev accuracy ≤ 0.8524 (label-linear) or instructions/item > 2× label-linear.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 10


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]
    n = X.shape[0]

    # Build expanded features: ordinal -> raw byte (0-255), categorical -> one-hot per seen category
    expanded = []          # list of (orig_idx, is_cat, category)
    Z_list = []
    for j, t in enumerate(types):
        if t == "ordinal":
            Z_list.append(X[:, j].astype(np.uint8).reshape(-1, 1))
            expanded.append((j, 0, 255))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((n, len(vals)), dtype=np.uint8)
            for idx, v in enumerate(vals):
                E[X[:, j] == v, idx] = 1
            Z_list.append(E)
            for v in vals:
                expanded.append((j, 1, int(v)))
    Z = np.concatenate(Z_list, axis=1)   # (n, n_expanded), uint8
    n_expanded = Z.shape[1]

    # Decision tree training (CART, Gini, histogram-based split finding)
    nodes = []          # each node: (split_feat, threshold, left, right, leaf_class)
    nodes.append(None)  # root placeholder
    stack = [(np.arange(n), 0, 0)]  # (indices, depth, node_idx)

    while stack:
        idx, depth, node_idx = stack.pop()
        n_idx = len(idx)
        # Stopping criteria
        if depth >= MAX_DEPTH or n_idx < 2 * MIN_SAMPLES_LEAF or np.unique(y[idx]).size == 1:
            leaf_class = int(np.bincount(y[idx]).argmax())
            nodes[node_idx] = (-1, 0.0, -1, -1, leaf_class)
            continue

        best_gini = 1.0
        best_feat = -1
        best_thresh = 0.0

        # For each expanded feature, compute histograms per class
        for feat in range(n_expanded):
            vals = Z[idx, feat]          # uint8, 0-255
            labels = y[idx]
            # Histogram per class: shape (C, 256)
            hist = np.zeros((C, 256), dtype=np.int32)
            for c in range(C):
                hist[c] = np.bincount(vals[labels == c], minlength=256)
            total = hist.sum(axis=0)                     # (256,)
            cum_hist = np.cumsum(hist, axis=1)           # (C, 256)
            cum_total = np.cumsum(total)                 # (256,)

            # Thresholds t = 0..254 (split between t and t+1)
            left_counts = cum_hist[:, :-1]               # (C, 255)
            left_total = cum_total[:-1]                  # (255,)
            right_counts = hist.sum(axis=1, keepdims=True) - left_counts
            right_total = n_idx - left_total

            mask = (left_total >= MIN_SAMPLES_LEAF) & (right_total >= MIN_SAMPLES_LEAF)
            if not mask.any():
                continue

            left_probs = left_counts / left_total
            right_probs = right_counts / right_total
            gini_left = 1.0 - np.sum(left_probs ** 2, axis=0)
            gini_right = 1.0 - np.sum(right_probs ** 2, axis=0)
            gini = (left_total * gini_left + right_total * gini_right) / n_idx

            gini_masked = np.where(mask, gini, np.inf)
            t = int(np.argmin(gini_masked))
            if gini_masked[t] < best_gini:
                best_gini = gini_masked[t]
                best_feat = feat
                best_thresh = t + 0.5

        if best_feat == -1:
            leaf_class = int(np.bincount(y[idx]).argmax())
            nodes[node_idx] = (-1, 0.0, -1, -1, leaf_class)
            continue

        # Split
        left_mask = Z[idx, best_feat] <= best_thresh
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        left_node = len(nodes)
        right_node = left_node + 1
        nodes.append(None)
        nodes.append(None)
        nodes[node_idx] = (best_feat, float(best_thresh), left_node, right_node, -1)
        stack.append((right_idx, depth + 1, right_node))
        stack.append((left_idx, depth + 1, left_node))

    # Write weights.bin
    magic = 0x54524545  # "TREE"
    out = struct.pack("<4i", magic, d, n_expanded, len(nodes))
    for orig_idx, is_cat, category in expanded:
        out += struct.pack("<BBBx", orig_idx, is_cat, category)
    for split_feat, thresh, left, right, leaf_class in nodes:
        out += struct.pack("<iffii", split_feat, thresh, left, right, leaf_class)
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

typedef struct {
    uint8_t orig_idx;
    uint8_t is_cat;
    uint8_t category;
} ExpFeat;

typedef struct {
    int32_t split_feat;
    float threshold;
    int32_t left;
    int32_t right;
    int8_t leaf_class;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], n_expanded = hdr[2], num_nodes = hdr[3];
    if (d != 14 || n_expanded <= 0 || num_nodes <= 0) { fprintf(stderr, "bad header\n"); return 3; }

    size_t exp_offset = 16;
    size_t node_offset = exp_offset + n_expanded * 4;
    size_t expected = node_offset + num_nodes * 20;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    ExpFeat *exp = (ExpFeat *)(wb + exp_offset);
    Node *nodes = (Node *)(wb + node_offset);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int node_idx = 0;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.leaf_class >= 0) {
                put_rec(stdout, (unsigned char *)&nd.leaf_class, 1);
                break;
            }
            ExpFeat ef = exp[nd.split_feat];
            float val = ef.is_cat ? (xi[ef.orig_idx] == ef.category ? 1.0f : 0.0f) : (float)xi[ef.orig_idx];
            node_idx = (val <= nd.threshold) ? nd.left : nd.right;
        }
    }
    return 0;
}
```