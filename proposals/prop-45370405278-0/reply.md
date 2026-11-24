```hypothesis
A depth-5 decision tree with target-encoded categorical features should capture non-linear interactions that linear models miss, while inference requires only a few table lookups and integer comparisons. Expect ~0.860 accuracy at ~200 Ir/item. If dev accuracy ≤ 0.8524 or cost > 1000 Ir/item, the hypothesis fails.
```
```python
"""label-tree: shallow decision tree with target encoding for categorical features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, struct, sys
import numpy as np

MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 10
SEED = 1

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

def build_tree(X, y, depth, node_list, feature_maps):
    n = len(y)
    if depth >= MAX_DEPTH or n < MIN_SAMPLES_LEAF or len(np.unique(y)) == 1:
        leaf_class = 0 if np.sum(y == 0) >= np.sum(y == 1) else 1
        node_list.append((255, leaf_class, 0, 0))  # feature_idx=255 marks leaf, threshold=class
        return len(node_list) - 1

    parent_gini = gini(y)
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1

    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        if len(vals) <= 1:
            continue
        for t in vals[:-1]:  # split on <= t
            left_mask = X[:, f] <= t
            right_mask = ~left_mask
            n_left, n_right = left_mask.sum(), right_mask.sum()
            if n_left == 0 or n_right == 0:
                continue
            gain = parent_gini - (n_left / n) * gini(y[left_mask]) - (n_right / n) * gini(y[right_mask])
            if gain > best_gain or (gain == best_gain and (f < best_feat or (f == best_feat and t < best_thresh))):
                best_gain, best_feat, best_thresh = gain, f, t

    if best_gain <= 0:
        leaf_class = 0 if np.sum(y == 0) >= np.sum(y == 1) else 1
        node_list.append((255, leaf_class, 0, 0))
        return len(node_list) - 1

    left_mask = X[:, best_feat] <= best_thresh
    right_mask = ~left_mask
    left_idx = build_tree(X[left_mask], y[left_mask], depth + 1, node_list, feature_maps)
    right_idx = build_tree(X[right_mask], y[right_mask], depth + 1, node_list, feature_maps)
    node_list.append((best_feat, best_thresh, left_idx, right_idx))
    return len(node_list) - 1

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    # Build target encoding maps for categorical features
    rng = np.random.default_rng(SEED)
    maps = []
    for j, t in enumerate(types):
        if t == "categorical":
            # Compute target mean per seen category (0-254)
            seen_vals = np.unique(X[:, j])
            seen_vals = seen_vals[seen_vals != 255]  # exclude unseen token
            means = {}
            for v in seen_vals:
                mask = X[:, j] == v
                means[v] = y[mask].mean() if mask.any() else 0.5
            # Sort categories by mean target, assign ranks 0..k-1
            sorted_cats = sorted(means.keys(), key=lambda v: means[v])
            rank_map = np.full(256, len(sorted_cats), dtype=np.uint8)  # default rank for unseen (255) and missing
            for rank, v in enumerate(sorted_cats):
                rank_map[v] = rank
            maps.append(rank_map)
        else:  # ordinal: identity map
            maps.append(np.arange(256, dtype=np.uint8))

    # Apply maps
    X_mapped = np.empty_like(X)
    for j in range(d):
        X_mapped[:, j] = maps[j][X[:, j]]

    # Train tree
    node_list = []
    build_tree(X_mapped, y, 0, node_list, maps)
    num_nodes = len(node_list)

    # Serialize weights.bin
    # Magic: 0x54524545 ('TREE' little-endian), d, num_nodes
    out = struct.pack("<3I", 0x54524545, d, num_nodes)
    # Feature maps: d * 256 uint8
    for m in maps:
        out += m.tobytes()
    # Nodes: each node = uint8 feat, uint8 thresh, uint16 left, uint16 right
    for feat, thresh, left, right in node_list:
        out += struct.pack("<BBHH", feat, thresh, left, right)

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output record: 1 byte class index. */
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
    if (nw < 12) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    uint32_t d = wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24;
    uint32_t num_nodes = wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24;
    if (magic != 0x54524545 || d != 14) { fprintf(stderr, "bad weights: magic or d\n"); return 3; }
    long expected = 12L + (long)d * 256 + (long)num_nodes * 6;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const unsigned char *maps = wb + 12;
    const unsigned char *nodes = maps + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char mapped[14];
    unsigned char out_class;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            mapped[j] = maps[(long)j * 256 + xi[j]];
        }
        uint32_t node_idx = 0;
        while (1) {
            const unsigned char *node = nodes + (long)node_idx * 6;
            uint8_t feat = node[0];
            uint8_t thresh = node[1];
            uint16_t left = node[2] | node[3] << 8;
            uint16_t right = node[4] | node[5] << 8;
            if (feat == 255) {
                out_class = thresh;
                break;
            }
            if (mapped[feat] <= thresh) node_idx = left;
            else node_idx = right;
        }
        put_rec(stdout, &out_class, 1);
    }
    return 0;
}
```