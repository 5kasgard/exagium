```hypothesis
We propose a "label-rf" family: a small random forest (5 trees, max depth 8) with majority voting.
It should beat linear logistic regression on optdigits because random forests capture non-linear pixel interactions,
and the dataset size (2867) is sufficient for a few trees. The model is fast at inference (only integer comparisons)
and has low memory footprint. If dev accuracy does not exceed 0.9613, or if inference cost exceeds label-linear,
the hypothesis fails.
```
```python
import json, struct, sys
import numpy as np

NUM_TREES = 5
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = 8
BASE_SEED = 1

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

def build_tree(X, y, depth, rng, max_features):
    n, d = X.shape
    if depth >= MAX_DEPTH or n <= MIN_SAMPLES_LEAF:
        leaf_class = int(np.bincount(y).argmax())
        return [(-1, -1, -1, -1, leaf_class)]
    if np.all(y == y[0]):
        leaf_class = int(y[0])
        return [(-1, -1, -1, -1, leaf_class)]
    
    features = rng.choice(d, size=min(max_features, d), replace=False)
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    
    for feat in features:
        vals = X[:, feat]
        hist = np.zeros((17, 10), dtype=np.int32)
        for v, c in zip(vals, y):
            hist[v, c] += 1
        cum_left = np.zeros(10, dtype=np.int32)
        cum_right = hist.sum(axis=0).astype(np.int32)
        for t in range(16):
            cum_left += hist[t]
            cum_right -= hist[t]
            n_left = cum_left.sum()
            n_right = cum_right.sum()
            if n_left < MIN_SAMPLES_LEAF or n_right < MIN_SAMPLES_LEAF:
                continue
            gini_left = 1.0 - np.sum((cum_left / n_left) ** 2) if n_left > 0 else 0
            gini_right = 1.0 - np.sum((cum_right / n_right) ** 2) if n_right > 0 else 0
            weighted_gini = (n_left * gini_left + n_right * gini_right) / n
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feat = feat
                best_thresh = t
    
    if best_feat == -1:
        leaf_class = int(np.bincount(y).argmax())
        return [(-1, -1, -1, -1, leaf_class)]
    
    left_mask = X[:, best_feat] <= best_thresh
    right_mask = ~left_mask
    left_nodes = build_tree(X[left_mask], y[left_mask], depth + 1, rng, max_features)
    right_nodes = build_tree(X[right_mask], y[right_mask], depth + 1, rng, max_features)
    
    left_len = len(left_nodes)
    current_node = (best_feat, best_thresh, 1, 1 + left_len, -1)
    return [current_node] + left_nodes + right_nodes

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    
    all_trees = []
    for t in range(NUM_TREES):
        rng = np.random.default_rng(BASE_SEED + t)
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        tree = build_tree(X_boot, y_boot, 0, rng, MAX_FEATURES)
        all_trees.append(tree)
    
    magic = 0x464F5245
    header = struct.pack("<5i", magic, NUM_TREES, MAX_DEPTH, d, C)
    out = bytearray(header)
    for tree in all_trees:
        num_nodes = len(tree)
        out += struct.pack("<i", num_nodes)
        for node in tree:
            feat, thresh, left, right, leaf = node
            out += struct.pack("<5i", feat, thresh, left, right, leaf)
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

struct Node { int32_t feat, thresh, left, right, leaf; };

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x464F5245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int num_trees = hd[1];
    int max_depth = hd[2];
    int d = hd[3];
    int C = hd[4];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights params\n"); return 3; }
    
    int32_t *ptr = hd + 5;
    int32_t *end = hd + nw / 4;
    
    struct Node **trees = malloc(num_trees * sizeof(struct Node*));
    int *tree_nodes = malloc(num_trees * sizeof(int));
    
    for (int t = 0; t < num_trees; t++) {
        if (ptr + 1 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        int n_nodes = *ptr++;
        tree_nodes[t] = n_nodes;
        if (ptr + n_nodes * 5 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t] = (struct Node*)ptr;
        ptr += n_nodes * 5;
    }
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    int *votes = calloc(C, sizeof(int));
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        
        memset(votes, 0, C * sizeof(int));
        for (int t = 0; t < num_trees; t++) {
            int node_idx = 0;
            while (1) {
                struct Node node = trees[t][node_idx];
                if (node.leaf != -1) {
                    votes[node.leaf]++;
                    break;
                }
                if (xi[node.feat] <= node.thresh) {
                    node_idx = node.left;
                } else {
                    node_idx = node.right;
                }
            }
        }
        int best = 0;
        for (int c = 1; c < C; c++) {
            if (votes[c] > votes[best]) best = c;
        }
        unsigned char out = (unsigned char)best;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```