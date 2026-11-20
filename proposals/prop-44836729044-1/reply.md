```hypothesis
Family: label-tree (single decision tree on ordinal pixels).
Why: Optdigits 8x8 pixels have strong local patterns; a depth-15 CART with Gini splitting captures non-linear interactions that linear models miss, while inference is just a few integer comparisons per sample (far fewer cycles than 64x10 MACs). If dev accuracy falls below 0.95, the hypothesis fails.
```
```python
"""label-tree: single deterministic CART on ordinal 0..16 pixels.
python3 -s -P train.py card.json train.in train.key weights.bin
Weights format: int32 magic 0x54524545, int32 n_nodes, then n_nodes * Node
Node (C struct, packed): uint8 is_leaf, uint8 feature, uint8 threshold, uint8 pad, int32 left, int32 right, uint8 cls.
Split: x[feature] <= threshold -> left, else right. Threshold in 0..15 (meaning <= threshold)."""
import json, struct, sys
import numpy as np

MAX_DEPTH = 15
MIN_LEAF = 5
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(y):
    if len(y) == 0: return 0.0
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, rng):
    n, d = X.shape
    best_gini = 1.0
    best_feat = best_thr = None
    for f in range(d):
        vals = np.unique(X[:, f])
        for v in vals[:-1]:
            mask = X[:, f] <= v
            if mask.sum() < MIN_LEAF or (~mask).sum() < MIN_LEAF:
                continue
            g = (mask.sum() * gini(y[mask]) + (~mask).sum() * gini(y[~mask])) / n
            if g < best_gini:
                best_gini, best_feat, best_thr = g, f, v
    return best_feat, best_thr, best_gini

def build_tree(X, y, depth, nodes, rng):
    idx = len(nodes)
    if depth >= MAX_DEPTH or len(np.unique(y)) == 1 or len(y) < 2 * MIN_LEAF:
        cls = np.bincount(y).argmax()
        nodes.append((1, 0, 0, 0, -1, -1, cls))
        return idx
    feat, thr, _ = best_split(X, y, rng)
    if feat is None:
        cls = np.bincount(y).argmax()
        nodes.append((1, 0, 0, 0, -1, -1, cls))
        return idx
    mask = X[:, feat] <= thr
    left = build_tree(X[mask], y[mask], depth + 1, nodes, rng)
    right = build_tree(X[~mask], y[~mask], depth + 1, nodes, rng)
    nodes.append((0, feat, thr, 0, left, right, 0))
    return idx

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    nodes = []
    build_tree(X, y, 0, nodes, rng)
    # Write weights
    magic = 0x54524545  # 'TREE'
    out = struct.pack("<2i", magic, len(nodes))
    for nd in nodes:
        out += struct.pack("<BBBBiiiB", *nd)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N */
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

typedef struct { uint8_t leaf, feat, thr, pad; int32_t left, right; uint8_t cls; } Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t nnodes = (int32_t)(wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24);
    if (nw != 8 + nnodes * 16) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 8);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int idx = 0;
        while (1) {
            Node nd = nodes[idx];
            if (nd.leaf) { put_rec(stdout, &nd.cls, 1); break; }
            if (xi[nd.feat] <= nd.thr) idx = nd.left;
            else idx = nd.right;
        }
    }
    return 0;
}
```