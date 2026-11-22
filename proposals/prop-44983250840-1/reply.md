```hypothesis
Family: label-tinygbdt — a minimal gradient-boosted decision tree ensemble (30 trees, depth 3) with per-pixel z-scoring.
Why: 8×8 digits have strong local spatial interactions (strokes, loops) that linear models miss; shallow trees capture these with few splits.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or inference cost ≥ 5000 Ir/item.
```

```python
"""label-tinygbdt: tiny GBDT for ordinal byte features.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, numpy+stdlib only.
Exports: magic(0x47424454), d, C, n_trees, max_depth, n_nodes_total;
         nodes: feat_i16, thresh_f32, left_i16, right_i16;
         leaves: C*f32 per leaf (interleaved after nodes)."""
import json, math, struct, sys
import numpy as np

SEED = 1
N_TREES = 30
MAX_DEPTH = 3
LR = 0.1
L2 = 1e-4
SUBSAMPLE = 1.0

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Tree:
    __slots__ = ("feat", "thresh", "left", "right", "leaf_val", "is_leaf")
    def __init__(self, max_nodes):
        self.feat = np.zeros(max_nodes, np.int16)
        self.thresh = np.zeros(max_nodes, np.float32)
        self.left = np.zeros(max_nodes, np.int16)
        self.right = np.zeros(max_nodes, np.int16)
        self.leaf_val = np.zeros((max_nodes, 10), np.float32)
        self.is_leaf = np.zeros(max_nodes, np.bool_)
        self.n_nodes = 1
        self.is_leaf[0] = True

def softmax_grad(logits, y):
    """logits: (n, C), y: (n,) -> grad: (n, C) = prob - onehot"""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    prob = exp / exp.sum(axis=1, keepdims=True)
    prob[np.arange(len(y)), y] -= 1.0
    return prob.astype(np.float32)

def mse_grad(pred, target):
    return (pred - target).astype(np.float32)

def build_tree(X, g, h, idxs, depth, rng, max_depth, node, tree):
    """Fit a regression tree to gradients g (n, C) with Hessian h (n, C) approx 1."""
    n = len(idxs)
    if depth == max_depth or n < 10:
        # leaf: Newton step = -sum(g) / (sum(h) + L2)
        tree.is_leaf[node] = True
        tree.leaf_val[node] = -g[idxs].sum(axis=0) / (n + L2)
        return
    best_gain = -1.0
    best_f, best_t = -1, 0.0
    # try each feature
    for f in range(X.shape[1]):
        vals = X[idxs, f]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        # midpoints
        threshes = (uniq[:-1] + uniq[1:]) * 0.5
        for t in threshes:
            left_mask = vals <= t
            if not left_mask.any() or left_mask.all():
                continue
            l_idx = idxs[left_mask]
            r_idx = idxs[~left_mask]
            # gain = GL^2/(HL+L2) + GR^2/(HR+L2) - G^2/(H+L2)
            GL = g[l_idx].sum(axis=0); GR = g[r_idx].sum(axis=0)
            HL = len(l_idx); HR = len(r_idx)
            gain = np.sum(GL*GL/(HL+L2) + GR*GR/(HR+L2)) - np.sum((GL+GR)*(GL+GR)/(n+L2))
            if gain > best_gain:
                best_gain, best_f, best_t = gain, f, t
    if best_gain <= 0:
        tree.is_leaf[node] = True
        tree.leaf_val[node] = -g[idxs].sum(axis=0) / (n + L2)
        return
    tree.feat[node] = best_f
    tree.thresh[node] = best_t
    left_mask = X[idxs, best_f] <= best_t
    l_idx = idxs[left_mask]
    r_idx = idxs[~left_mask]
    tree.left[node] = tree.n_nodes
    tree.n_nodes += 1
    build_tree(X, g, h, l_idx, depth+1, rng, max_depth, tree.left[node], tree)
    tree.right[node] = tree.n_nodes
    tree.n_nodes += 1
    build_tree(X, g, h, r_idx, depth+1, rng, max_depth, tree.right[node], tree)

def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X_raw.shape[1], card["classes"]
    n = len(y)
    # z-score per pixel
    mu = X_raw.mean(axis=0).astype(np.float32)
    sd = X_raw.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X = ((X_raw.astype(np.float32) - mu) / sd)
    # validation split (last 20%)
    val_start = int(n * 0.8)
    idx_tr = np.arange(val_start)
    idx_val = np.arange(val_start, n)
    rng = np.random.default_rng(SEED)
    # init logits
    logits = np.zeros((n, C), np.float32)
    trees = []
    for _ in range(N_TREES):
        tree = Tree(2**(MAX_DEPTH+1))
        g = softmax_grad(logits[idx_tr], y[idx_tr])
        build_tree(X, g, None, idx_tr, 0, rng, MAX_DEPTH, 0, tree)
        # apply to all
        for i in range(n):
            node = 0
            while not tree.is_leaf[node]:
                node = tree.left[node] if X[i, tree.feat[node]] <= tree.thresh[node] else tree.right[node]
            logits[i] += LR * tree.leaf_val[node]
        trees.append(tree)
    # serialize
    max_nodes_per = 2**(MAX_DEPTH+1)
    total_nodes = sum(t.n_nodes for t in trees)
    out = struct.pack("<5i", 0x47424454, d, C, N_TREES, MAX_DEPTH)
    out += mu.astype("<f4").tobytes() + sd.astype("<f4").tobytes()
    # nodes
    for t in trees:
        out += t.feat[:t.n_nodes].astype("<i2").tobytes()
        out += t.thresh[:t.n_nodes].astype("<f4").tobytes()
        out += t.left[:t.n_nodes].astype("<i2").tobytes()
        out += t.right[:t.n_nodes].astype("<i2").tobytes()
        out += t.is_leaf[:t.n_nodes].astype("b").tobytes()
    # leaves (only for leaf nodes)
    for t in trees:
        for i in range(t.n_nodes):
            if t.is_leaf[i]:
                out += t.leaf_val[i].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-tinygbdt inference: ./infer weights.bin in.rec N
 * Weights format (little-endian):
 *   int32 magic=0x47424454, d, C, n_trees, max_depth
 *   float32 mu[d], sd[d]
 *   per tree: int16 feat[n_nodes], float32 thresh[n_nodes],
 *             int16 left[n_nodes], int16 right[n_nodes], int8 is_leaf[n_nodes]
 *   then per tree: for each leaf node (in node order), float32 leaf_val[C] */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hdr = (int32_t*)wb;
    if (nw < 20 || hdr[0] != 0x47424454) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hdr[1], C = hdr[2], n_trees = hdr[3], max_depth = hdr[4];
    if (d != 64 || C != 10 || n_trees <= 0 || max_depth <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    float *mu = (float*)(hdr + 5);
    float *sd = mu + d;
    unsigned char *p = (unsigned char*)(sd + d);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = calloc(C, sizeof(float));
    float *xin = malloc(d * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xin[j] = (xi[j] - mu[j]) / sd[j];
        memset(scores, 0, C * sizeof(float));
        for (int t = 0; t < n_trees; t++) {
            int16_t *feat = (int16_t*)p; p += 2 * (1<<(max_depth+1));
            float *thresh = (float*)p; p += 4 * (1<<(max_depth+1));
            int16_t *left = (int16_t*)p; p += 2 * (1<<(max_depth+1));
            int16_t *right = (int16_t*)p; p += 2 * (1<<(max_depth+1));
            int8_t *is_leaf = (int8_t*)p; p += 1 * (1<<(max_depth+1));
            int node = 0;
            while (!is_leaf[node]) {
                int f = feat[node];
                node = (xin[f] <= thresh[node]) ? left[node] : right[node];
            }
            float *leaf = (float*)p; p += C * 4;
            for (int c = 0; c < C; c++) scores[c] += leaf[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```