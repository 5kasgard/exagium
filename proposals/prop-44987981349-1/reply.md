```hypothesis
A single CART decision tree (depth ≤ 10) on raw 0-16 pixel values captures non-linear interactions
and should exceed label-linear's 0.9613 accuracy. Inference is a handful of integer comparisons
per sample, so CPU cost stays far below label-linear's 5058 Ir/item. If dev accuracy ≤ 0.9613,
the tree is too weak for this task.
```
```python
"""label-tree: single CART decision tree for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, struct, sys
import numpy as np

SEED = 1
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(y):
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    return 1.0 - np.sum(p * p)

def best_split(X, y, rng):
    n, d = X.shape
    best_gini = 1.0
    best_feat = -1
    best_thr = -1
    for feat in range(d):
        vals = np.unique(X[:, feat])
        if len(vals) <= 1:
            continue
        for thr in vals[:-1]:
            left = y[X[:, feat] <= thr]
            right = y[X[:, feat] > thr]
            if len(left) < MIN_SAMPLES_LEAF or len(right) < MIN_SAMPLES_LEAF:
                continue
            g = (len(left) * gini(left) + len(right) * gini(right)) / n
            if g < best_gini:
                best_gini, best_feat, best_thr = g, feat, thr
    return best_feat, best_thr, best_gini

def build_tree(X, y, depth, rng):
    n = len(y)
    if depth >= MAX_DEPTH or n < MIN_SAMPLES_SPLIT or gini(y) == 0.0:
        return -1, -1, -1, -1, int(np.bincount(y).argmax())
    feat, thr, _ = best_split(X, y, rng)
    if feat == -1:
        return -1, -1, -1, -1, int(np.bincount(y).argmax())
    left_idx = X[:, feat] <= thr
    right_idx = X[:, feat] > thr
    left = build_tree(X[left_idx], y[left_idx], depth + 1, rng)
    right = build_tree(X[right_idx], y[right_idx], depth + 1, rng)
    return feat, int(thr), left, right, -1

def flatten_tree(node, nodes):
    """node = (feat, thr, left, right, cls). Returns index of this node in nodes."""
    feat, thr, left, right, cls = node
    idx = len(nodes)
    nodes.append([feat, thr, 0, 0, cls])  # left/right filled later
    if feat != -1:
        left_idx = flatten_tree(left, nodes)
        right_idx = flatten_tree(right, nodes)
        nodes[idx][2] = left_idx
        nodes[idx][3] = right_idx
    return idx

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    root = build_tree(X, y, 0, rng)
    nodes = []
    flatten_tree(root, nodes)
    # header: magic, num_nodes, num_classes, reserved
    hdr = struct.pack("<4i", 0x54524545, len(nodes), card["classes"], 0)
    body = b"".join(struct.pack("<5i", *n) for n in nodes)
    open(sys.argv[4], "wb").write(hdr + body)

if __name__ == "__main__":
    main()
```
```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x54524545) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t num_nodes = hd[1], C = hd[2];
    if (C < 1 || C > 256 || nw != 16 + (long)num_nodes * 20) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    int32_t (*nodes)[5] = (int32_t (*)[5])(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int idx = 0;
        while (1) {
            int32_t feat = nodes[idx][0];
            if (feat == -1) { put_rec(stdout, (unsigned char *)&nodes[idx][4], 1); break; }
            int32_t thr = nodes[idx][1];
            if (xi[feat] <= thr) idx = nodes[idx][2];
            else idx = nodes[idx][3];
        }
    }
    return 0;
}
```