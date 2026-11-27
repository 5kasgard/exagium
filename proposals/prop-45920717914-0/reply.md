```hypothesis
Family: label-tree (small deterministic decision tree, max depth 5, min leaf 10).
Why: Adult has strong non-linear feature interactions (e.g., education × occupation × hours) that a linear model misses. A shallow tree captures these with few splits, keeping inference extremely fast (≤5 comparisons per item). It should beat label-linear's 0.8524 accuracy while staying well under its CPU cost.
Failure signal: dev accuracy ≤ 0.8524 or inference cost ≥ label-linear.
```
```python
import json, struct, sys
import numpy as np

MAX_DEPTH = 5
MIN_LEAF = 10
SEED = 1
MAGIC = 0x54524545  # 'TREE' little-endian
NODE_FMT = '<BBBBiiii'
NODE_SZ = struct.calcsize(NODE_FMT)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def build_tree(X, y, indices, depth, nodes):
    n = len(indices)
    if depth >= MAX_DEPTH or n < 2 * MIN_LEAF:
        return None
    y_node = y[indices]
    c0 = np.count_nonzero(y_node == 0)
    c1 = n - c0
    if c0 == 0 or c1 == 0:
        return None
    parent_gini = 1.0 - (c0/n)**2 - (c1/n)**2
    best_gain = 0.0
    best_f = -1
    best_t = -1
    for f in range(X.shape[1]):
        vals = X[indices, f]
        hist0 = np.bincount(vals[y_node==0], minlength=256).astype(np.float64)
        hist1 = np.bincount(vals[y_node==1], minlength=256).astype(np.float64)
        cum0 = np.cumsum(hist0)
        cum1 = np.cumsum(hist1)
        n_left = cum0[:255] + cum1[:255]
        n_right = n - n_left
        mask = (n_left >= MIN_LEAF) & (n_right >= MIN_LEAF)
        if not np.any(mask):
            continue
        p0l = cum0[:255] / n_left
        p1l = cum1[:255] / n_left
        gini_l = 1 - p0l**2 - p1l**2
        p0r = (c0 - cum0[:255]) / n_right
        p1r = (c1 - cum1[:255]) / n_right
        gini_r = 1 - p0r**2 - p1r**2
        gain = parent_gini - (n_left/n)*gini_l - (n_right/n)*gini_r
        gain[~mask] = -1.0
        idx = int(np.argmax(gain))
        if gain[idx] > best_gain:
            best_gain, best_f, best_t = gain[idx], f, idx
    if best_gain <= 0:
        return None
    left_idx = [i for i in indices if X[i, best_f] <= best_t]
    right_idx = [i for i in indices if X[i, best_f] > best_t]
    return best_f, best_t, left_idx, right_idx

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    # deterministic tie-breaking: no randomness used in split selection
    nodes = [{'leaf': True, 'class': 0, 'feature': 0, 'thr': 0, 'left': -1, 'right': -1}]  # root placeholder
    queue = [(0, list(range(n)), 0)]
    while queue:
        node_idx, idxs, depth = queue.pop(0)
        split = build_tree(X, y, idxs, depth, nodes)
        if split is None:
            # make leaf
            c0 = np.count_nonzero(y[idxs] == 0)
            c1 = len(idxs) - c0
            nodes[node_idx] = {'leaf': True, 'class': 0 if c0 >= c1 else 1,
                               'feature': 0, 'thr': 0, 'left': -1, 'right': -1}
            continue
        f, t, left_idxs, right_idxs = split
        left_node = len(nodes)
        right_node = len(nodes) + 1
        nodes.append({'leaf': True, 'class': 0, 'feature': 0, 'thr': 0, 'left': -1, 'right': -1})
        nodes.append({'leaf': True, 'class': 0, 'feature': 0, 'thr': 0, 'left': -1, 'right': -1})
        nodes[node_idx] = {'leaf': False, 'class': 0, 'feature': f, 'thr': t,
                           'left': left_node, 'right': right_node}
        queue.append((left_node, left_idxs, depth+1))
        queue.append((right_node, right_idxs, depth+1))
    # write weights
    out = struct.pack('<II', MAGIC, len(nodes))
    for nd in nodes:
        if nd['leaf']:
            out += struct.pack(NODE_FMT, 0, nd['class'], 0, 0, -1, -1, 0, 0)
        else:
            out += struct.pack(NODE_FMT, 1, nd['feature'], nd['thr'], 0, nd['left'], nd['right'], 0, 0)
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

typedef struct {
    uint8_t type;          // 0 leaf, 1 internal
    uint8_t class_or_feat; // class if leaf, feature if internal
    uint8_t threshold;     // 0 if leaf, split threshold if internal
    uint8_t pad;
    int32_t left;          // -1 if leaf, else left child index
    int32_t right;         // -1 if leaf, else right child index
    int32_t pad2[2];
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t n_nodes = hdr[1];
    long expected = 8 + (long)n_nodes * sizeof(Node);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 8);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }
        uint32_t node = 0;
        while (1) {
            Node nd = nodes[node];
            if (nd.type == 0) {
                unsigned char cls = nd.class_or_feat;
                put_rec(stdout, &cls, 1);
                break;
            }
            uint8_t feat = nd.class_or_feat;
            uint8_t thr  = nd.threshold;
            if (xi[feat] <= thr) node = nd.left;
            else node = nd.right;
        }
    }
    return 0;
}
```