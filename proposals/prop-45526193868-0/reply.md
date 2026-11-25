```hypothesis
Family: label-tree (shallow decision tree).  
Why: The adult dataset has mixed ordinal/categorical features with known interactions (e.g., education × occupation). A depth‑5 tree captures non‑linearities that a linear model misses, while inference stays extremely fast (≤5 comparisons per record).  
Failure signal: dev accuracy ≤ 0.8524 (label‑linear) or instructions/item ≥ 1000.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split (last 20% of train.in)
MAX_DEPTHS = [3, 4, 5, 6, 7]
MIN_LEAVES = [1, 5, 10, 20, 50]

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
    c0 = np.count_nonzero(y == 0)
    c1 = len(y) - c0
    p0 = c0 / len(y)
    p1 = c1 / len(y)
    return 1.0 - p0*p0 - p1*p1

def best_split(X, y, types, min_leaf):
    n, d = X.shape
    best_gain = -1.0
    best_feat = -1
    best_type = -1   # 0: ordinal <= thr, 1: categorical == val
    best_val = -1
    best_left_mask = None
    parent_gini = gini(y)
    for j in range(d):
        vals = X[:, j]
        if types[j] == "ordinal":
            # thresholds 0..254
            for thr in range(255):
                left = vals <= thr
                right = ~left
                if left.sum() < min_leaf or right.sum() < min_leaf:
                    continue
                gain = parent_gini - (left.sum()*gini(y[left]) + right.sum()*gini(y[right])) / n
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feat = j
                    best_type = 0
                    best_val = thr
                    best_left_mask = left
        else:  # categorical
            # split on equality to a single value
            for v in range(256):
                left = vals == v
                right = ~left
                if left.sum() < min_leaf or right.sum() < min_leaf:
                    continue
                gain = parent_gini - (left.sum()*gini(y[left]) + right.sum()*gini(y[right])) / n
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feat = j
                    best_type = 1
                    best_val = v
                    best_left_mask = left
    if best_gain <= 1e-12:
        return None
    return best_feat, best_type, best_val, best_left_mask

def build_tree(X, y, types, max_depth, min_leaf, depth=0, node_list=None):
    if node_list is None:
        node_list = []
    # leaf condition
    if depth >= max_depth or len(y) < 2*min_leaf or gini(y) == 0.0:
        leaf_class = 0 if np.count_nonzero(y == 0) >= np.count_nonzero(y == 1) else 1
        node_list.append(('leaf', leaf_class))
        return len(node_list) - 1
    split = best_split(X, y, types, min_leaf)
    if split is None:
        leaf_class = 0 if np.count_nonzero(y == 0) >= np.count_nonzero(y == 1) else 1
        node_list.append(('leaf', leaf_class))
        return len(node_list) - 1
    feat, stype, val, left_mask = split
    right_mask = ~left_mask
    left_idx = build_tree(X[left_mask], y[left_mask], types, max_depth, min_leaf, depth+1, node_list)
    right_idx = build_tree(X[right_mask], y[right_mask], types, max_depth, min_leaf, depth+1, node_list)
    node_list.append(('node', feat, stype, val, left_idx, right_idx))
    return len(node_list) - 1

def predict_tree(node_list, root_idx, x):
    idx = root_idx
    while True:
        node = node_list[idx]
        if node[0] == 'leaf':
            return node[1]
        _, feat, stype, val, left_idx, right_idx = node
        if stype == 0:  # ordinal <=
            if x[feat] <= val:
                idx = left_idx
            else:
                idx = right_idx
        else:  # categorical ==
            if x[feat] == val:
                idx = left_idx
            else:
                idx = right_idx

def accuracy(tree, root, X, y):
    correct = 0
    for i in range(len(y)):
        if predict_tree(tree, root, X[i]) == y[i]:
            correct += 1
    return correct / len(y)

def write_weights(node_list, d, max_depth, out_path):
    magic = 0x54524545  # "TREE" in little-endian
    num_nodes = len(node_list)
    # First pass: compute size
    # Header: 4 int32 (magic, d, max_depth, num_nodes)
    # Each node: if leaf -> int32 is_leaf=1, int32 class
    #            else -> int32 is_leaf=0, uint8 feat, uint8 type, uint8 val, int32 left, int32 right
    # We'll pack sequentially.
    buf = bytearray()
    buf += struct.pack("<iiii", magic, d, max_depth, num_nodes)
    for node in node_list:
        if node[0] == 'leaf':
            buf += struct.pack("<ii", 1, node[1])
        else:
            _, feat, stype, val, left, right = node
            buf += struct.pack("<iBBBii", 0, feat, stype, val, left, right)
    open(out_path, "wb").write(buf)

def main():
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc = -1.0
    best_params = None
    best_tree = None
    best_root = None

    for max_depth in MAX_DEPTHS:
        for min_leaf in MIN_LEAVES:
            tree = []
            root = build_tree(X_train, y_train, types, max_depth, min_leaf, node_list=tree)
            acc = accuracy(tree, root, X_val, y_val)
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_tree = tree
                best_root = root

    # Retrain on full data with best params
    max_depth, min_leaf = best_params
    final_tree = []
    final_root = build_tree(X, y, types, max_depth, min_leaf, node_list=final_tree)
    write_weights(final_tree, d, max_depth, sys.argv[4])

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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], max_depth = hd[2], num_nodes = hd[3];
    if (d != 14 || num_nodes <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }

    /* Parse nodes */
    unsigned char *ptr = wb + 16;
    unsigned char *end = wb + nw;
    int32_t *is_leaf = malloc(num_nodes * sizeof(int32_t));
    int32_t *leaf_class = malloc(num_nodes * sizeof(int32_t));
    uint8_t *feat = malloc(num_nodes * sizeof(uint8_t));
    uint8_st *stype = malloc(num_nodes * sizeof(uint8_t));
    uint8_t *val = malloc(num_nodes * sizeof(uint8_t));
    int32_t *left = malloc(num_nodes * sizeof(int32_t));
    int32_t *right = malloc(num_nodes * sizeof(int32_t));

    for (int i = 0; i < num_nodes; i++) {
        if (ptr + 4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        int32_t leaf = *(int32_t *)ptr; ptr += 4;
        is_leaf[i] = leaf;
        if (leaf) {
            if (ptr + 4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
            leaf_class[i] = *(int32_t *)ptr; ptr += 4;
            feat[i] = stype[i] = val[i] = 0;
            left[i] = right[i] = -1;
        } else {
            if (ptr + 1+1+1+4+4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
            feat[i] = *ptr++; stype[i] = *ptr++; val[i] = *ptr++;
            left[i] = *(int32_t *)ptr; ptr += 4;
            right[i] = *(int32_t *)ptr; ptr += 4;
            leaf_class[i] = -1;
        }
    }
    if (ptr != end) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int idx = 0;
        while (!is_leaf[idx]) {
            uint8_t f = feat[idx];
            if (stype[idx] == 0) { /* ordinal <= */
                if (xi[f] <= val[idx]) idx = left[idx];
                else idx = right[idx];
            } else { /* categorical == */
                if (xi[f] == val[idx]) idx = left[idx];
                else idx = right[idx];
            }
        }
        unsigned char out = (unsigned char)leaf_class[idx];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```