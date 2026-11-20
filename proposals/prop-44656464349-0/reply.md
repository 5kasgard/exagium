```hypothesis
Family: label-dtree (single deterministic decision tree). Why: Adult has mixed ordinal/categorical features; a shallow tree captures non-linear interactions that logistic regression misses. With max_depth 5-7 and min_samples_leaf tuned on the 20% tuning split, it should exceed label-linear's 0.8524 accuracy while inference stays extremely cheap (a handful of integer comparisons per record). Failure mode: if best tuned depth ≤3 or accuracy ≤0.8524 on dev, the family does not justify its existence.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
TUNE_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)

def best_split(X, y, min_leaf):
    n, d = X.shape
    best_gini = 1.0
    best_f = -1
    best_t = -1
    for f in range(d):
        vals = X[:, f]
        # histogram of class counts per value (0..255)
        hist = np.zeros((256, 2), dtype=np.int32)
        for v in range(256):
            mask = (vals == v)
            if np.any(mask):
                hist[v, 0] = np.sum(y[mask] == 0)
                hist[v, 1] = np.sum(y[mask] == 1)
        # cumulative sums
        cum = np.cumsum(hist, axis=0)
        total = cum[-1]
        for t in range(1, 256):
            left = cum[t-1]
            right = total - left
            if left.sum() < min_leaf or right.sum() < min_leaf:
                continue
            g = (left.sum() * gini_from_counts(left) + right.sum() * gini_from_counts(right)) / n
            if g < best_gini:
                best_gini = g
                best_f = f
                best_t = t
    return best_f, best_t, best_gini

def build_tree(X, y, depth, max_depth, min_leaf, min_split):
    n = len(y)
    if depth == max_depth or n < min_split or len(np.unique(y)) == 1:
        leaf_class = int(np.bincount(y).argmax())
        return {'leaf': True, 'class': leaf_class}
    f, t, _ = best_split(X, y, min_leaf)
    if f == -1:
        leaf_class = int(np.bincount(y).argmax())
        return {'leaf': True, 'class': leaf_class}
    left_mask = X[:, f] < t
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], depth+1, max_depth, min_leaf, min_split)
    right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth, min_leaf, min_split)
    return {'leaf': False, 'feature': f, 'threshold': t, 'left': left, 'right': right}

def flatten_tree(node, nodes):
    idx = len(nodes)
    if node['leaf']:
        nodes.append((1, node['class'], 0, 0, 0, 0))
    else:
        nodes.append((0, node['feature'], node['threshold'], 0, 0, 0))
        left_idx = flatten_tree(node['left'], nodes)
        right_idx = flatten_tree(node['right'], nodes)
        nodes[idx] = (0, node['feature'], node['threshold'], left_idx, right_idx, 0)
    return idx

def accuracy(tree_nodes, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        node = 0
        while True:
            is_leaf, feat, thresh, left, right, _ = tree_nodes[node]
            if is_leaf:
                if feat == yi:
                    correct += 1
                break
            if xi[feat] < thresh:
                node = left
            else:
                node = right
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    tune_size = int(n * TUNE_FRAC)
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    # hyperparameter grid
    best_acc = -1.0
    best_params = None
    for max_depth in (3, 4, 5, 6, 7):
        for min_leaf in (1, 5, 10, 20, 50):
            for min_split in (2, 10, 20):
                tree = build_tree(train_X, train_y, 0, max_depth, min_leaf, min_split)
                nodes = []
                flatten_tree(tree, nodes)
                acc = accuracy(nodes, tune_X, tune_y)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (max_depth, min_leaf, min_split)
    max_depth, min_leaf, min_split = best_params
    tree = build_tree(X, y, 0, max_depth, min_leaf, min_split)
    nodes = []
    flatten_tree(tree, nodes)

    # write weights.bin
    # magic 'DTRE' (0x45525444 little-endian), n_nodes, max_depth, feature_types[14]
    feat_types = bytearray(1 if t == "categorical" else 0 for t in card["input"]["types"])
    out = struct.pack("<Iii", 0x45525444, len(nodes), max_depth) + feat_types
    for nd in nodes:
        is_leaf, a, b, c, d, _ = nd
        if is_leaf:
            out += struct.pack("<BBBBii", 1, a, 0, 0, 0, 0)
        else:
            out += struct.pack("<BBBBii", 0, a, b, 0, c, d)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    np.random.seed(SEED)
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
    uint8_t is_leaf;
    uint8_t feature;
    uint8_t threshold;
    uint8_t pad;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 4 + 4 + 4 + 14) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x45525444) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_nodes = *(int32_t*)(wb + 4);
    int32_t max_depth = *(int32_t*)(wb + 8);
    unsigned char *feat_types = wb + 12;
    if (nw != 4 + 4 + 4 + 14 + (long)n_nodes * 16) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node*)(wb + 4 + 4 + 4 + 14);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }
        int node = 0;
        while (1) {
            Node nd = nodes[node];
            if (nd.is_leaf) {
                put_rec(stdout, &nd.feature, 1);
                break;
            }
            if (xi[nd.feature] < nd.threshold) node = nd.left;
            else node = nd.right;
        }
    }
    return 0;
}
```