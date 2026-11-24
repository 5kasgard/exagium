```hypothesis
Family: label-tree. A single deterministic decision tree (CART) with Gini impurity, tuned max_depth and min_samples_split on the last 20% of train. Trees capture non-linear interactions in the 8x8 pixel grid better than linear models, while inference is just a few integer comparisons (depth ~10-20), so CPU cost should be similar to linear. If dev accuracy does not exceed label-linear (0.9613), the hypothesis fails.
```

```python
import json, struct, sys
import numpy as np

SEED = 1
np.random.seed(SEED)
sys.setrecursionlimit(10000)

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, n_classes):
    n, d = X.shape
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    for feat in range(d):
        vals = np.unique(X[:, feat])
        for v in vals:
            mask = X[:, feat] <= v
            left_n = mask.sum()
            right_n = n - left_n
            if left_n == 0 or right_n == 0:
                continue
            g = (left_n * gini(y[mask]) + right_n * gini(y[~mask])) / n
            if g < best_gini or (g == best_gini and (feat < best_feat or (feat == best_feat and v < best_thresh))):
                best_gini = g
                best_feat = feat
                best_thresh = v
    return best_feat, best_thresh, best_gini

def build_tree(X, y, depth, max_depth, min_samples_split, n_classes):
    n = X.shape[0]
    if depth == max_depth or n < min_samples_split or len(np.unique(y)) == 1:
        counts = np.bincount(y, minlength=n_classes)
        return {'leaf': True, 'class': int(np.argmax(counts))}
    feat, thresh, _ = best_split(X, y, n_classes)
    if feat == -1:
        counts = np.bincount(y, minlength=n_classes)
        return {'leaf': True, 'class': int(np.argmax(counts))}
    mask = X[:, feat] <= thresh
    left = build_tree(X[mask], y[mask], depth+1, max_depth, min_samples_split, n_classes)
    right = build_tree(X[~mask], y[~mask], depth+1, max_depth, min_samples_split, n_classes)
    return {'leaf': False, 'feat': feat, 'thresh': thresh, 'left': left, 'right': right}

def tree_to_arrays(node, arrays):
    idx = len(arrays['feat'])
    if node['leaf']:
        arrays['feat'].append(-1)
        arrays['thresh'].append(0)
        arrays['left'].append(-1)
        arrays['right'].append(-1)
        arrays['class'].append(node['class'])
    else:
        arrays['feat'].append(node['feat'])
        arrays['thresh'].append(node['thresh'])
        left_idx = tree_to_arrays(node['left'], arrays)
        right_idx = tree_to_arrays(node['right'], arrays)
        arrays['left'].append(left_idx)
        arrays['right'].append(right_idx)
        arrays['class'].append(0)
    return idx

def evaluate_tree(X, y, arrays):
    n = X.shape[0]
    pred = np.zeros(n, dtype=np.int64)
    for i in range(n):
        node = 0
        while arrays['feat'][node] != -1:
            if X[i, arrays['feat'][node]] <= arrays['thresh'][node]:
                node = arrays['left'][node]
            else:
                node = arrays['right'][node]
        pred[i] = arrays['class'][node]
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    n_classes = card["classes"]

    max_depths = [5, 10, 15, 20, 25, 30, None]
    min_samples_splits = [2, 5, 10]
    best_acc = -1.0
    best_params = None
    best_arrays = None

    for md in max_depths:
        for mss in min_samples_splits:
            tree = build_tree(X_train, y_train, 0, md if md is not None else 1000, mss, n_classes)
            arrays = {'feat':[], 'thresh':[], 'left':[], 'right':[], 'class':[]}
            tree_to_arrays(tree, arrays)
            acc = evaluate_tree(X_val, y_val, arrays)
            if acc > best_acc:
                best_acc = acc
                best_params = (md, mss)
                best_arrays = arrays

    md, mss = best_params
    tree = build_tree(X, y, 0, md if md is not None else 1000, mss, n_classes)
    arrays = {'feat':[], 'thresh':[], 'left':[], 'right':[], 'class':[]}
    tree_to_arrays(tree, arrays)

    magic = 0x54524545  # "TREE"
    n_nodes = len(arrays['feat'])
    out = struct.pack("<4I", magic, n_nodes, X.shape[1], n_classes)
    for name in ['feat', 'thresh', 'left', 'right', 'class']:
        out += np.array(arrays[name], dtype=np.int32).tobytes()
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x54524545) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t n_nodes = hd[1];
    uint32_t n_features = hd[2];
    uint32_t n_classes = hd[3];
    if (n_features != 64 || n_classes != 10) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    if (nw != 16 + 20 * n_nodes) { fprintf(stderr, "bad weights size\n"); return 3; }
    int32_t *feat = (int32_t *)(hd + 4);
    int32_t *thresh = feat + n_nodes;
    int32_t *left = thresh + n_nodes;
    int32_t *right = left + n_nodes;
    int32_t *class_arr = right + n_nodes;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, n_features); return 3; }
        uint32_t node = 0;
        while (feat[node] != -1) {
            if (xi[feat[node]] <= (unsigned char)thresh[node])
                node = left[node];
            else
                node = right[node];
        }
        unsigned char best = (unsigned char)class_arr[node];
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```