```hypothesis
Family: label-tree (single CART decision tree). Optdigits' 8x8 spatial structure creates pixel interactions that linear models miss; a depth-10 tree captures these with ~1000 nodes. Inference is a few dozen comparisons per item (<< linear's 64*10 adds). If dev accuracy < 0.96 or Ir/item > label-linear, the hypothesis fails.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
TUNING_FRACTION = 0.2

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

def best_split(X, y, indices, rng):
    n = len(indices)
    if n == 0:
        return None, None, None
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1
    parent_gini = gini(y[indices])
    for feat in range(X.shape[1]):
        vals = X[indices, feat]
        for thresh in range(16):
            left_mask = vals <= thresh
            n_left = np.sum(left_mask)
            n_right = n - n_left
            if n_left == 0 or n_right == 0:
                continue
            gain = parent_gini - (n_left/n)*gini(y[indices[left_mask]]) - (n_right/n)*gini(y[indices[~left_mask]])
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh
            elif abs(gain - best_gain) < 1e-12:
                if feat < best_feat or (feat == best_feat and thresh < best_thresh):
                    best_feat = feat
                    best_thresh = thresh
    if best_gain <= 1e-12:
        return None, None, None
    return best_feat, best_thresh, best_gain

def build_tree(X, y, indices, depth, max_depth, min_samples_leaf, rng, nodes):
    n = len(indices)
    if depth >= max_depth or n < min_samples_leaf:
        classes, counts = np.unique(y[indices], return_counts=True)
        leaf_class = classes[np.argmax(counts)]
        node_idx = len(nodes)
        nodes.append((-1, 0, -1, -1, int(leaf_class)))
        return node_idx
    feat, thresh, gain = best_split(X, y, indices, rng)
    if feat is None:
        classes, counts = np.unique(y[indices], return_counts=True)
        leaf_class = classes[np.argmax(counts)]
        node_idx = len(nodes)
        nodes.append((-1, 0, -1, -1, int(leaf_class)))
        return node_idx
    left_mask = X[indices, feat] <= thresh
    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]
    left_child = build_tree(X, y, left_indices, depth+1, max_depth, min_samples_leaf, rng, nodes)
    right_child = build_tree(X, y, right_indices, depth+1, max_depth, min_samples_leaf, rng, nodes)
    node_idx = len(nodes)
    nodes.append((feat, thresh, left_child, right_child, 0))
    return node_idx

def evaluate_tree(X, y, indices, nodes):
    correct = 0
    for idx in indices:
        node = 0
        while True:
            feat, thresh, left, right, cls = nodes[node]
            if feat == -1:
                if cls == y[idx]:
                    correct += 1
                break
            if X[idx, feat] <= thresh:
                node = left
            else:
                node = right
    return correct / len(indices)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    d = X.shape[1]
    split = int(n * (1 - TUNING_FRACTION))
    train_idx = np.arange(split)
    tune_idx = np.arange(split, n)
    
    rng = np.random.default_rng(SEED)
    
    max_depths = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    min_samples_leafs = [1, 2, 5, 10, 20]
    
    best_acc = -1
    best_params = None
    best_nodes = None
    
    for max_depth in max_depths:
        for min_samples_leaf in min_samples_leafs:
            nodes = []
            build_tree(X, y, train_idx, 0, max_depth, min_samples_leaf, rng, nodes)
            acc = evaluate_tree(X, y, tune_idx, nodes)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)
                best_nodes = nodes
    
    max_depth, min_samples_leaf = best_params
    rng = np.random.default_rng(SEED)
    nodes = []
    build_tree(X, y, np.arange(n), 0, max_depth, min_samples_leaf, rng, nodes)
    
    magic = 0x54524545
    num_nodes = len(nodes)
    out = struct.pack("<3i", magic, num_nodes, d)
    for feat, thresh, left, right, cls in nodes:
        out += struct.pack("<5i", feat, thresh, left, right, cls)
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x54524545) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t num_nodes = hd[1];
    int32_t d = hd[2];
    if (nw != 12 + (long)num_nodes * 20) { fprintf(stderr, "bad weights size\n"); return 3; }
    int32_t *nodes = hd + 3;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int32_t node_idx = 0;
        while (1) {
            int32_t feat = nodes[node_idx * 5 + 0];
            int32_t thresh = nodes[node_idx * 5 + 1];
            int32_t left = nodes[node_idx * 5 + 2];
            int32_t right = nodes[node_idx * 5 + 3];
            int32_t cls = nodes[node_idx * 5 + 4];
            if (feat == -1) {
                unsigned char best = (unsigned char)cls;
                put_rec(stdout, &best, 1);
                break;
            }
            if (xi[feat] <= thresh) {
                node_idx = left;
            } else {
                node_idx = right;
            }
        }
    }
    return 0;
}
```