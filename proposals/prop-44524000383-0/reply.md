```hypothesis
Family: label-rf (small Random Forest). Adult has non-linear feature interactions (e.g., education×occupation) that linear models miss. A forest of 20 trees (max depth 5, 4 features per split) on the native 14 quantized columns should capture these at near-linear inference cost (~1000 Ir/item). Failure: dev accuracy ≤ 0.8524 or Ir/item > 2000.
```
```python
import json, struct, sys
import numpy as np

SEED = 1
N_TREES = 20
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 10
MIN_SAMPLES_SPLIT = 20
MAX_FEATURES = 4

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def gini(y):
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, feature_indices, feature_types):
    best_gain = -1.0
    best_feat = -1
    best_thresh = 0
    parent_gini = gini(y)
    n = len(y)
    for feat in feature_indices:
        vals = X[:, feat]
        if feature_types[feat] == 0:  # ordinal
            uniq = np.unique(vals)
            if len(uniq) <= 1: continue
            thresholds = (uniq[:-1] + uniq[1:]) // 2
            for t in thresholds:
                left = vals <= t
                right = ~left
                if left.sum() < MIN_SAMPLES_LEAF or right.sum() < MIN_SAMPLES_LEAF:
                    continue
                gain = parent_gini - (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / n
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t
        else:  # categorical
            uniq = np.unique(vals)
            for v in uniq:
                left = vals == v
                right = ~left
                if left.sum() < MIN_SAMPLES_LEAF or right.sum() < MIN_SAMPLES_LEAF:
                    continue
                gain = parent_gini - (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / n
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, v
    return best_feat, best_thresh, best_gain

def build_tree(X, y, feature_types, depth, rng):
    n = len(y)
    if depth >= MAX_DEPTH or n < MIN_SAMPLES_SPLIT or gini(y) == 0:
        classes, counts = np.unique(y, return_counts=True)
        return {'leaf': True, 'class': int(classes[np.argmax(counts)])}
    all_feats = np.arange(X.shape[1])
    rng.shuffle(all_feats)
    feature_indices = all_feats[:MAX_FEATURES]
    feat, thresh, gain = best_split(X, y, feature_indices, feature_types)
    if feat == -1 or gain <= 0:
        classes, counts = np.unique(y, return_counts=True)
        return {'leaf': True, 'class': int(classes[np.argmax(counts)])}
    if feature_types[feat] == 0:
        left_mask = X[:, feat] <= thresh
    else:
        left_mask = X[:, feat] == thresh
    right_mask = ~left_mask
    left_tree = build_tree(X[left_mask], y[left_mask], feature_types, depth+1, rng)
    right_tree = build_tree(X[right_mask], y[right_mask], feature_types, depth+1, rng)
    return {'leaf': False, 'feature': int(feat), 'threshold': int(thresh), 'left': left_tree, 'right': right_tree}

def tree_to_arrays(tree, nodes):
    idx = len(nodes['feature'])
    if tree['leaf']:
        nodes['feature'].append(255)
        nodes['threshold'].append(0)
        nodes['left'].append(-1)
        nodes['right'].append(-1)
        nodes['leaf_class'].append(tree['class'])
    else:
        left_idx = tree_to_arrays(tree['left'], nodes)
        right_idx = tree_to_arrays(tree['right'], nodes)
        nodes['feature'].append(tree['feature'])
        nodes['threshold'].append(tree['threshold'])
        nodes['left'].append(left_idx)
        nodes['right'].append(right_idx)
        nodes['leaf_class'].append(255)
    return idx

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    d = card["input"]["bytes"]
    feature_types = np.array([0 if t == "ordinal" else 1 for t in card["input"]["types"]], dtype=np.uint8)
    C = card["classes"]
    rng = np.random.default_rng(SEED)
    trees_nodes = []
    tree_node_counts = []
    for _ in range(N_TREES):
        idx = rng.integers(0, len(X), len(X))
        Xb, yb = X[idx], y[idx]
        tree = build_tree(Xb, yb, feature_types, 0, rng)
        nodes = {'feature': [], 'threshold': [], 'left': [], 'right': [], 'leaf_class': []}
        tree_to_arrays(tree, nodes)
        trees_nodes.append(nodes)
        tree_node_counts.append(len(nodes['feature']))
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<IIIII", 0x52464553, N_TREES, d, C, 0))
        f.write(feature_types.tobytes())
        for i in range(N_TREES):
            nodes = trees_nodes[i]
            n_nodes = tree_node_counts[i]
            f.write(struct.pack("<I", n_nodes))
            for j in range(n_nodes):
                f.write(struct.pack("<BBiiiB",
                    nodes['feature'][j],
                    nodes['threshold'][j],
                    nodes['left'][j],
                    nodes['right'][j],
                    nodes['leaf_class'][j]))

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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    const unsigned char *ptr = wb;
    long rem = nw;
    #define READ(dst, sz) do { if (rem < (long)sz) { fprintf(stderr, "weights truncated\n"); return 3; } memcpy(dst, ptr, sz); ptr += sz; rem -= sz; } while(0)
    uint32_t magic, n_trees, d, C, reserved;
    READ(&magic, 4); READ(&n_trees, 4); READ(&d, 4); READ(&C, 4); READ(&reserved, 4);
    if (magic != 0x52464553) { fprintf(stderr, "bad weights magic\n"); return 3; }
    if (C != 2) { fprintf(stderr, "expected 2 classes\n"); return 3; }
    if (rem < (long)d) { fprintf(stderr, "weights truncated feature_types\n"); return 3; }
    uint8_t *feature_types = malloc(d);
    memcpy(feature_types, ptr, d); ptr += d; rem -= d;
    typedef struct { uint8_t feature, threshold, leaf_class; int32_t left, right; } Node;
    Node **trees = malloc(n_trees * sizeof(Node*));
    uint32_t *tree_n_nodes = malloc(n_trees * sizeof(uint32_t));
    for (uint32_t t = 0; t < n_trees; t++) {
        uint32_t n_nodes; READ(&n_nodes, 4);
        tree_n_nodes[t] = n_nodes;
        trees[t] = malloc(n_nodes * sizeof(Node));
        for (uint32_t i = 0; i < n_nodes; i++) {
            Node node;
            READ(&node.feature, 1);
            READ(&node.threshold, 1);
            READ(&node.left, 4);
            READ(&node.right, 4);
            READ(&node.leaf_class, 1);
            trees[t][i] = node;
        }
    }
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        int votes[2] = {0, 0};
        for (uint32_t t = 0; t < n_trees; t++) {
            uint32_t node_idx = 0;
            while (1) {
                Node node = trees[t][node_idx];
                if (node.left == -1) { votes[node.leaf_class]++; break; }
                uint8_t feat_val = xi[node.feature];
                int go_left = (feature_types[node.feature] == 0) ? (feat_val <= node.threshold) : (feat_val == node.threshold);
                node_idx = go_left ? node.left : node.right;
            }
        }
        unsigned char best = (votes[1] > votes[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```