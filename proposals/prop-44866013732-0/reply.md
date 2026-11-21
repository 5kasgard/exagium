```hypothesis
Family: label-tree — a single decision tree with target encoding for categorical features.
Why: The Adult dataset is known to favor tree-based models (label-gbdt is top). A single tree captures non-linear interactions at a fraction of GBDT's compute cost. Target encoding maps each category to P(y=1|category), turning categoricals into ordinals for simple threshold splits.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear's ~719 Ir/item.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
MAX_DEPTH_CANDIDATES = range(3, 9)  # 3..8

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
    p = y.mean()
    return 2.0 * p * (1.0 - p)

def build_tree(X, y, max_depth, min_samples_split, min_samples_leaf):
    n, d = X.shape
    nodes = []  # post-order list of node dicts

    def build(indices, depth):
        n_samples = len(indices)
        if n_samples < min_samples_split or depth == max_depth:
            leaf_class = 1 if y[indices].sum() > n_samples / 2 else 0
            nodes.append({'is_leaf': True, 'feature': 0, 'threshold': 0.0,
                          'left': -1, 'right': -1, 'leaf_class': leaf_class})
            return len(nodes) - 1

        parent_gini = gini(y[indices])
        best_gini = parent_gini
        best_feature = -1
        best_threshold = 0.0
        best_left_mask = None
        best_right_mask = None

        for j in range(d):
            vals = X[indices, j]
            unique_vals = np.unique(vals)
            if len(unique_vals) <= 1:
                continue
            for k in range(len(unique_vals) - 1):
                thresh = (unique_vals[k] + unique_vals[k + 1]) * 0.5
                left_mask = vals <= thresh
                right_mask = vals > thresh
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue
                gini_left = gini(y[indices[left_mask]])
                gini_right = gini(y[indices[right_mask]])
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = j
                    best_threshold = thresh
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        if best_feature == -1:
            leaf_class = 1 if y[indices].sum() > n_samples / 2 else 0
            nodes.append({'is_leaf': True, 'feature': 0, 'threshold': 0.0,
                          'left': -1, 'right': -1, 'leaf_class': leaf_class})
            return len(nodes) - 1

        left_idx = build(indices[best_left_mask], depth + 1)
        right_idx = build(indices[best_right_mask], depth + 1)
        nodes.append({'is_leaf': False, 'feature': best_feature, 'threshold': best_threshold,
                      'left': left_idx, 'right': right_idx, 'leaf_class': 0})
        return len(nodes) - 1

    root_idx = build(np.arange(n), 0)

    # Reorder to BFS (root first)
    from collections import deque
    new_nodes = []
    q = deque([root_idx])
    old_to_new = {}
    while q:
        old = q.popleft()
        node = nodes[old]
        new_idx = len(new_nodes)
        old_to_new[old] = new_idx
        if not node['is_leaf']:
            q.append(node['left'])
            q.append(node['right'])
        new_nodes.append(node)

    for node in new_nodes:
        if not node['is_leaf']:
            node['left'] = old_to_new[node['left']]
            node['right'] = old_to_new[node['right']]

    return new_nodes

def predict_tree(nodes, X):
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        node_idx = 0
        while True:
            node = nodes[node_idx]
            if node['is_leaf']:
                preds[i] = node['leaf_class']
                break
            if X[i, node['feature']] <= node['threshold']:
                node_idx = node['left']
            else:
                node_idx = node['right']
    return preds

def main():
    np.random.seed(SEED)
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    types = card["input"]["types"]
    cat_features = [i for i, t in enumerate(types) if t == "categorical"]

    # Tuning split: last 20%
    tune_size = n // 5
    X_train, X_tune = X[:-tune_size], X[-tune_size:]
    y_train, y_tune = y[:-tune_size], y[-tune_size:]

    best_depth = 3
    best_acc = -1.0
    for depth in MAX_DEPTH_CANDIDATES:
        # Build mapping on training portion
        map_table = np.zeros((d, 256), dtype=np.float32)
        for j in range(d):
            if j in cat_features:
                global_mean = y_train.mean()
                for v in range(256):
                    mask = (X_train[:, j] == v)
                    map_table[j, v] = y_train[mask].mean() if mask.any() else global_mean
            else:
                map_table[j, :] = np.arange(256, dtype=np.float32)

        X_train_trans = map_table[np.arange(d), X_train.T].T
        tree = build_tree(X_train_trans, y_train, depth, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF)
        X_tune_trans = map_table[np.arange(d), X_tune.T].T
        preds = predict_tree(tree, X_tune_trans)
        acc = (preds == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # Final training on all data
    map_table = np.zeros((d, 256), dtype=np.float32)
    for j in range(d):
        if j in cat_features:
            global_mean = y.mean()
            for v in range(256):
                mask = (X[:, j] == v)
                map_table[j, v] = y[mask].mean() if mask.any() else global_mean
        else:
            map_table[j, :] = np.arange(256, dtype=np.float32)

    X_trans = map_table[np.arange(d), X.T].T
    tree = build_tree(X_trans, y, best_depth, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF)

    # Serialize weights.bin
    # Magic 'TREE' = 0x54524545, d, C, max_depth, node_count, map_table[d][256] (float32),
    # then nodes: each node = uint8 is_leaf, uint8 feature, float32 threshold, int32 left, int32 right, uint8 leaf_class
    node_count = len(tree)
    magic = 0x54524545
    header = struct.pack('<I BB B I', magic, d, C, best_depth, node_count)
    map_bytes = map_table.astype('<f4').tobytes()
    node_bytes = bytearray()
    for node in tree:
        node_bytes += struct.pack('<B B f i i B',
                                  1 if node['is_leaf'] else 0,
                                  node['feature'],
                                  node['threshold'],
                                  node['left'],
                                  node['right'],
                                  node['leaf_class'])
    with open(sys.argv[4], 'wb') as f:
        f.write(header)
        f.write(map_bytes)
        f.write(node_bytes)

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

typedef struct {
    uint8_t is_leaf;
    uint8_t feature;
    float threshold;
    int32_t left;
    int32_t right;
    uint8_t leaf_class;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]);
    unsigned char *weights = slurp(argv[1], &nw);
    unsigned char *p = weights;

    if (nw < 4) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = *(uint32_t*)p; p += 4;
    if (magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }

    if (nw < (p - weights) + 3) { fprintf(stderr, "weights too small\n"); return 3; }
    uint8_t d = *p++; uint8_t C = *p++; uint8_t max_depth = *p++;

    if (nw < (p - weights) + 4) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t node_count = *(uint32_t*)p; p += 4;

    size_t map_size = (size_t)d * 256 * sizeof(float);
    if (nw < (p - weights) + map_size) { fprintf(stderr, "weights too small\n"); return 3; }
    float *map_table = (float*)p; p += map_size;

    Node *nodes = malloc(node_count * sizeof(Node));
    for (uint32_t i = 0; i < node_count; i++) {
        if (nw < (p - weights) + 15) { fprintf(stderr, "weights too small\n"); return 3; }
        nodes[i].is_leaf = *p++;
        nodes[i].feature = *p++;
        nodes[i].threshold = *(float*)p; p += 4;
        nodes[i].left = *(int32_t*)p; p += 4;
        nodes[i].right = *(int32_t*)p; p += 4;
        nodes[i].leaf_class = *p++;
    }
    if (p - weights != nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;
    float vals[14];  // d <= 14 per card
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            vals[j] = map_table[j * 256 + xi[j]];
        }
        uint32_t node_idx = 0;
        while (1) {
            Node *nd = &nodes[node_idx];
            if (nd->is_leaf) {
                unsigned char out = nd->leaf_class;
                put_rec(stdout, &out, 1);
                break;
            }
            if (vals[nd->feature] <= nd->threshold) node_idx = nd->left;
            else node_idx = nd->right;
        }
    }
    return 0;
}
```