```hypothesis
Family: label-tree (single decision tree with tuned max depth).
Why: Adult income has non-linear feature interactions (e.g., education × occupation) that linear models miss; a shallow tree captures these with few splits.
Failure signal: Dev accuracy ≤ 0.8524 (label-linear) would indicate a single tree is too weak for this task.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 20
MAX_DEPTH_CANDIDATES = list(range(1, 9))


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
    p = np.mean(y == 1)
    return 2.0 * p * (1.0 - p)


def information_gain(y, left_mask, right_mask):
    n = len(y)
    n_left = np.sum(left_mask)
    n_right = n - n_left
    if n_left == 0 or n_right == 0:
        return -1.0
    gini_parent = gini(y)
    gini_left = gini(y[left_mask])
    gini_right = gini(y[right_mask])
    gain = gini_parent - (n_left / n) * gini_left - (n_right / n) * gini_right
    return gain


def find_best_split(X, y, feature_types, indices):
    n = len(indices)
    if n < 2 * MIN_SAMPLES_LEAF:
        return None, None, None, None
    best_gain = -1.0
    best_feat = None
    best_type = None
    best_thresh = None
    best_left_mask = None
    X_sub = X[indices]
    y_sub = y[indices]
    for feat in range(X.shape[1]):
        ftype = feature_types[feat]
        vals = X_sub[:, feat]
        if ftype == "ordinal":
            unique = np.unique(vals)
            if len(unique) <= 1:
                continue
            thresholds = (unique[:-1] + unique[1:]) // 2
            for thresh in thresholds:
                left_mask = vals <= thresh
                if np.sum(left_mask) < MIN_SAMPLES_LEAF or np.sum(~left_mask) < MIN_SAMPLES_LEAF:
                    continue
                gain = information_gain(y_sub, left_mask, ~left_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_type = 0
                    best_thresh = thresh
                    best_left_mask = left_mask
        else:
            unique = np.unique(vals)
            for v in unique:
                left_mask = vals == v
                if np.sum(left_mask) < MIN_SAMPLES_LEAF or np.sum(~left_mask) < MIN_SAMPLES_LEAF:
                    continue
                gain = information_gain(y_sub, left_mask, ~left_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_type = 1
                    best_thresh = v
                    best_left_mask = left_mask
    if best_gain <= 0:
        return None, None, None, None
    return best_feat, best_type, best_thresh, best_left_mask


def build_tree(X, y, feature_types, indices, depth, max_depth, nodes):
    node_idx = len(nodes)
    nodes.append(None)
    n = len(indices)
    if depth == max_depth or n < 2 * MIN_SAMPLES_LEAF or gini(y[indices]) == 0:
        leaf_class = int(np.round(np.mean(y[indices])))
        node = {
            'feature_idx': -1,
            'split_type': -1,
            'threshold': 0,
            'left_child': -1,
            'right_child': -1,
            'leaf_class': leaf_class
        }
        nodes[node_idx] = node
        return node_idx
    feat, stype, thresh, left_mask = find_best_split(X, y, feature_types, indices)
    if feat is None:
        leaf_class = int(np.round(np.mean(y[indices])))
        node = {
            'feature_idx': -1,
            'split_type': -1,
            'threshold': 0,
            'left_child': -1,
            'right_child': -1,
            'leaf_class': leaf_class
        }
        nodes[node_idx] = node
        return node_idx
    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]
    left_child = build_tree(X, y, feature_types, left_indices, depth + 1, max_depth, nodes)
    right_child = build_tree(X, y, feature_types, right_indices, depth + 1, max_depth, nodes)
    node = {
        'feature_idx': feat,
        'split_type': stype,
        'threshold': int(thresh),
        'left_child': left_child,
        'right_child': right_child,
        'leaf_class': 0
    }
    nodes[node_idx] = node
    return node_idx


def predict_tree(nodes, x):
    node_idx = 0
    while True:
        node = nodes[node_idx]
        if node['left_child'] == -1:
            return node['leaf_class']
        feat = node['feature_idx']
        if node['split_type'] == 0:
            if x[feat] <= node['threshold']:
                node_idx = node['left_child']
            else:
                node_idx = node['right_child']
        else:
            if x[feat] == node['threshold']:
                node_idx = node['left_child']
            else:
                node_idx = node['right_child']


def evaluate_tree(nodes, X, y):
    preds = np.array([predict_tree(nodes, X[i]) for i in range(X.shape[0])])
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    feature_types = card["input"]["types"]
    n = X.shape[0]
    split_idx = int(n * 0.8)
    train_idx = np.arange(split_idx)
    val_idx = np.arange(split_idx, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    rng = np.random.default_rng(SEED)
    best_depth = 1
    best_acc = 0.0
    for max_depth in MAX_DEPTH_CANDIDATES:
        nodes = []
        build_tree(X_train, y_train, feature_types, train_idx, 0, max_depth, nodes)
        acc = evaluate_tree(nodes, X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_depth = max_depth
    nodes = []
    all_indices = np.arange(n)
    build_tree(X, y, feature_types, all_indices, 0, best_depth, nodes)
    magic = 0x54524545
    num_nodes = len(nodes)
    num_features = X.shape[1]
    num_classes = card["classes"]
    out = struct.pack("<4i", magic, num_nodes, num_features, num_classes)
    for node in nodes:
        feat = node['feature_idx']
        stype = node['split_type']
        thresh = node['threshold']
        left = node['left_child']
        right = node['right_child']
        leaf = node['leaf_class']
        out += struct.pack("<6i", feat, stype, thresh, left, right, leaf)
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

typedef struct {
    int32_t feature_idx;
    int32_t split_type;
    int32_t threshold;
    int32_t left_child;
    int32_t right_child;
    int32_t leaf_class;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    uint32_t magic = hd[0];
    if (magic != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t num_nodes = hd[1];
    int32_t num_features = hd[2];
    int32_t num_classes = hd[3];
    if (num_nodes <= 0 || num_features != 14 || num_classes != 2) {
        fprintf(stderr, "bad weights header\n"); return 3;
    }
    long expected_size = 16 + (long)num_nodes * 24;
    if (nw != expected_size) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != num_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, num_features); return 3; }
        int node_idx = 0;
        while (1) {
            Node *nd = &nodes[node_idx];
            if (nd->left_child == -1) {
                unsigned char best = (unsigned char)nd->leaf_class;
                put_rec(stdout, &best, 1);
                break;
            }
            uint8_t feat_val = xi[nd->feature_idx];
            if (nd->split_type == 0) {
                if (feat_val <= (uint8_t)nd->threshold) node_idx = nd->left_child;
                else node_idx = nd->right_child;
            } else {
                if (feat_val == (uint8_t)nd->threshold) node_idx = nd->left_child;
                else node_idx = nd->right_child;
            }
        }
    }
    return 0;
}
```