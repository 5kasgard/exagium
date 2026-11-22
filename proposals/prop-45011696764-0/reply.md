```hypothesis
Family: label-tree (single decision tree with ordinal/categorical splits). The adult dataset has mixed ordinal and categorical features; a shallow tree can capture non-linear interactions that linear models miss, while remaining extremely fast at inference (just a few integer comparisons per sample). We tune max_depth on the held-out 20% tuning split. If dev accuracy does not exceed label-linear (0.8524) or inference cost is not lower, the approach fails.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
MIN_SAMPLES_SPLIT = 20
MIN_SAMPLES_LEAF = 10
MAX_DEPTH_CANDIDATES = list(range(1, 11))

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

def gini_from_counts(c0, c1):
    total = c0 + c1
    if total == 0:
        return 0.0
    p = c1 / total
    return 2.0 * p * (1.0 - p)

def best_split(X, y, feature_types, min_samples_leaf):
    n, d = X.shape
    best_gini = gini(y)
    best_feature = -1
    best_threshold = -1
    for j in range(d):
        vals = X[:, j]
        if feature_types[j] == 0:  # ordinal
            # values 0..254 (255 bins)
            count0 = np.zeros(255, dtype=np.int64)
            count1 = np.zeros(255, dtype=np.int64)
            for v in range(255):
                mask = (vals == v)
                if mask.any():
                    count0[v] = np.sum(y[mask] == 0)
                    count1[v] = np.sum(y[mask] == 1)
            cum0 = np.cumsum(count0)
            cum1 = np.cumsum(count1)
            total0 = cum0[-1]
            total1 = cum1[-1]
            for t in range(254):  # threshold 0..253
                left0, left1 = cum0[t], cum1[t]
                right0, right1 = total0 - left0, total1 - left1
                if left0 + left1 < min_samples_leaf or right0 + right1 < min_samples_leaf:
                    continue
                weighted_gini = ((left0 + left1) * gini_from_counts(left0, left1) +
                                 (right0 + right1) * gini_from_counts(right0, right1)) / n
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = j
                    best_threshold = t
        else:  # categorical
            cats = np.unique(vals)
            for cat in cats:
                left_mask = (vals == cat)
                right_mask = ~left_mask
                left_n = left_mask.sum()
                right_n = right_mask.sum()
                if left_n < min_samples_leaf or right_n < min_samples_leaf:
                    continue
                left_y = y[left_mask]
                right_y = y[right_mask]
                weighted_gini = (left_n * gini(left_y) + right_n * gini(right_y)) / n
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = j
                    best_threshold = int(cat)
    return best_feature, best_threshold

def build_tree(X, y, feature_types, max_depth, min_samples_split, min_samples_leaf, depth=0):
    n = len(y)
    if depth == max_depth or n < min_samples_split or gini(y) == 0.0:
        return {'is_leaf': True, 'class': int(np.bincount(y).argmax())}
    feature, threshold = best_split(X, y, feature_types, min_samples_leaf)
    if feature == -1:
        return {'is_leaf': True, 'class': int(np.bincount(y).argmax())}
    if feature_types[feature] == 0:
        left_mask = X[:, feature] <= threshold
    else:
        left_mask = X[:, feature] == threshold
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], feature_types, max_depth, min_samples_split, min_samples_leaf, depth+1)
    right = build_tree(X[right_mask], y[right_mask], feature_types, max_depth, min_samples_split, min_samples_leaf, depth+1)
    return {'is_leaf': False, 'feature': feature, 'threshold': threshold, 'left': left, 'right': right}

def tree_to_array(node, arr):
    if node['is_leaf']:
        idx = len(arr)
        arr.append((1, 0, 0, 0, 0, node['class']))
        return idx
    else:
        left_idx = tree_to_array(node['left'], arr)
        right_idx = tree_to_array(node['right'], arr)
        idx = len(arr)
        arr.append((0, node['feature'], node['threshold'], left_idx, right_idx, 0))
        return idx

def evaluate_tree(root, X, feature_types):
    # iterative traversal for speed
    nodes = root  # actually we need the array and root index; we'll implement separately
    pass  # not used in training evaluation; we'll write a simple recursive evaluator for tuning

def predict_tree(node, x, feature_types):
    if node['is_leaf']:
        return node['class']
    if feature_types[node['feature']] == 0:
        go_left = x[node['feature']] <= node['threshold']
    else:
        go_left = x[node['feature']] == node['threshold']
    return predict_tree(node['left'] if go_left else node['right'], x, feature_types)

def accuracy(tree, X, y, feature_types):
    correct = 0
    for i in range(len(y)):
        if predict_tree(tree, X[i], feature_types) == y[i]:
            correct += 1
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    feature_types = np.array([0 if t == "ordinal" else 1 for t in card["input"]["types"]], dtype=np.uint8)

    # split: last 20% for tuning
    n = len(y)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_depth = 1
    best_acc = -1.0
    for depth in MAX_DEPTH_CANDIDATES:
        tree = build_tree(X_train, y_train, feature_types, depth, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF)
        acc = accuracy(tree, X_tune, y_tune, feature_types)
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # retrain on full data
    final_tree = build_tree(X, y, feature_types, best_depth, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF)

    # flatten to array (post-order so children indices < parent index? Actually we want parent to know children indices.
    # We'll do post-order: children added first, then parent. Root will be last.
    nodes_arr = []
    root_idx = tree_to_array(final_tree, nodes_arr)
    num_nodes = len(nodes_arr)

    # write weights.bin
    MAGIC = 0x45455254  # 'TREE'
    header = struct.pack("<BBI B 14s", MAGIC, d, best_depth, num_nodes, root_idx, feature_types.tobytes())
    # But the above format is messy; let's do it stepwise.
    # We'll use a bytearray.
    out = bytearray()
    out += struct.pack("<I", MAGIC)
    out += struct.pack("<B", d)
    out += struct.pack("<B", best_depth)
    out += struct.pack("<I", num_nodes)
    out += struct.pack("<I", root_idx)
    out += feature_types.tobytes()
    for node in nodes_arr:
        is_leaf, feature, threshold, left, right, cls = node
        out += struct.pack("<BBBIIB", is_leaf, feature, threshold, left, right, cls)
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

typedef struct __attribute__((packed)) {
    uint8_t is_leaf;
    uint8_t feature_idx;
    uint8_t threshold;
    uint32_t left;
    uint32_t right;
    uint8_t class_label;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 4) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x45455254) { fprintf(stderr, "bad weights magic\n"); return 3; }
    if (nw < 4 + 1 + 1 + 4 + 4 + 14) { fprintf(stderr, "bad weights header\n"); return 3; }
    uint8_t d = wb[4];
    uint8_t max_depth = wb[5]; (void)max_depth;
    uint32_t num_nodes = wb[6] | wb[7] << 8 | wb[8] << 16 | (uint32_t)wb[9] << 24;
    uint32_t root_idx = wb[10] | wb[11] << 8 | wb[12] << 16 | (uint32_t)wb[13] << 24;
    const uint8_t *feature_types = wb + 14;
    const Node *nodes = (const Node *)(wb + 14 + d);
    long expected = 14 + d + (long)num_nodes * (long)sizeof(Node);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    if (root_idx >= num_nodes) { fprintf(stderr, "bad root index\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        uint32_t node_idx = root_idx;
        while (1) {
            const Node *node = &nodes[node_idx];
            if (node->is_leaf) {
                unsigned char pred = node->class_label;
                put_rec(stdout, &pred, 1);
                break;
            }
            uint8_t feat = node->feature_idx;
            uint8_t thresh = node->threshold;
            int go_left;
            if (feature_types[feat] == 0) { // ordinal
                go_left = xi[feat] <= thresh;
            } else { // categorical
                go_left = xi[feat] == thresh;
            }
            node_idx = go_left ? node->left : node->right;
            if (node_idx >= num_nodes) { fprintf(stderr, "bad child index\n"); return 3; }
        }
    }
    return 0;
}
```