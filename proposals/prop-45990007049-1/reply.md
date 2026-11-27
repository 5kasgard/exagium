```hypothesis
Family: label-dtree (shallow decision tree). Optdigits pixels have strong local spatial structure (strokes, loops) that linear models miss; a depth-6 tree captures this with few splits. Tuning max_depth/min_samples_leaf on the last 20% of train, then retraining on all data. Success: dev accuracy > 0.965 at < 2000 Ir/item. Failure: accuracy ≤ 0.960 or cost ≥ 5000 Ir/item (no better than label-linear).
```

```python
import json, struct, sys
import numpy as np

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'class_label', 'is_leaf')
    def __init__(self):
        self.feature = -1
        self.threshold = 0
        self.left = None
        self.right = None
        self.class_label = -1
        self.is_leaf = False

def gini(y):
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, feature_indices, rng):
    best_gini = 1.0
    best_feat = -1
    best_thresh = 0
    n_samples = len(y)
    for feat in feature_indices:
        values = X[:, feat]
        unique_vals = np.unique(values)
        if len(unique_vals) <= 1:
            continue
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
        for thresh in thresholds:
            left_mask = values <= thresh
            right_mask = ~left_mask
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            left_gini = gini(y[left_mask])
            right_gini = gini(y[right_mask])
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            weighted_gini = (n_left * left_gini + n_right * right_gini) / n_samples
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh, best_gini

def build_tree(X, y, depth, max_depth, min_samples_leaf, feature_indices, rng):
    node = Node()
    if depth >= max_depth or len(y) < 2 * min_samples_leaf or gini(y) == 0.0:
        node.is_leaf = True
        node.class_label = int(np.bincount(y).argmax())
        return node
    feat, thresh, split_gini = best_split(X, y, feature_indices, rng)
    if feat == -1 or split_gini >= gini(y):
        node.is_leaf = True
        node.class_label = int(np.bincount(y).argmax())
        return node
    node.feature = feat
    node.threshold = int(thresh)
    left_mask = X[:, feat] <= node.threshold
    right_mask = ~left_mask
    node.left = build_tree(X[left_mask], y[left_mask], depth+1, max_depth, min_samples_leaf, feature_indices, rng)
    node.right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth, min_samples_leaf, feature_indices, rng)
    return node

def flatten_tree(node, nodes_list):
    idx = len(nodes_list)
    if node.is_leaf:
        nodes_list.append((-(node.class_label + 1), 0, 0, 0))
    else:
        nodes_list.append((0, 0, node.feature, node.threshold))
        left_idx = flatten_tree(node.left, nodes_list)
        right_idx = flatten_tree(node.right, nodes_list)
        nodes_list[idx] = (left_idx, right_idx, node.feature, node.threshold)
    return idx

def predict_tree(nodes, x):
    idx = 0
    while True:
        left, right, feat, thresh = nodes[idx]
        if left < 0:
            return -(left + 1)
        if x[feat] <= thresh:
            idx = left
        else:
            idx = right

def accuracy(nodes, X, y):
    correct = 0
    for i in range(len(y)):
        if predict_tree(nodes, X[i]) == y[i]:
            correct += 1
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    rng = np.random.default_rng(SEED)
    split_idx = int(n * 0.8)
    indices = np.arange(n)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    feature_indices = np.arange(card["input"]["bytes"])
    best_params = None
    best_val_acc = -1.0
    for max_depth in [4, 5, 6, 7]:
        for min_samples_leaf in [1, 2, 5, 10]:
            tree = build_tree(X_train, y_train, 0, max_depth, min_samples_leaf, feature_indices, rng)
            nodes = []
            flatten_tree(tree, nodes)
            val_acc = accuracy(nodes, X_val, y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (max_depth, min_samples_leaf)
    max_depth, min_samples_leaf = best_params
    tree = build_tree(X, y, 0, max_depth, min_samples_leaf, feature_indices, rng)
    nodes = []
    flatten_tree(tree, nodes)
    out = struct.pack("<IH", 0x44545245, len(nodes))
    for left, right, feat, thresh in nodes:
        out += struct.pack("<iiBBH", left, right, feat, thresh, 0)
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
    int32_t left;
    int32_t right;
    uint8_t feature;
    uint8_t threshold;
    uint16_t pad;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 6) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t*)wbuf;
    if (magic != 0x44545245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint16_t num_nodes = *(uint16_t*)(wbuf + 4);
    if (nw != 6 + (long)num_nodes * 12) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node*)(wbuf + 6);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int idx = 0;
        while (1) {
            Node *node = &nodes[idx];
            if (node->left < 0) {
                unsigned char cls = (unsigned char)(-(node->left + 1));
                put_rec(stdout, &cls, 1);
                break;
            }
            if (xi[node->feature] <= node->threshold) {
                idx = node->left;
            } else {
                idx = node->right;
            }
        }
    }
    return 0;
}
```