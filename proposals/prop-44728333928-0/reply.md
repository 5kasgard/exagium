```hypothesis
We propose a decision tree family "label-tree". Decision trees capture non-linear feature interactions and are extremely fast at inference (only a few integer comparisons per record). On the Adult dataset with 14 binned features, a shallow tree (depth 5–7) should outperform logistic regression (label-linear, 0.8524) while keeping CPU cost very low. If the tree does not beat label-linear on dev accuracy, or if its inference cost exceeds label-linear significantly, the hypothesis fails.
```

```python
import json, struct, sys
import numpy as np

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def build_tree(X, y, max_depth, min_samples_leaf, depth=0):
    n = X.shape[0]
    if n < min_samples_leaf or depth == max_depth:
        return ('leaf', float(y.mean()))
    if np.all(y == y[0]):
        return ('leaf', float(y[0]))
    best_gini = 1.0
    best_feature = -1
    best_threshold = -1
    best_left_mask = None
    best_right_mask = None
    for j in range(X.shape[1]):
        vals = X[:, j]
        hist_total = np.bincount(vals, minlength=256)
        hist_class1 = np.bincount(vals, weights=y.astype(np.float64), minlength=256)
        cum_total = np.cumsum(hist_total)
        cum_class1 = np.cumsum(hist_class1)
        total_n = cum_total[-1]
        total_class1 = cum_class1[-1]
        for t in range(255):
            left_n = cum_total[t]
            right_n = total_n - left_n
            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue
            left_class1 = cum_class1[t]
            right_class1 = total_class1 - left_class1
            p_left = left_class1 / left_n
            gini_left = 1.0 - p_left*p_left - (1.0-p_left)*(1.0-p_left)
            p_right = right_class1 / right_n
            gini_right = 1.0 - p_right*p_right - (1.0-p_right)*(1.0-p_right)
            weighted_gini = (left_n * gini_left + right_n * gini_right) / total_n
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = j
                best_threshold = t
                best_left_mask = vals <= t
                best_right_mask = vals > t
    if best_feature == -1:
        return ('leaf', float(y.mean()))
    left_tree = build_tree(X[best_left_mask], y[best_left_mask], max_depth, min_samples_leaf, depth+1)
    right_tree = build_tree(X[best_right_mask], y[best_right_mask], max_depth, min_samples_leaf, depth+1)
    return ('node', best_feature, best_threshold, left_tree, right_tree)

def predict_tree(tree, x):
    node = tree
    while node[0] != 'leaf':
        _, feature, threshold, left, right = node
        if x[feature] <= threshold:
            node = left
        else:
            node = right
    return node[1]

def evaluate_tree(tree, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        p = predict_tree(tree, X[i])
        pred = 1 if p > 0.5 else 0
        if pred == y[i]:
            correct += 1
    return correct / n

def serialize_tree(tree):
    nodes = []
    max_depth = 0
    def traverse(node, depth):
        nonlocal max_depth
        if depth > max_depth:
            max_depth = depth
        if node[0] == 'leaf':
            idx = len(nodes)
            nodes.append((255, 0, -1, -1, node[1]))
            return idx
        else:
            _, feature, threshold, left, right = node
            idx = len(nodes)
            nodes.append((feature, threshold, -1, -1, 0.0))
            left_idx = traverse(left, depth+1)
            right_idx = traverse(right, depth+1)
            nodes[idx] = (feature, threshold, left_idx, right_idx, 0.0)
            return idx
    traverse(tree, 0)
    return nodes, max_depth

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_all.shape[0]
    split = int(n * 0.8)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    best_acc = -1.0
    best_params = None
    best_tree = None

    for max_depth in [3,4,5,6,7,8]:
        for min_samples_leaf in [1,5,10,20,50]:
            tree = build_tree(X_train, y_train, max_depth, min_samples_leaf)
            acc = evaluate_tree(tree, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)
                best_tree = tree

    final_tree = build_tree(X_all, y_all, best_params[0], best_params[1])
    nodes, max_depth_actual = serialize_tree(final_tree)

    magic = 0x45455254  # 'TREE' little-endian
    n_nodes = len(nodes)
    out = struct.pack('<Iiii', magic, d, n_nodes, max_depth_actual)
    for node in nodes:
        feature, threshold, left, right, value = node
        out += struct.pack('<BBiiif', feature, threshold, left, right, value)
    open(sys.argv[4], 'wb').write(out)

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

struct __attribute__((packed)) Node {
    uint8_t feature;
    uint8_t threshold;
    int32_t left;
    int32_t right;
    float value;
};

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *weights = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = weights[0] | weights[1] << 8 | weights[2] << 16 | (uint32_t)weights[3] << 24;
    if (magic != 0x45455254) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int32_t d = *(int32_t*)(weights + 4);
    int32_t n_nodes = *(int32_t*)(weights + 8);
    int32_t max_depth = *(int32_t*)(weights + 12);
    if (d != 14) { fprintf(stderr, "bad weights: d=%d != 14\n", d); return 3; }
    if (n_nodes <= 0) { fprintf(stderr, "bad weights: n_nodes=%d\n", n_nodes); return 3; }
    size_t expected = 16 + (size_t)n_nodes * sizeof(struct Node);
    if (nw != (long)expected) { fprintf(stderr, "bad weights: size mismatch\n"); return 3; }
    struct Node *nodes = (struct Node*)(weights + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node_idx = 0;
        while (1) {
            struct Node *node = &nodes[node_idx];
            if (node->feature == 255) {
                unsigned char best = (node->value > 0.5f) ? 1 : 0;
                put_rec(stdout, &best, 1);
                break;
            }
            uint8_t feat = node->feature;
            uint8_t thresh = node->threshold;
            if (xi[feat] <= thresh) {
                node_idx = node->left;
            } else {
                node_idx = node->right;
            }
        }
    }
    return 0;
}
```