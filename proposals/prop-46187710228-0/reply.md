```hypothesis
Family: label-dtree (single decision tree). Decision trees capture non-linear feature interactions with very few comparisons per item. On this tabular task with mixed ordinal/categorical features encoded as bytes, a well-tuned shallow tree (max depth ~6) should approach label-linear accuracy (0.8524) while using far fewer instructions (target <500 Ir/item). If dev accuracy falls below 0.85 or cost exceeds label-linear, the hypothesis fails.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
TUNING_FRAC = 0.2

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

def build_tree(X, y, max_depth, min_samples_leaf, min_samples_split, depth=0):
    n = len(y)
    # leaf conditions
    if depth >= max_depth or n < min_samples_split or gini(y) == 0.0:
        vals, cnts = np.unique(y, return_counts=True)
        leaf_class = int(vals[np.argmax(cnts)])  # majority, ties broken by smallest class
        return {"leaf": True, "class": leaf_class}
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1
    parent_gini = gini(y)
    for feat in range(X.shape[1]):
        vals = X[:, feat]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        # try thresholds between unique values
        for t in uniq[:-1]:  # split <= t vs > t
            left = y[vals <= t]
            right = y[vals > t]
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue
            gain = parent_gini - (len(left)/n)*gini(left) - (len(right)/n)*gini(right)
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feat = feat
                best_thresh = int(t)
    if best_gain <= 0:
        vals, cnts = np.unique(y, return_counts=True)
        leaf_class = int(vals[np.argmax(cnts)])
        return {"leaf": True, "class": leaf_class}
    left_idx = X[:, best_feat] <= best_thresh
    right_idx = ~left_idx
    left_tree = build_tree(X[left_idx], y[left_idx], max_depth, min_samples_leaf, min_samples_split, depth+1)
    right_tree = build_tree(X[right_idx], y[right_idx], max_depth, min_samples_leaf, min_samples_split, depth+1)
    return {"leaf": False, "feat": best_feat, "thresh": best_thresh, "left": left_tree, "right": right_tree}

def tree_to_arrays(node, arrays):
    """arrays: dict with lists feature, thresh, class, left, right"""
    idx = len(arrays["feature"])
    if node["leaf"]:
        arrays["feature"].append(-1)
        arrays["thresh"].append(0)
        arrays["class"].append(node["class"])
        arrays["left"].append(-1)
        arrays["right"].append(-1)
    else:
        arrays["feature"].append(node["feat"])
        arrays["thresh"].append(node["thresh"])
        arrays["class"].append(0)
        left_idx = tree_to_arrays(node["left"], arrays)
        right_idx = tree_to_arrays(node["right"], arrays)
        arrays["left"].append(left_idx)
        arrays["right"].append(right_idx)
    return idx

def evaluate_tree(X, y, arrays):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        node = 0
        while True:
            feat = arrays["feature"][node]
            if feat == -1:
                pred = arrays["class"][node]
                break
            if X[i, feat] <= arrays["thresh"][node]:
                node = arrays["left"][node]
            else:
                node = arrays["right"][node]
        if pred == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    d = card["input"]["bytes"]
    C = card["classes"]
    # tuning split: last 20%
    tune_size = int(n * TUNING_FRAC)
    tune_idx = np.arange(n - tune_size, n)
    train_idx = np.arange(n - tune_size)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]
    # hyperparameter grid
    best_acc = -1.0
    best_params = None
    best_arrays = None
    for max_depth in [4, 5, 6, 7, 8]:
        for min_samples_leaf in [1, 2, 5, 10, 20]:
            for min_samples_split in [2, 5, 10, 20]:
                if min_samples_split <= min_samples_leaf:
                    continue
                tree = build_tree(X_train, y_train, max_depth, min_samples_leaf, min_samples_split)
                arrays = {"feature": [], "thresh": [], "class": [], "left": [], "right": []}
                tree_to_arrays(tree, arrays)
                acc = evaluate_tree(X_tune, y_tune, arrays)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (max_depth, min_samples_leaf, min_samples_split)
                    best_arrays = arrays
    # retrain on full data with best params
    tree = build_tree(X, y, *best_params)
    arrays = {"feature": [], "thresh": [], "class": [], "left": [], "right": []}
    tree_to_arrays(tree, arrays)
    n_nodes = len(arrays["feature"])
    # serialize
    magic = 0x44545245  # 'DTRE'
    out = struct.pack("<4i", magic, d, C, n_nodes)
    out += np.array(arrays["feature"], dtype=np.int32).tobytes()
    out += np.array(arrays["thresh"], dtype=np.uint8).tobytes()
    out += np.array(arrays["class"], dtype=np.uint8).tobytes()
    out += np.array(arrays["left"], dtype=np.int32).tobytes()
    out += np.array(arrays["right"], dtype=np.int32).tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Decision tree inference for label tasks: ./infer weights.bin in.rec N */
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
    if (hd[0] != 0x44545245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], n_nodes = hd[3];
    if (d <= 0 || C <= 0 || C > 256 || n_nodes <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    size_t expected = 16
        + (size_t)n_nodes * sizeof(int32_t)   // feature
        + (size_t)n_nodes * sizeof(uint8_t)   // thresh
        + (size_t)n_nodes * sizeof(uint8_t)   // class
        + (size_t)n_nodes * sizeof(int32_t)   // left
        + (size_t)n_nodes * sizeof(int32_t);  // right
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    int32_t *feature = (int32_t *)(wb + 16);
    uint8_t *thresh = (uint8_t *)(feature + n_nodes);
    uint8_t *cls = thresh + n_nodes;
    int32_t *left = (int32_t *)(cls + n_nodes);
    int32_t *right = left + n_nodes;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (1) {
            int32_t f = feature[node];
            if (f == -1) { unsigned char pred = cls[node]; put_rec(stdout, &pred, 1); break; }
            if (xi[f] <= thresh[node]) node = left[node];
            else node = right[node];
        }
    }
    return 0;
}
```