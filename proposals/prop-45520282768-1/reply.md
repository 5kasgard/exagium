```hypothesis
Family: label-dtree (decision tree on 64 raw pixel features).
Why: Optdigits has piecewise-constant decision boundaries; a shallow tree captures non-linearities better than linear model.
Inference is just integer comparisons (O(depth) ~ 10), so Ir/item should be << label-linear's 5058.
Failure: If dev accuracy <= 0.9613 or Ir/item >= 5058, the family doesn't improve on label-linear.
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
    def __init__(self, feature=-1, threshold=0.0, left=-1, right=-1, class_label=-1):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.class_label = class_label
        self.is_leaf = (feature == -1)

def gini(y):
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, feature_idx, min_samples_leaf):
    vals = X[:, feature_idx]
    sorted_idx = np.argsort(vals, kind='mergesort')
    sorted_vals = vals[sorted_idx]
    sorted_y = y[sorted_idx]
    
    best_gain = -1.0
    best_thresh = 0.0
    best_left_idx = None
    best_right_idx = None
    
    n = len(y)
    parent_gini = gini(y)
    
    left_counts = np.zeros(10, dtype=np.int64)
    right_counts = np.bincount(y, minlength=10).astype(np.int64)
    left_n = 0
    right_n = n
    
    i = 0
    while i < n - 1:
        v = sorted_vals[i]
        j = i
        while j < n and sorted_vals[j] == v:
            c = sorted_y[j]
            left_counts[c] += 1
            right_counts[c] -= 1
            left_n += 1
            right_n -= 1
            j += 1
        
        if left_n >= min_samples_leaf and right_n >= min_samples_leaf:
            next_v = sorted_vals[j] if j < n else v + 1
            thresh = (v + next_v) / 2.0
            
            left_gini = 1.0 - np.sum((left_counts / left_n) ** 2) if left_n > 0 else 0
            right_gini = 1.0 - np.sum((right_counts / right_n) ** 2) if right_n > 0 else 0
            gain = parent_gini - (left_n * left_gini + right_n * right_gini) / n
            
            if gain > best_gain:
                best_gain = gain
                best_thresh = thresh
                best_left_idx = sorted_idx[:j].copy()
                best_right_idx = sorted_idx[j:].copy()
        
        i = j
    
    if best_gain <= 0:
        return None
    return best_thresh, best_gain, best_left_idx, best_right_idx

def build_tree(X, y, depth, max_depth, min_samples_leaf, min_samples_split):
    if depth >= max_depth:
        return Node(class_label=np.bincount(y, minlength=10).argmax())
    if len(y) < min_samples_split:
        return Node(class_label=np.bincount(y, minlength=10).argmax())
    if gini(y) == 0.0:
        return Node(class_label=y[0])
    
    best_feature = -1
    best_thresh = 0.0
    best_left_idx = None
    best_right_idx = None
    best_gain = -1.0
    
    for f in range(X.shape[1]):
        res = best_split(X, y, f, min_samples_leaf)
        if res is not None:
            thresh, gain, left_idx, right_idx = res
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_thresh = thresh
                best_left_idx = left_idx
                best_right_idx = right_idx
    
    if best_feature == -1 or best_gain <= 0:
        return Node(class_label=np.bincount(y, minlength=10).argmax())
    
    node = Node(feature=best_feature, threshold=best_thresh)
    node.left = build_tree(X[best_left_idx], y[best_left_idx], depth+1, max_depth, min_samples_leaf, min_samples_split)
    node.right = build_tree(X[best_right_idx], y[best_right_idx], depth+1, max_depth, min_samples_leaf, min_samples_split)
    return node

def tree_to_array(root):
    features = []
    thresholds = []
    lefts = []
    rights = []
    classes = []
    
    def traverse(node):
        idx = len(features)
        if node.is_leaf:
            features.append(-1)
            thresholds.append(0.0)
            lefts.append(-1)
            rights.append(-1)
            classes.append(node.class_label)
        else:
            features.append(node.feature)
            thresholds.append(node.threshold)
            left_idx = traverse(node.left)
            right_idx = traverse(node.right)
            lefts.append(left_idx)
            rights.append(right_idx)
            classes.append(-1)
        return idx
    
    traverse(root)
    return (np.array(features, dtype=np.int32),
            np.array(thresholds, dtype=np.float32),
            np.array(lefts, dtype=np.int32),
            np.array(rights, dtype=np.int32),
            np.array(classes, dtype=np.int32))

def evaluate_tree(features, thresholds, lefts, rights, classes, X):
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.int32)
    for i in range(n):
        node = 0
        while features[node] != -1:
            f = features[node]
            if X[i, f] <= thresholds[node]:
                node = lefts[node]
            else:
                node = rights[node]
        preds[i] = classes[node]
    return preds

def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    
    # Use raw pixels (0-16) directly; tree splits are invariant to monotonic scaling
    X = X_raw.astype(np.float32)
    
    # Validation split: last 20% in file order
    n = len(X)
    val_start = int(0.8 * n)
    X_train, X_val = X[:val_start], X[val_start:]
    y_train, y_val = y[:val_start], y[val_start:]
    
    best_acc = -1.0
    best_params = None
    best_arrays = None
    
    for max_depth in [4, 5, 6, 7, 8, 9, 10]:
        for min_samples_leaf in [1, 2, 5, 10]:
            for min_samples_split in [2, 5, 10]:
                np.random.seed(SEED)
                root = build_tree(X_train, y_train, 0, max_depth, min_samples_leaf, min_samples_split)
                features, thresholds, lefts, rights, classes = tree_to_array(root)
                preds = evaluate_tree(features, thresholds, lefts, rights, classes, X_val)
                acc = (preds == y_val).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = (max_depth, min_samples_leaf, min_samples_split)
                    best_arrays = (features, thresholds, lefts, rights, classes)
    
    # Retrain on all data
    max_depth, min_samples_leaf, min_samples_split = best_params
    np.random.seed(SEED)
    root = build_tree(X, y, 0, max_depth, min_samples_leaf, min_samples_split)
    features, thresholds, lefts, rights, classes = tree_to_array(root)
    
    # Serialize: magic, d, C, n_nodes, then nodes (feature, threshold, left, right, class)
    n_nodes = len(features)
    out = struct.pack("<4i", 0x45525444, d, C, n_nodes)
    for i in range(n_nodes):
        out += struct.pack("<ifiii", features[i], thresholds[i], lefts[i], rights[i], classes[i])
    
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x45525444) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], n_nodes = hd[3];
    if (d != 64 || C != 10 || n_nodes <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + (long)n_nodes * 20;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }
    
    int32_t *features = (int32_t *)(wb + 16);
    float *thresholds = (float *)(features + n_nodes);
    int32_t *lefts = (int32_t *)(thresholds + n_nodes);
    int32_t *rights = (int32_t *)(lefts + n_nodes);
    int32_t *classes = (int32_t *)(rights + n_nodes);
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        
        int node = 0;
        while (features[node] != -1) {
            int f = features[node];
            float x_val = (float)xi[f];
            if (x_val <= thresholds[node]) {
                node = lefts[node];
            } else {
                node = rights[node];
            }
        }
        unsigned char best = (unsigned char)classes[node];
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```