```hypothesis
Family: label-tree (single CART decision tree on raw byte features).
Why: The Adult dataset has 14 byte features (ordinal quantile bins + categorical indices). A well-tuned decision tree can capture non-linear interactions that logistic regression misses, while remaining extremely fast at inference (just a few comparisons per record). We treat all features as ordinal and split on thresholds 0..254; this is deterministic and requires no embedding tables.
Failure signal: If dev accuracy stays ≤ 0.8524 (label-linear) or inference cost exceeds label-linear by >20%, the family does not justify its existence.
```

```python
"""label-tree: single CART decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Chooses max_depth and min_samples_leaf
on the last 20% of train.in (tuning split), then retrains on all data.
Weights format: magic 0x54524545, d (uint8), num_nodes (int32),
feature_idx[num_nodes] (uint8), threshold[num_nodes] (uint8),
left[num_nodes] (int32), right[num_nodes] (int32), leaf_class[num_nodes] (uint8).
Internal nodes have leaf_class=255; leaves have feature_idx=255, threshold=255, left=right=-1.
"""
import json, math, struct, sys
import numpy as np

SEED = 1

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
    counts = np.bincount(y, minlength=2)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def build_tree(X, y, max_depth, min_samples_leaf, depth=0):
    n = len(y)
    # stopping criteria
    if depth >= max_depth or n < min_samples_leaf or gini(y) == 0.0:
        leaf_class = np.bincount(y, minlength=2).argmax()
        return {'leaf': True, 'class': leaf_class}
    best_gain = 0.0
    best_feat = -1
    best_thresh = -1
    parent_gini = gini(y)
    # try all features and thresholds
    for feat in range(X.shape[1]):
        vals = X[:, feat]
        # thresholds from 0 to 254
        for thresh in range(255):
            left_mask = vals <= thresh
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            gain = parent_gini - (n_left/n)*gini(y[left_mask]) - (n_right/n)*gini(y[right_mask])
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh
    if best_gain <= 0.0:
        leaf_class = np.bincount(y, minlength=2).argmax()
        return {'leaf': True, 'class': leaf_class}
    left_mask = X[:, best_feat] <= best_thresh
    right_mask = ~left_mask
    left_tree = build_tree(X[left_mask], y[left_mask], max_depth, min_samples_leaf, depth+1)
    right_tree = build_tree(X[right_mask], y[right_mask], max_depth, min_samples_leaf, depth+1)
    return {'leaf': False, 'feat': best_feat, 'thresh': best_thresh,
            'left': left_tree, 'right': right_tree}

def flatten_tree(node, arrays):
    """arrays: list of (feat, thresh, left, right, leaf_class)"""
    if node['leaf']:
        idx = len(arrays)
        arrays.append((255, 255, -1, -1, node['class']))
        return idx
    else:
        left_idx = flatten_tree(node['left'], arrays)
        right_idx = flatten_tree(node['right'], arrays)
        idx = len(arrays)
        arrays.append((node['feat'], node['thresh'], left_idx, right_idx, 255))
        return idx

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n_total = len(y)
    # tuning split: last 20%
    split_idx = int(n_total * 0.8)
    X_train, y_train = X_bytes[:split_idx], y[:split_idx]
    X_val, y_val = X_bytes[split_idx:], y[split_idx:]

    # hyperparameter search
    best_acc = -1.0
    best_params = (5, 10)  # default
    for max_depth in [3, 4, 5, 6, 7, 8, 9, 10]:
        for min_samples_leaf in [1, 5, 10, 20, 50]:
            tree = build_tree(X_train, y_train, max_depth, min_samples_leaf)
            arrays = []
            flatten_tree(tree, arrays)
            # predict on validation
            preds = []
            for x in X_val:
                node_idx = 0
                while True:
                    feat, thresh, left, right, leaf_class = arrays[node_idx]
                    if leaf_class != 255:
                        preds.append(leaf_class)
                        break
                    if x[feat] <= thresh:
                        node_idx = left
                    else:
                        node_idx = right
            preds = np.array(preds, dtype=np.uint8)
            acc = (preds == y_val).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)

    # retrain on all data
    max_depth, min_samples_leaf = best_params
    tree = build_tree(X_bytes, y, max_depth, min_samples_leaf)
    arrays = []
    flatten_tree(tree, arrays)
    num_nodes = len(arrays)
    # pack weights
    feats = np.array([a[0] for a in arrays], dtype=np.uint8)
    threshes = np.array([a[1] for a in arrays], dtype=np.uint8)
    lefts = np.array([a[2] for a in arrays], dtype=np.int32)
    rights = np.array([a[3] for a in arrays], dtype=np.int32)
    leaf_classes = np.array([a[4] for a in arrays], dtype=np.uint8)
    d = X_bytes.shape[1]
    magic = 0x54524545  # 'TREE'
    out = struct.pack("<I B i", magic, d, num_nodes)
    out += feats.tobytes()
    out += threshes.tobytes()
    out += lefts.tobytes()
    out += rights.tobytes()
    out += leaf_classes.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class per record. */
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
    if (nw < 9) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    uint8_t d = wb[4];
    int32_t num_nodes = *(int32_t*)(wb + 5);
    if (num_nodes <= 0) { fprintf(stderr, "bad weights: num_nodes <= 0\n"); return 3; }
    long expected = 9 + (long)num_nodes * (1 + 1 + 4 + 4 + 1);
    if (nw != expected) { fprintf(stderr, "bad weights: size mismatch (got %ld, expected %ld)\n", nw, expected); return 3; }
    const uint8_t *feat = wb + 9;
    const uint8_t *thresh = feat + num_nodes;
    const int32_t *left = (const int32_t*)(thresh + num_nodes);
    const int32_t *right = (const int32_t*)((const uint8_t*)left + 4 * num_nodes);
    const uint8_t *leaf_class = (const uint8_t*)((const uint8_t*)right + 4 * num_nodes);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (1) {
            uint8_t lc = leaf_class[node];
            if (lc != 255) {
                put_rec(stdout, &lc, 1);
                break;
            }
            uint8_t f = feat[node];
            uint8_t t = thresh[node];
            if (xi[f] <= t) node = left[node];
            else node = right[node];
        }
    }
    return 0;
}
```