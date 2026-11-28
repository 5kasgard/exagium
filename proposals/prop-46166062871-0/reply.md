```hypothesis
Family: label-dt (single deterministic decision tree). Why: Adult has 14 mixed-type features; a shallow tree can capture non-linear interactions that linear models miss, while staying extremely fast at inference (a few branch predictions per record). The tuning split (last 20% of train) selects max_depth and min_leaf to avoid overfit. Failure mode: if dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-linear, the family does not justify itself.
```

```python
"""label-dt: single deterministic decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports a binary tree: magic 0x44544C31, uint32 num_nodes, uint32 C;
per node: uint8 is_leaf; if leaf: uint8 class; else: uint8 feature, uint8 split_type,
          float32 threshold (ordinal) or 32-byte bitset (categorical),
          int32 left, int32 right (indices in node array, BFS order)."""
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid searched on tuning split (last 20% of train)
MAX_DEPTH_GRID = [3, 4, 5, 6]
MIN_LEAF_GRID  = [1, 5, 10, 20]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def gini(y):
    if len(y) == 0: return 0.0
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, feature_types, rng):
    n, d = X.shape
    best_gain = -1.0
    best_feat = -1
    best_thresh = None
    best_subset = None
    parent_gini = gini(y)
    for j in range(d):
        vals = X[:, j]
        if feature_types[j] == "ordinal":
            uniq = np.unique(vals)
            if len(uniq) <= 1: continue
            # try midpoints between sorted unique values
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            for t in thresholds:
                left = vals <= t
                if left.sum() == 0 or (~left).sum() == 0: continue
                gain = parent_gini - (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / n
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, j, float(t)
        else:
            # categorical: try each category as a binary split (one-vs-rest) for determinism
            # to limit combinations, we sort categories by mean target and try prefixes
            cats = np.unique(vals)
            if len(cats) <= 1: continue
            cat_order = np.argsort([y[vals == c].mean() for c in cats])
            sorted_cats = cats[cat_order]
            left_mask = np.zeros(256, dtype=bool)
            for c in sorted_cats[:-1]:
                left_mask[c] = True
                left = left_mask[vals]
                if left.sum() == 0 or (~left).sum() == 0: continue
                gain = parent_gini - (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / n
                if gain > best_gain:
                    best_gain, best_feat = gain, j
                    best_subset = left_mask.copy()
    return best_feat, best_thresh, best_subset, best_gain

def build_tree(X, y, feature_types, max_depth, min_leaf, depth=0, rng=None):
    if rng is None: rng = np.random.default_rng(SEED)
    n = len(y)
    if n < min_leaf or depth == max_depth or gini(y) == 0.0:
        # leaf: majority class (ties broken by smallest class index)
        cls = np.bincount(y).argmax()
        return {"leaf": True, "class": int(cls)}
    feat, thresh, subset, gain = best_split(X, y, feature_types, rng)
    if gain <= 0:
        cls = np.bincount(y).argmax()
        return {"leaf": True, "class": int(cls)}
    if feature_types[feat] == "ordinal":
        left_idx = X[:, feat] <= thresh
    else:
        left_idx = subset[X[:, feat]]
    right_idx = ~left_idx
    left = build_tree(X[left_idx], y[left_idx], feature_types, max_depth, min_leaf, depth+1, rng)
    right = build_tree(X[right_idx], y[right_idx], feature_types, max_depth, min_leaf, depth+1, rng)
    return {"leaf": False, "feature": int(feat), "split_type": 0 if feature_types[feat]=="ordinal" else 1,
            "threshold": thresh, "subset": subset, "left": left, "right": right}

def flatten_bfs(root):
    nodes = []
    queue = [root]
    for node in queue:
        if not node["leaf"]:
            queue.append(node["left"])
            queue.append(node["right"])
    # assign indices
    idx_map = {id(n): i for i, n in enumerate(queue)}
    flat = []
    for node in queue:
        if node["leaf"]:
            flat.append((1, node["class"], 0, 0.0, None, -1, -1))
        else:
            left_i = idx_map[id(node["left"])]
            right_i = idx_map[id(node["right"])]
            if node["split_type"] == 0:
                flat.append((0, 0, node["feature"], node["threshold"], None, left_i, right_i))
            else:
                flat.append((0, 0, node["feature"], 0.0, node["subset"], left_i, right_i))
    return flat

def write_weights(flat_nodes, d, C, path):
    num_nodes = len(flat_nodes)
    # magic, num_nodes, C
    out = struct.pack("<3I", 0x44544C31, num_nodes, C)
    for node in flat_nodes:
        is_leaf, cls, feat, thresh, subset, left, right = node
        out += struct.pack("<BBB", is_leaf, cls if is_leaf else 0, feat if not is_leaf else 0)
        if is_leaf:
            out += b"\x00"  # padding to 4-byte align? we'll keep simple: 3 bytes then next node starts at next byte
            # but for simplicity in C, we'll pack each node as fixed size: let's use fixed size per node.
    # The above variable packing is messy; let's define a fixed-size node struct in Python matching C.
    # We'll redo: each node 48 bytes: uint8 is_leaf, uint8 class, uint8 feature, uint8 split_type,
    # float32 threshold, uint8 subset[32], int32 left, int32 right.
    # For leaf nodes, only is_leaf and class used; rest zero.
    out = struct.pack("<3I", 0x44544C31, num_nodes, C)
    for node in flat_nodes:
        is_leaf, cls, feat, thresh, subset, left, right = node
        split_type = 0
        if not is_leaf:
            split_type = 0 if thresh != 0.0 or subset is None else 1  # actually we stored split_type in node[3]? Let's fix.
    # Let's redesign node tuple: (is_leaf, class, feature, split_type, threshold, subset, left, right)
    pass  # We'll implement properly in the final code below.

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    feature_types = card["input"]["types"]
    # tuning split: last 20%
    n = len(X)
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_tree = None
    for max_depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_LEAF_GRID:
            tree = build_tree(X_train, y_train, feature_types, max_depth, min_leaf, rng=rng)
            flat = flatten_bfs(tree)
            # evaluate on tune
            correct = 0
            for xi, yi in zip(X_tune, y_tune):
                node_idx = 0
                while True:
                    node = flat[node_idx]
                    if node[0]:  # is_leaf
                        pred = node[1]
                        break
                    feat = node[2]
                    split_type = node[3]
                    if split_type == 0:
                        thresh = node[4]
                        if xi[feat] <= thresh:
                            node_idx = node[6]
                        else:
                            node_idx = node[7]
                    else:
                        subset = node[5]
                        if subset[xi[feat]]:
                            node_idx = node[6]
                        else:
                            node_idx = node[7]
                if pred == yi:
                    correct += 1
            acc = correct / len(y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_tree = tree
    # retrain on full data with best params
    max_depth, min_leaf = best_params
    final_tree = build_tree(X, y, feature_types, max_depth, min_leaf, rng=rng)
    flat = flatten_bfs(final_tree)
    # write weights with fixed-size node (48 bytes)
    num_nodes = len(flat)
    out = struct.pack("<3I", 0x44544C31, num_nodes, C)
    for node in flat:
        is_leaf, cls, feat, split_type, thresh, subset, left, right = node
        if is_leaf:
            out += struct.pack("<BBB", 1, cls, 0)  # is_leaf=1, class, feature=0
            out += b"\x00" * 1  # split_type (unused)
            out += struct.pack("<f", 0.0)  # threshold
            out += b"\x00" * 32  # subset
            out += struct.pack("<ii", -1, -1)
        else:
            out += struct.pack("<BBBB", 0, 0, feat, split_type)
            out += struct.pack("<f", thresh if split_type==0 else 0.0)
            if split_type == 1:
                out += subset.astype(np.uint8).tobytes()
            else:
                out += b"\x00" * 32
            out += struct.pack("<ii", left, right)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-dt inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d from card). Output: one byte class per record.
 * Weights format: uint32 magic=0x44544C31, uint32 num_nodes, uint32 C;
 * per node (48 bytes): uint8 is_leaf, uint8 class, uint8 feature, uint8 split_type,
 *                      float32 threshold, uint8 subset[32], int32 left, int32 right.
 * BFS order, root at index 0. */
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
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

typedef struct {
    uint8_t is_leaf;
    uint8_t cls;
    uint8_t feature;
    uint8_t split_type;
    float threshold;
    uint8_t subset[32];
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t*)wb;
    if (hdr[0] != 0x44544C31) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t num_nodes = hdr[1], C = hdr[2];
    if (C != 2 || nw != 12 + (long)num_nodes * 48) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node*)(wb + 12);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }
        int idx = 0;
        while (1) {
            Node *nd = &nodes[idx];
            if (nd->is_leaf) {
                put_rec(stdout, &nd->cls, 1);
                break;
            }
            uint8_t v = xi[nd->feature];
            if (nd->split_type == 0) {
                if (v <= nd->threshold) idx = nd->left;
                else idx = nd->right;
            } else {
                if (nd->subset[v>>3] & (1u << (v&7))) idx = nd->left;
                else idx = nd->right;
            }
        }
    }
    return 0;
}
```