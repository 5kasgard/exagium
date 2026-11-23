```hypothesis
Family: label-dtree (single deterministic decision tree with one-hot categorical handling).
Why: Adult has mixed ordinal/categorical features; a shallow tree captures non-linear interactions that logistic regression misses, while staying faster than boosted ensembles. The one-hot encoding matches the card's quantile/category representation, and a depth-6 tree should exceed linear's 0.8524 dev accuracy with only ~100 nodes.
Failure sign: Dev accuracy ≤ 0.8524 or inference > 1000 ns/item.
```

```python
"""label-dtree: single decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-scored float, categorical as one-hot over training values (255=unseen).
Tunes max_depth/min_samples_leaf on last 20% of train.in, then trains on all.
Exports: magic 0x44545245, d, C, n_feat, n_nodes; per-column preprocessing; tree nodes."""
import json, math, struct, sys

import numpy as np

SEED = 1
TUNE_FRAC = 0.2
DEPTH_GRID = [3, 4, 5, 6, 7, 8, 9, 10]
LEAF_GRID = [1, 5, 10, 20, 50]


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
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / len(y)
    return 1.0 - np.sum(p * p)


def best_split(X, y, feat_idx, min_leaf):
    """Return (threshold, left_idx, right_idx, gain) or None."""
    n = len(y)
    parent_gini = gini(y)
    best_gain = -1.0
    best_thr = None
    best_l = best_r = None
    col = X[:, feat_idx]
    # unique sorted values
    vals = np.unique(col)
    if len(vals) == 1:
        return None
    # candidate thresholds: midpoints
    for i in range(len(vals) - 1):
        thr = (vals[i] + vals[i + 1]) * 0.5
        left = col <= thr
        right = ~left
        if left.sum() < min_leaf or right.sum() < min_leaf:
            continue
        g = parent_gini - (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / n
        if g > best_gain + 1e-12:
            best_gain = g
            best_thr = thr
            best_l = left
            best_r = right
    if best_gain <= 0:
        return None
    return best_thr, best_l, best_r, best_gain


def build_tree(X, y, max_depth, min_leaf, depth=0, rng=None):
    """Returns list of nodes (dicts) in BFS order, root at index 0."""
    nodes = []
    # stack of (node_idx, sample_indices, depth)
    stack = [(0, np.arange(len(y)), 0)]
    node_list = []  # will fill as we go
    while stack:
        node_idx, idx, d = stack.pop()
        if len(idx) == 0:
            # should not happen
            node_list.append({"leaf": True, "class": int(np.bincount(y).argmax())})
            continue
        # stopping criteria
        if d >= max_depth or len(idx) < 2 * min_leaf or gini(y[idx]) == 0.0:
            cls = int(np.bincount(y[idx]).argmax())
            node_list.append({"leaf": True, "class": cls})
            continue
        # find best split over all features
        best_feat = -1
        best_thr = 0.0
        best_gain = -1.0
        best_left = best_right = None
        for f in range(X.shape[1]):
            res = best_split(X[idx], y[idx], f, min_leaf)
            if res is not None:
                thr, left_mask, right_mask, gain = res
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feat = f
                    best_thr = thr
                    best_left = idx[left_mask]
                    best_right = idx[right_mask]
        if best_feat == -1:
            cls = int(np.bincount(y[idx]).argmax())
            node_list.append({"leaf": True, "class": cls})
            continue
        # internal node
        node_list.append({"leaf": False, "feat": best_feat, "thr": best_thr,
                          "left": len(node_list) + 1, "right": None})  # right will be set after left subtree
        left_idx = len(node_list)
        # push right then left so left is processed next (DFS)
        stack.append((left_idx + 1, best_right, d + 1))  # placeholder for right
        stack.append((left_idx, best_left, d + 1))
        # we need to fix right index after left subtree is built; we'll do a second pass
        # Instead, we can build recursively. Let's switch to recursive for simplicity.
    # The iterative approach with placeholder is messy; use recursion.
    return None  # placeholder


# Recursive version
def build_tree_rec(X, y, max_depth, min_leaf, depth=0):
    nodes = []
    def recurse(idx, d):
        nonlocal nodes
        if len(idx) == 0:
            cls = int(np.bincount(y).argmax())
            nodes.append({"leaf": True, "class": cls})
            return len(nodes) - 1
        if d >= max_depth or len(idx) < 2 * min_leaf or gini(y[idx]) == 0.0:
            cls = int(np.bincount(y[idx]).argmax())
            nodes.append({"leaf": True, "class": cls})
            return len(nodes) - 1
        best_feat = -1
        best_thr = 0.0
        best_gain = -1.0
        best_left = best_right = None
        for f in range(X.shape[1]):
            res = best_split(X[idx], y[idx], f, min_leaf)
            if res is not None:
                thr, left_mask, right_mask, gain = res
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feat = f
                    best_thr = thr
                    best_left = idx[left_mask]
                    best_right = idx[right_mask]
        if best_feat == -1:
            cls = int(np.bincount(y[idx]).argmax())
            nodes.append({"leaf": True, "class": cls})
            return len(nodes) - 1
        node_idx = len(nodes)
        nodes.append({"leaf": False, "feat": best_feat, "thr": best_thr,
                      "left": None, "right": None})  # fill later
        left_idx = recurse(best_left, d + 1)
        right_idx = recurse(best_right, d + 1)
        nodes[node_idx]["left"] = left_idx
        nodes[node_idx]["right"] = right_idx
        return node_idx
    recurse(np.arange(len(y)), 0)
    return nodes


def evaluate(nodes, X, y):
    preds = np.zeros(len(y), dtype=np.int64)
    for i in range(len(y)):
        x = X[i]
        node = 0
        while True:
            nd = nodes[node]
            if nd["leaf"]:
                preds[i] = nd["class"]
                break
            if x[nd["feat"]] <= nd["thr"]:
                node = nd["left"]
            else:
                node = nd["right"]
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    n = len(y)

    # Preprocessing
    ordinal_cols = []
    cat_cols = []
    cat_maps = []  # list of (n_cat, lookup[256])
    cat_offsets = []  # start index in feature vector for each categorical column
    ord_params = []  # (mean, std)
    feat_dim = 0
    for j, t in enumerate(types):
        col = X_raw[:, j].astype(np.float64)
        if t == "ordinal":
            mu = col.mean()
            sd = col.std()
            if sd == 0:
                sd = 1.0
            ord_params.append((mu, sd))
            ordinal_cols.append(j)
            feat_dim += 1
        else:
            # categorical
            vals = np.unique(col)
            # ensure 255 (unseen) is included
            if 255 not in vals:
                vals = np.append(vals, 255)
            vals.sort()
            n_cat = len(vals)
            lookup = np.full(256, n_cat - 1, dtype=np.uint16)  # default to unseen index
            for idx, v in enumerate(vals):
                lookup[v] = idx
            cat_maps.append((n_cat, lookup))
            cat_cols.append(j)
            cat_offsets.append(feat_dim)
            feat_dim += n_cat

    # Build feature matrix
    X = np.zeros((n, feat_dim), dtype=np.float32)
    # ordinal
    for k, j in enumerate(ordinal_cols):
        mu, sd = ord_params[k]
        X[:, k] = (X_raw[:, j] - mu) / sd
    # categorical
    for k, j in enumerate(cat_cols):
        off = cat_offsets[k]
        n_cat, lookup = cat_maps[k]
        idx = lookup[X_raw[:, j]]
        X[np.arange(n), off + idx] = 1.0

    # Tuning split (last 20% in file order)
    tune_start = int(n * (1 - TUNE_FRAC))
    train_idx = np.arange(tune_start)
    tune_idx = np.arange(tune_start, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Grid search
    best_acc = -1.0
    best_params = (6, 10)
    rng = np.random.default_rng(SEED)
    for max_depth in DEPTH_GRID:
        for min_leaf in LEAF_GRID:
            nodes = build_tree_rec(X_train, y_train, max_depth, min_leaf)
            acc = evaluate(nodes, X_tune, y_tune)
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_params = (max_depth, min_leaf)
    print(f"Tuned: depth={best_params[0]}, min_leaf={best_params[1]}, tune_acc={best_acc:.4f}", file=sys.stderr)

    # Train on full data
    nodes = build_tree_rec(X, y, *best_params)
    print(f"Tree nodes: {len(nodes)}", file=sys.stderr)

    # Export weights.bin
    # Magic: 0x44545245 ('ERTD' little-endian)
    # Format:
    #   int32 magic, int32 d, int32 C, int32 n_feat, int32 n_nodes
    #   per column (d times):
    #     uint8 type (0=ordinal, 1=categorical)
    #     if ordinal: float32 mean, float32 std
    #     if categorical: uint16 n_cat, uint16 lookup[256]
    #   per node (n_nodes times):
    #     uint8 is_leaf
    #     if leaf: uint8 class
    #     else: int32 feat, float32 thr, int32 left, int32 right
    out = bytearray()
    out += struct.pack("<iiiii", 0x44545245, d, C, feat_dim, len(nodes))
    for j, t in enumerate(types):
        if t == "ordinal":
            mu, sd = ord_params[ordinal_cols.index(j)]
            out += struct.pack("<Bff", 0, mu, sd)
        else:
            k = cat_cols.index(j)
            n_cat, lookup = cat_maps[k]
            out += struct.pack("<BH", 1, n_cat)
            out += lookup.astype("<u2").tobytes()
    for nd in nodes:
        if nd["leaf"]:
            out += struct.pack("<BB", 1, nd["class"])
        else:
            out += struct.pack("<Bifi", 0, nd["feat"], nd["thr"], nd["left"], nd["right"])
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-dtree inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output record: 1 byte class.
 * Weights format: see train.py. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    uint8_t type;        // 0 ordinal, 1 categorical
    float mean, std;     // ordinal
    uint16_t n_cat;      // categorical
    uint16_t *lookup;    // size 256
    uint32_t feat_off;   // feature vector offset
    uint32_t feat_len;   // 1 for ordinal, n_cat for categorical
} Column;

typedef struct {
    uint8_t is_leaf;
    uint8_t cls;         // if leaf
    uint32_t feat;       // if internal
    float thr;
    uint32_t left, right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x44545245) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2], n_feat = hd[3], n_nodes = hd[4];
    if (C != 2 || d != 14) { fprintf(stderr, "unexpected d/C\n"); return 3; }
    unsigned char *ptr = wb + 20;

    Column cols[14];
    uint32_t feat_dim = 0;
    for (int j = 0; j < d; j++) {
        cols[j].type = *ptr++;
        cols[j].feat_off = feat_dim;
        if (cols[j].type == 0) { // ordinal
            cols[j].mean = *(float *)ptr; ptr += 4;
            cols[j].std  = *(float *)ptr; ptr += 4;
            cols[j].feat_len = 1;
            feat_dim += 1;
        } else { // categorical
            cols[j].n_cat = *(uint16_t *)ptr; ptr += 2;
            cols[j].lookup = (uint16_t *)ptr; ptr += 256 * 2;
            cols[j].feat_len = cols[j].n_cat;
            feat_dim += cols[j].n_cat;
        }
    }
    if (feat_dim != (uint32_t)n_feat) { fprintf(stderr, "feat dim mismatch\n"); return 3; }

    Node *nodes = malloc(n_nodes * sizeof(Node));
    for (int i = 0; i < n_nodes; i++) {
        nodes[i].is_leaf = *ptr++;
        if (nodes[i].is_leaf) {
            nodes[i].cls = *ptr++;
            nodes[i].feat = nodes[i].left = nodes[i].right = 0;
            nodes[i].thr = 0.0f;
        } else {
            nodes[i].feat = *(uint32_t *)ptr; ptr += 4;
            nodes[i].thr = *(float *)ptr; ptr += 4;
            nodes[i].left = *(uint32_t *)ptr; ptr += 4;
            nodes[i].right = *(uint32_t *)ptr; ptr += 4;
        }
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *features = malloc(feat_dim * sizeof(float));
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(features, 0, feat_dim * sizeof(float));
        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            Column *col = &cols[j];
            if (col->type == 0) {
                features[col->feat_off] = (v - col->mean) / col->std;
            } else {
                uint16_t idx = col->lookup[v];
                features[col->feat_off + idx] = 1.0f;
            }
        }
        uint32_t node = 0;
        while (1) {
            Node *nd = &nodes[node];
            if (nd->is_leaf) {
                out_byte = nd->cls;
                break;
            }
            if (features[nd->feat] <= nd->thr) node = nd->left;
            else node = nd->right;
        }
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```