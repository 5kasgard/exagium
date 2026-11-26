```hypothesis
Family: label-tree (single decision tree with ordinal/categorical splits).
Why: Adult has mixed ordinal and categorical features; a tree natively handles both without one-hot expansion, giving non-linear decision boundaries at very low inference cost (a few branch comparisons per record). Tuning max_depth on the held-out 20% should beat linear logistic regression (0.8524) while staying far cheaper than MLP/GBDT.
Failure: If dev accuracy ≤ 0.8524 or instructions/item ≥ label-linear (719 Ir/item), the model does not justify its existence.
```

```python
"""label-tree: single decision tree for tabular label tasks with byte features.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Encodes ordinal splits as <= threshold, categorical splits as == value.
Tunes max_depth on last 20% of train (file order), then trains on all data.
Weights format: magic(0x54524545), d, num_nodes, then per node:
  uint8 feature_idx, uint8 split_type (0=ordinal_le,1=cat_eq,2=leaf),
  uint8 threshold (split value or leaf class), uint8 pad,
  int32 left, int32 right (child indices, -1 for none)."""
import json, struct, sys
import numpy as np

SEED = 1
VAL_FRACTION = 0.2
MAX_DEPTH_CANDIDATES = list(range(2, 11))  # 2..10
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5

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

def weighted_gini(y_left, y_right):
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    return (len(y_left) * gini(y_left) + len(y_right) * gini(y_right)) / n

def find_best_split(X, y, feat_types):
    n, d = X.shape
    best_gini = 1.0
    best_split = None  # (feat_idx, split_type, threshold, left_mask)
    parent_gini = gini(y)
    for j in range(d):
        vals = X[:, j]
        if feat_types[j] == "ordinal":
            # try thresholds from unique values (0..254)
            uniq = np.unique(vals)
            for t in uniq:
                if t == 255:  # unseen, treat as > any threshold
                    continue
                left = vals <= t
                right = ~left
                if left.sum() < MIN_SAMPLES_LEAF or right.sum() < MIN_SAMPLES_LEAF:
                    continue
                g = weighted_gini(y[left], y[right])
                if g < best_gini:
                    best_gini = g
                    best_split = (j, 0, t, left)
        else:  # categorical
            # try equality split for each category present
            uniq = np.unique(vals)
            for v in uniq:
                if v == 255:  # unseen, treat as != any category
                    continue
                left = vals == v
                right = ~left
                if left.sum() < MIN_SAMPLES_LEAF or right.sum() < MIN_SAMPLES_LEAF:
                    continue
                g = weighted_gini(y[left], y[right])
                if g < best_gini:
                    best_gini = g
                    best_split = (j, 1, v, left)
    if best_split is None or best_gini >= parent_gini - 1e-12:
        return None
    return best_split

def build_tree(X, y, feat_types, max_depth):
    n, d = X.shape
    nodes = []  # each node: dict with keys: feat, type, thresh, left, right, leaf_class
    # stack entries: (node_index, X_idx, y_idx, depth)
    # we'll store indices into X,y for each node
    root_idx = 0
    nodes.append({"feat": -1, "type": -1, "thresh": -1, "left": -1, "right": -1, "leaf_class": -1})
    stack = [(root_idx, np.arange(n), 0)]
    while stack:
        node_idx, idx, depth = stack.pop()
        y_node = y[idx]
        # check stopping criteria
        if depth >= max_depth or len(idx) < MIN_SAMPLES_SPLIT or gini(y_node) == 0.0:
            # make leaf
            classes, counts = np.unique(y_node, return_counts=True)
            leaf_class = classes[np.argmax(counts)]
            nodes[node_idx]["leaf_class"] = int(leaf_class)
            continue
        split = find_best_split(X[idx], y_node, feat_types)
        if split is None:
            classes, counts = np.unique(y_node, return_counts=True)
            leaf_class = classes[np.argmax(counts)]
            nodes[node_idx]["leaf_class"] = int(leaf_class)
            continue
        feat_idx, split_type, thresh, left_mask = split
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        # create children
        left_child = len(nodes)
        nodes.append({"feat": -1, "type": -1, "thresh": -1, "left": -1, "right": -1, "leaf_class": -1})
        right_child = len(nodes)
        nodes.append({"feat": -1, "type": -1, "thresh": -1, "left": -1, "right": -1, "leaf_class": -1})
        nodes[node_idx]["feat"] = feat_idx
        nodes[node_idx]["type"] = split_type
        nodes[node_idx]["thresh"] = int(thresh)
        nodes[node_idx]["left"] = left_child
        nodes[node_idx]["right"] = right_child
        # push children (right first so left processed first? order doesn't matter)
        stack.append((right_child, right_idx, depth + 1))
        stack.append((left_child, left_idx, depth + 1))
    return nodes

def evaluate_tree(nodes, X, y):
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        node = 0
        while nodes[node]["leaf_class"] == -1:
            feat = nodes[node]["feat"]
            val = X[i, feat]
            if nodes[node]["type"] == 0:  # ordinal <=
                if val <= nodes[node]["thresh"] or val == 255:
                    node = nodes[node]["left"]
                else:
                    node = nodes[node]["right"]
            else:  # categorical ==
                if val == nodes[node]["thresh"]:
                    node = nodes[node]["left"]
                else:
                    node = nodes[node]["right"]
        preds[i] = nodes[node]["leaf_class"]
    return (preds == y).mean()

def nodes_to_bytes(nodes, d):
    # header: magic, d, num_nodes
    magic = 0x54524545  # "TREE"
    num_nodes = len(nodes)
    header = struct.pack("<3I", magic, d, num_nodes)
    body = bytearray()
    for nd in nodes:
        if nd["leaf_class"] != -1:
            feat = 255
            stype = 2
            thresh = nd["leaf_class"]
            left = -1
            right = -1
        else:
            feat = nd["feat"]
            stype = nd["type"]
            thresh = nd["thresh"]
            left = nd["left"]
            right = nd["right"]
        body.extend(struct.pack("<BBBbii", feat, stype, thresh, 0, left, right))
    return header + body

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    feat_types = card["input"]["types"]
    d = X.shape[1]
    n = X.shape[0]
    # split last 20% for validation (file order)
    val_size = int(n * VAL_FRACTION)
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    # tune max_depth
    best_depth = MAX_DEPTH_CANDIDATES[0]
    best_acc = -1.0
    for depth in MAX_DEPTH_CANDIDATES:
        nodes = build_tree(X_train, y_train, feat_types, depth)
        acc = evaluate_tree(nodes, X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_depth = depth
    # retrain on all data
    final_nodes = build_tree(X, y, feat_types, best_depth)
    out_bytes = nodes_to_bytes(final_nodes, d)
    open(sys.argv[4], "wb").write(out_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: 1 byte class.
 * Weights format: magic(0x54524545), d, num_nodes,
 * then per node: uint8 feat, uint8 type (0=ordinal_le,1=cat_eq,2=leaf),
 * uint8 thresh, uint8 pad, int32 left, int32 right. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t feat;
    uint8_t type;
    uint8_t thresh;
    uint8_t pad;
    int32_t left;
    int32_t right;
} Node;

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
    if (nw < 12) { fprintf(stderr, "bad weights (too small)\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t d = wb[4] | wb[5]<<8 | wb[6]<<16 | wb[7]<<24;
    uint32_t num_nodes = wb[8] | wb[9]<<8 | wb[10]<<16 | wb[11]<<24;
    long expected = 12 + (long)num_nodes * 16;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 12);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        int node_idx = 0;
        while (nodes[node_idx].type != 2) {
            uint8_t feat = nodes[node_idx].feat;
            uint8_t val = xi[feat];
            if (nodes[node_idx].type == 0) { /* ordinal <= */
                if (val <= nodes[node_idx].thresh || val == 255)
                    node_idx = nodes[node_idx].left;
                else
                    node_idx = nodes[node_idx].right;
            } else { /* categorical == */
                if (val == nodes[node_idx].thresh)
                    node_idx = nodes[node_idx].left;
                else
                    node_idx = nodes[node_idx].right;
            }
        }
        unsigned char cls = nodes[node_idx].thresh;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```