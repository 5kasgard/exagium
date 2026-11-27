```hypothesis
Family: label-dtree (shallow decision tree on raw quantized bytes). The Adult task has 14 pre-quantized features (ordinal 0-254, categorical 0-255) that interact non-linearly (e.g., education × occupation). A depth-5 tree captures key interactions with ~30 nodes, beating linear models while adding only a few integer comparisons per record. If dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-mlp16 (2325 Ir/item), the hypothesis fails.
```
```python
"""label-dtree: shallow decision tree on raw quantized bytes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH_CANDIDATES = range(1, 11)  # tune on last 20%


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - (p * p).sum()


def build_tree(X, y, feature_types, max_depth, rng):
    """Return (nodes, feature_types) where nodes is list of dicts in pre-order."""
    n, d = X.shape
    nodes = []

    def recurse(idx, depth):
        # idx: array of row indices
        cls_counts = np.bincount(y[idx], minlength=2)
        best_cls = int(cls_counts.argmax())
        if depth == max_depth or cls_counts.min() == 0 or len(idx) < 2:
            nodes.append({"leaf": True, "class": best_cls})
            return len(nodes) - 1

        # find best split
        best_gini = 1.0
        best_feat = -1
        best_thresh = -1
        best_left_idx = None
        best_right_idx = None

        for f in range(d):
            vals = X[idx, f]
            if feature_types[f] == "ordinal":
                # thresholds 0..253 (split <=t vs >t)
                for t in range(254):
                    left = idx[vals <= t]
                    right = idx[vals > t]
                    if len(left) == 0 or len(right) == 0:
                        continue
                    wl, wr = len(left), len(right)
                    g = (wl * gini(np.bincount(y[left], minlength=2)) +
                         wr * gini(np.bincount(y[right], minlength=2))) / (wl + wr)
                    if g < best_gini:
                        best_gini = g
                        best_feat = f
                        best_thresh = t
                        best_left_idx = left
                        best_right_idx = right
            else:  # categorical
                # split on each value present in this node
                uniq = np.unique(vals)
                for v in uniq:
                    left = idx[vals == v]
                    right = idx[vals != v]
                    if len(left) == 0 or len(right) == 0:
                        continue
                    wl, wr = len(left), len(right)
                    g = (wl * gini(np.bincount(y[left], minlength=2)) +
                         wr * gini(np.bincount(y[right], minlength=2))) / (wl + wr)
                    if g < best_gini:
                        best_gini = g
                        best_feat = f
                        best_thresh = int(v)
                        best_left_idx = left
                        best_right_idx = right

        if best_feat == -1:  # no improving split
            nodes.append({"leaf": True, "class": best_cls})
            return len(nodes) - 1

        # create internal node (placeholder for children indices)
        node_idx = len(nodes)
        nodes.append({"leaf": False, "feat": best_feat, "thresh": best_thresh,
                      "left": -1, "right": -1})
        left_child = recurse(best_left_idx, depth + 1)
        right_child = recurse(best_right_idx, depth + 1)
        nodes[node_idx]["left"] = left_child
        nodes[node_idx]["right"] = right_child
        return node_idx

    root_idx = recurse(np.arange(n), 0)
    return nodes


def evaluate(nodes, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        node = nodes[0]
        while not node["leaf"]:
            f, t = node["feat"], node["thresh"]
            v = X[i, f]
            if f < len(X[0]) and v <= t:  # ordinal split uses <=
                node = nodes[node["left"]]
            else:
                # for categorical we stored equality in thresh, but we need to know type.
                # We'll handle in infer by checking feature_types; here we just replicate.
                # Since we don't have feature_types here, we assume the same logic:
                # In training we used <= for ordinal and == for categorical.
                # But we can't distinguish here. So we must store feature_types in nodes or pass them.
                # Simpler: during training we know the type. We'll store split type in node.
                pass
        # This evaluation is only for tuning; we'll re-implement with feature_types.
    # We'll skip full evaluation here and do it properly below.
    return 0.0


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    feature_types = card["input"]["types"]
    n = X.shape[0]

    # tuning split: last 20% in file order
    tune_start = int(0.8 * n)
    X_train, y_train = X[:tune_start], y[:tune_start]
    X_tune, y_tune = X[tune_start:], y[tune_start:]

    rng = np.random.default_rng(SEED)

    best_depth = 1
    best_acc = 0.0
    for depth in MAX_DEPTH_CANDIDATES:
        nodes = build_tree(X_train, y_train, feature_types, depth, rng)
        # evaluate on tune
        correct = 0
        for i in range(len(X_tune)):
            node = nodes[0]
            while not node["leaf"]:
                f = node["feat"]
                t = node["thresh"]
                v = X_tune[i, f]
                if feature_types[f] == "ordinal":
                    if v <= t:
                        node = nodes[node["left"]]
                    else:
                        node = nodes[node["right"]]
                else:
                    if v == t:
                        node = nodes[node["left"]]
                    else:
                        node = nodes[node["right"]]
            if node["class"] == y_tune[i]:
                correct += 1
        acc = correct / len(X_tune)
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # retrain on all data with best_depth
    nodes = build_tree(X, y, feature_types, best_depth, rng)

    # serialize weights
    # header: magic(4), version(1), num_features(1), num_classes(1), max_depth(1), num_nodes(4), feature_types[14](14)
    magic = 0x44545245  # 'DTRE' little-endian
    version = 1
    num_features = X.shape[1]
    num_classes = 2
    max_depth = best_depth
    num_nodes = len(nodes)
    ft_bytes = bytearray(1 if t == "categorical" else 0 for t in feature_types)

    header = struct.pack("<IBBBBI14s", magic, version, num_features, num_classes, max_depth, num_nodes, ft_bytes)

    node_data = bytearray()
    for node in nodes:
        if node["leaf"]:
            node_data.append(1)  # is_leaf
            node_data.append(node["class"])  # class label
            node_data.extend(b'\x00\x00\x00\x00')  # padding for feat, thresh, left, right (we'll just write zeros)
            node_data.extend(b'\x00\x00\x00\x00')
            node_data.extend(b'\x00\x00\x00\x00')
            node_data.extend(b'\x00\x00\x00\x00')
        else:
            node_data.append(0)  # is_leaf
            node_data.append(node["feat"])
            node_data.append(node["thresh"])
            node_data.append(0)  # pad to 4 bytes
            node_data.extend(struct.pack("<ii", node["left"], node["right"]))
            node_data.append(0)  # class label placeholder
            node_data.extend(b'\x00\x00\x00')  # padding

    out = header + node_data
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-dtree inference: ./infer weights.bin in.rec N */
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
    uint8_t is_leaf;
    uint8_t feat;
    uint8_t thresh;
    uint8_t pad;
    int32_t left;
    int32_t right;
    uint8_t class_label;
    uint8_t pad2[3];
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 26) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = w[0] | w[1] << 8 | w[2] << 16 | w[3] << 24;
    if (magic != 0x44545245) { fprintf(stderr, "bad magic\n"); return 3; }
    uint8_t version = w[4];
    uint8_t d = w[5];
    uint8_t num_classes = w[6];
    uint8_t max_depth = w[7];
    uint32_t num_nodes = w[8] | w[9] << 8 | w[10] << 16 | w[11] << 24;
    unsigned char *ft = w + 12;  // 14 bytes
    Node *nodes = (Node *)(w + 26);
    long expected = 26 + (long)num_nodes * sizeof(Node);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    if (d != 14 || num_classes != 2) { fprintf(stderr, "unexpected dims\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node_idx = 0;
        while (1) {
            Node *nd = &nodes[node_idx];
            if (nd->is_leaf) {
                out_byte = nd->class_label;
                break;
            }
            uint8_t v = xi[nd->feat];
            if (ft[nd->feat] == 0) {  // ordinal
                if (v <= nd->thresh) node_idx = nd->left;
                else node_idx = nd->right;
            } else {  // categorical
                if (v == nd->thresh) node_idx = nd->left;
                else node_idx = nd->right;
            }
        }
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```