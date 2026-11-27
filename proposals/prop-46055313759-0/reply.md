```hypothesis
Family: label-tree (single decision tree, depth ≤ 6, Gini splits on all 256 thresholds per feature).
Why: Adult has mixed ordinal/categorical features; a shallow tree captures non-linear interactions that linear models miss, while staying extremely fast at inference (only a few integer comparisons per record). Training is deterministic via exhaustive histogram-based split search on the tuning split (last 20% of train.in) to pick max_depth ∈ {3,4,5,6}, then retraining on all data.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or inference cost ≥ 719 Ir/item.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH_CANDIDATES = [3, 4, 5, 6]
MIN_SAMPLES_LEAF = 5
MIN_SAMPLES_SPLIT = 10

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(counts):
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return 1.0 - np.sum(p * p)

def build_tree(X, y, max_depth, depth=0):
    n, d = X.shape
    if n < MIN_SAMPLES_LEAF or depth == max_depth:
        return {"leaf": True, "class": int(np.bincount(y).argmax())}
    if np.all(y == y[0]):
        return {"leaf": True, "class": int(y[0])}

    parent_gini = gini(np.bincount(y, minlength=2))
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1

    for feat in range(d):
        vals = X[:, feat]
        # histogram of class counts per feature value (0..255)
        hist = np.zeros((256, 2), dtype=np.int32)
        np.add.at(hist, (vals, y), 1)
        cum = hist.cumsum(axis=0)
        total_left = cum[:-1]
        total_right = cum[-1] - total_left
        n_left = total_left.sum(axis=1)
        n_right = total_right.sum(axis=1)
        mask = (n_left >= MIN_SAMPLES_LEAF) & (n_right >= MIN_SAMPLES_LEAF)
        if not mask.any():
            continue
        gini_left = 1.0 - np.sum((total_left / n_left[:, None]) ** 2, axis=1)
        gini_right = 1.0 - np.sum((total_right / n_right[:, None]) ** 2, axis=1)
        weighted_gini = (n_left * gini_left + n_right * gini_right) / n
        gain = parent_gini - weighted_gini
        gain[~mask] = -1
        idx = np.argmax(gain)
        if gain[idx] > best_gain:
            best_gain = gain[idx]
            best_feat = feat
            best_thresh = idx

    if best_gain <= 0:
        return {"leaf": True, "class": int(np.bincount(y).argmax())}

    left_mask = X[:, best_feat] <= best_thresh
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], max_depth, depth + 1)
    right = build_tree(X[right_mask], y[right_mask], max_depth, depth + 1)
    return {"leaf": False, "feat": best_feat, "thresh": best_thresh, "left": left, "right": right}

def flatten_tree(node, nodes):
    idx = len(nodes)
    if node["leaf"]:
        nodes.append((1, 0, 0, node["class"], 0, 0))
    else:
        left_idx = flatten_tree(node["left"], nodes)
        right_idx = flatten_tree(node["right"], nodes)
        nodes.append((0, node["feat"], node["thresh"], 0, left_idx, right_idx))
    return idx

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = X.shape[0]
    tune_size = n // 5
    tune_idx = slice(n - tune_size, n)
    train_idx = slice(0, n - tune_size)

    best_depth = MAX_DEPTH_CANDIDATES[0]
    best_acc = -1.0
    for depth in MAX_DEPTH_CANDIDATES:
        tree = build_tree(X[train_idx], y[train_idx], depth)
        nodes = []
        flatten_tree(tree, nodes)
        # evaluate on tuning split
        correct = 0
        for i in range(tune_idx.start, tune_idx.stop):
            xi = X[i]
            node_idx = 0
            while True:
                is_leaf, feat, thresh, cls, left, right = nodes[node_idx]
                if is_leaf:
                    pred = cls
                    break
                if xi[feat] <= thresh:
                    node_idx = left
                else:
                    node_idx = right
            if pred == y[i]:
                correct += 1
        acc = correct / tune_size
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # retrain on all data
    tree = build_tree(X, y, best_depth)
    nodes = []
    flatten_tree(tree, nodes)

    # write weights.bin
    magic = 0x54524531  # "TRE1"
    d = X.shape[1]
    C = card["classes"]
    n_nodes = len(nodes)
    header = struct.pack("<4i", magic, d, C, n_nodes)
    body = bytearray()
    for (is_leaf, feat, thresh, cls, left, right) in nodes:
        body += struct.pack("<BBBBII", is_leaf, feat, thresh, cls, left, right)
    open(sys.argv[4], "wb").write(header + body)

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
    uint8_t is_leaf;
    uint8_t feat;
    uint8_t thresh;
    uint8_t cls;
    uint32_t left;
    uint32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x54524531) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], n_nodes = hd[3];
    if (d != 14 || C != 2 || n_nodes <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16L + (long)n_nodes * 16L;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        uint32_t node_idx = 0;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.is_leaf) {
                put_rec(stdout, &nd.cls, 1);
                break;
            }
            if (xi[nd.feat] <= nd.thresh)
                node_idx = nd.left;
            else
                node_idx = nd.right;
        }
    }
    return 0;
}
```