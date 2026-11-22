```hypothesis
Family: label-dtree (single CART decision tree). Decision trees naturally capture pixel interactions (strokes, loops) that linear models miss, and a depth-limited tree stays tiny and fast. We tune max_depth on the last 20% of train.in, then refit on all data. If dev accuracy ≤ 0.9613 (label-linear), the family does not justify its existence.
```
```python
"""label-dtree: single decision tree for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread. Tunes max_depth on last 20% of train, then trains on all.
Exports: magic 0x44545245, n_nodes, n_feat, n_class; then per-node arrays (int16 feat, float32 thresh, int32 left, int32 right, uint8 leaf, uint8 class)."""
import json, math, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
MAX_DEPTH_CANDIDATES = list(range(1, 21))

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(y):
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, feat_idx):
    """Return (feat, thresh, left_idx, right_idx, gain) or None."""
    n = len(y)
    parent_gini = gini(y)
    best_gain = -1.0
    best = None
    for f in feat_idx:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) == 1:
            continue
        threshs = (uniq[:-1] + uniq[1:]) / 2.0
        for t in threshs:
            left = vals <= t
            nl, nr = left.sum(), (~left).sum()
            if nl < MIN_SAMPLES_LEAF or nr < MIN_SAMPLES_LEAF:
                continue
            gain = parent_gini - (nl * gini(y[left]) + nr * gini(y[~left])) / n
            if gain > best_gain:
                best_gain = gain
                best = (f, t, left, ~left)
    if best is None or best_gain <= 1e-9:
        return None
    f, t, left, right = best
    return f, t, left, right, best_gain

def build_tree(X, y, depth, max_depth, rng):
    """Returns (nodes_list, class_label) where nodes_list is list of dicts."""
    n = len(y)
    if n < MIN_SAMPLES_SPLIT or depth >= max_depth or gini(y) == 0.0:
        cls = int(np.bincount(y).argmax())
        return [], cls
    feat_idx = rng.permutation(X.shape[1])
    split = best_split(X, y, feat_idx)
    if split is None:
        cls = int(np.bincount(y).argmax())
        return [], cls
    f, t, left, right, _ = split
    left_nodes, left_cls = build_tree(X[left], y[left], depth + 1, max_depth, rng)
    right_nodes, right_cls = build_tree(X[right], y[right], depth + 1, max_depth, rng)
    # current node index will be 0 after we prepend
    left_idx = len(left_nodes)
    right_idx = left_idx + 1 + len(right_nodes)  # +1 for current node
    node = {"feat": f, "thresh": t, "left": left_idx, "right": right_idx, "leaf": 0, "cls": 0}
    return [node] + left_nodes + right_nodes, 0  # cls unused for internal

def flatten(nodes):
    """Convert node list to flat arrays in preorder (as built)."""
    n = len(nodes)
    feat = np.zeros(n, dtype=np.int16)
    thresh = np.zeros(n, dtype=np.float32)
    left = np.zeros(n, dtype=np.int32)
    right = np.zeros(n, dtype=np.int32)
    leaf = np.zeros(n, dtype=np.uint8)
    cls = np.zeros(n, dtype=np.uint8)
    for i, nd in enumerate(nodes):
        feat[i] = nd["feat"]
        thresh[i] = nd["thresh"]
        left[i] = nd["left"]
        right[i] = nd["right"]
        leaf[i] = nd["leaf"]
        cls[i] = nd["cls"]
    return feat, thresh, left, right, leaf, cls

def predict_one(x, feat, thresh, left, right, leaf, cls):
    idx = 0
    while leaf[idx] == 0:
        if x[feat[idx]] <= thresh[idx]:
            idx = left[idx]
        else:
            idx = right[idx]
    return cls[idx]

def accuracy(X, y, feat, thresh, left, right, leaf, cls):
    correct = 0
    for i in range(len(y)):
        if predict_one(X[i], feat, thresh, left, right, leaf, cls) == y[i]:
            correct += 1
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    rng = np.random.default_rng(SEED)

    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    best_depth = 1
    best_acc = -1.0
    for d in MAX_DEPTH_CANDIDATES:
        nodes, _ = build_tree(X_tr, y_tr, 0, d, rng)
        if not nodes:
            acc = np.mean(y_tr == np.bincount(y_tr).argmax())
        else:
            feat, thresh, left, right, leaf, cls = flatten(nodes)
            acc = accuracy(X_val, y_val, feat, thresh, left, right, leaf, cls)
        if acc > best_acc:
            best_acc, best_depth = acc, d

    # final train on all data
    nodes, _ = build_tree(X, y, 0, best_depth, rng)
    if not nodes:
        # degenerate: single leaf
        leaf_cls = int(np.bincount(y).argmax())
        n_nodes = 1
        feat = np.array([0], dtype=np.int16)
        thresh = np.array([0.0], dtype=np.float32)
        left = np.array([0], dtype=np.int32)
        right = np.array([0], dtype=np.int32)
        leaf = np.array([1], dtype=np.uint8)
        cls = np.array([leaf_cls], dtype=np.uint8)
    else:
        feat, thresh, left, right, leaf, cls = flatten(nodes)
        n_nodes = len(nodes)

    # write weights.bin
    magic = 0x44545245  # "ERTD"
    hdr = struct.pack("<4i", magic, n_nodes, X.shape[1], card["classes"])
    body = (feat.astype("<i2").tobytes() +
            thresh.astype("<f4").tobytes() +
            left.astype("<i4").tobytes() +
            right.astype("<i4").tobytes() +
            leaf.astype("<u1").tobytes() +
            cls.astype("<u1").tobytes())
    open(sys.argv[4], "wb").write(hdr + body)

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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x44545245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t n_nodes = (int32_t)hdr[1];
    int32_t n_feat = (int32_t)hdr[2];
    int32_t n_class = (int32_t)hdr[3];
    if (n_class < 1 || n_class > 256) { fprintf(stderr, "bad classes\n"); return 3; }

    size_t expected = 16
        + n_nodes * (2 + 4 + 4 + 4 + 1 + 1);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    int16_t *feat = (int16_t *)(wb + 16);
    float *thresh = (float *)(wb + 16 + n_nodes * 2);
    int32_t *left = (int32_t *)(wb + 16 + n_nodes * (2 + 4));
    int32_t *right = (int32_t *)(wb + 16 + n_nodes * (2 + 4 + 4));
    uint8_t *leaf = (uint8_t *)(wb + 16 + n_nodes * (2 + 4 + 4 + 4));
    uint8_t *cls = leaf + n_nodes;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_feat) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_feat); return 3; }
        int idx = 0;
        while (leaf[idx] == 0) {
            if (xi[feat[idx]] <= thresh[idx]) idx = left[idx];
            else idx = right[idx];
        }
        unsigned char out = cls[idx];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```