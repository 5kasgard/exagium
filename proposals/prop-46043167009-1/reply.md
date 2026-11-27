```hypothesis
Propose a decision-tree family "label-dtree". The 8x8 optdigits pixels are ordinal (0..16) and axis-aligned splits
can capture the stroke geometry that a linear model misses. A single deep tree (max_depth ~10) is extremely fast at
inference (a few dozen comparisons) and should reach ~97% accuracy. If dev accuracy stays below label-linear's 96.1%,
it shows a single tree is too weak and we would need an ensemble (which is a different family).
```

```python
"""Decision tree for label tasks with fixed-size byte items (ordinal features only).
python3 -s -P train.py card.json train.in train.key weights.bin
Trains a CART tree (Gini, deterministic tie-breaking) with max_depth chosen on the last 20% of train.in,
then re-trains on all data. Exports a compact pre-order node list."""
import json, math, struct, sys

import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 5
MAX_DEPTH_CANDIDATES = list(range(1, 21))  # 1..20


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


class TreeBuilder:
    def __init__(self, X, y, max_depth, n_classes):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.nodes = []  # each node: dict with keys; we fill in pre-order

    def build(self, indices, depth):
        # create placeholder for this node (pre-order)
        node_idx = len(self.nodes)
        self.nodes.append(None)

        # stopping criteria
        if depth == self.max_depth or len(indices) < 2 * MIN_SAMPLES_LEAF:
            leaf_class = int(np.bincount(self.y[indices], minlength=self.n_classes).argmax())
            self.nodes[node_idx] = (1, leaf_class)  # is_leaf=1, class_idx
            return node_idx

        # find best split
        best_gain = -1.0
        best_feat = -1
        best_thresh = -1
        parent_gini = gini(self.y[indices])

        for feat in range(self.X.shape[1]):  # 0..63
            vals = self.X[indices, feat]
            # possible thresholds: 0..15 (since max value 16)
            for thresh in range(16):
                left_mask = vals <= thresh
                n_left = left_mask.sum()
                n_right = len(indices) - n_left
                if n_left < MIN_SAMPLES_LEAF or n_right < MIN_SAMPLES_LEAF:
                    continue
                left_indices = indices[left_mask]
                right_indices = indices[~left_mask]
                gain = parent_gini - (n_left / len(indices)) * gini(self.y[left_indices]) - \
                       (n_right / len(indices)) * gini(self.y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_gain <= 0:  # no useful split
            leaf_class = int(np.bincount(self.y[indices], minlength=self.n_classes).argmax())
            self.nodes[node_idx] = (1, leaf_class)
            return node_idx

        # split
        left_mask = self.X[indices, best_feat] <= best_thresh
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        left_idx = self.build(left_indices, depth + 1)
        right_idx = self.build(right_indices, depth + 1)

        # fill internal node: (is_leaf=0, feature, threshold, left_idx, right_idx)
        self.nodes[node_idx] = (0, best_feat, best_thresh, left_idx, right_idx)
        return node_idx


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n, d = X.shape
    C = card["classes"]

    # tuning split: last 20% in file order
    split = int(0.8 * n)
    train_idx = np.arange(split)
    tune_idx = np.arange(split, n)

    best_depth = 1
    best_acc = -1.0
    for depth in MAX_DEPTH_CANDIDATES:
        builder = TreeBuilder(X[train_idx], y[train_idx], depth, C)
        builder.build(train_idx, 0)
        nodes = builder.nodes

        # evaluate on tune split
        correct = 0
        for idx in tune_idx:
            node = 0
            while True:
                nd = nodes[node]
                if nd[0] == 1:  # leaf
                    pred = nd[1]
                    break
                feat, thresh, left, right = nd[1], nd[2], nd[3], nd[4]
                if X[idx, feat] <= thresh:
                    node = left
                else:
                    node = right
            if pred == y[idx]:
                correct += 1
        acc = correct / len(tune_idx)
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # final training on all data
    builder = TreeBuilder(X, y, best_depth, C)
    builder.build(np.arange(n), 0)
    nodes = builder.nodes

    # write weights.bin
    # magic: 0x44545245 ('DTRE' little-endian)
    # num_nodes (uint32), num_classes (uint32)
    # then each node: if leaf: uint8(1), uint8(class); else: uint8(0), uint8(feat), uint8(thresh), int32(left), int32(right)
    out = bytearray()
    out += struct.pack("<I", 0x44545245)
    out += struct.pack("<II", len(nodes), C)
    for nd in nodes:
        if nd[0] == 1:
            out += struct.pack("<BB", 1, nd[1])
        else:
            out += struct.pack("<BBBii", 0, nd[1], nd[2], nd[3], nd[4])
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Decision tree inference for label-dtree: ./infer weights.bin in.rec N
 * Input record: d bytes (d=64). Output: one byte class index. */
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

typedef struct { uint8_t is_leaf; uint8_t feat; uint8_t thresh; int32_t left; int32_t right; uint8_t class_idx; } Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x44545245) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t num_nodes = wb[4] | wb[5] << 8 | wb[6] << 16 | (uint32_t)wb[7] << 24;
    uint32_t C = wb[8] | wb[9] << 8 | wb[10] << 16 | (uint32_t)wb[11] << 24;
    if (C != 10) { fprintf(stderr, "bad class count\n"); return 3; }

    Node *nodes = malloc(num_nodes * sizeof(Node));
    size_t offset = 12;
    for (uint32_t i = 0; i < num_nodes; i++) {
        if (offset + 1 > (size_t)nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        nodes[i].is_leaf = wb[offset++];
        if (nodes[i].is_leaf) {
            if (offset + 1 > (size_t)nw) { fprintf(stderr, "truncated weights\n"); return 3; }
            nodes[i].class_idx = wb[offset++];
            nodes[i].feat = nodes[i].thresh = 0;
            nodes[i].left = nodes[i].right = -1;
        } else {
            if (offset + 1 + 1 + 4 + 4 > (size_t)nw) { fprintf(stderr, "truncated weights\n"); return 3; }
            nodes[i].feat = wb[offset++];
            nodes[i].thresh = wb[offset++];
            nodes[i].left = (int32_t)(wb[offset] | wb[offset+1]<<8 | wb[offset+2]<<16 | (uint32_t)wb[offset+3]<<24); offset += 4;
            nodes[i].right = (int32_t)(wb[offset] | wb[offset+1]<<8 | wb[offset+2]<<16 | (uint32_t)wb[offset+3]<<24); offset += 4;
            nodes[i].class_idx = 0;
        }
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        uint32_t node = 0;
        while (1) {
            Node nd = nodes[node];
            if (nd.is_leaf) {
                put_rec(stdout, &nd.class_idx, 1);
                break;
            }
            if (xi[nd.feat] <= nd.thresh) node = nd.left;
            else node = nd.right;
        }
    }
    return 0;
}
```