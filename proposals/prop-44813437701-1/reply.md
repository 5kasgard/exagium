```hypothesis
Family: label-dtree (single deterministic CART decision tree). The 8x8 digit images have strong spatial interactions (strokes, loops) that a linear model cannot capture, but a shallow tree can isolate with axis-aligned splits on raw pixel counts (0..16). Inference needs only ~depth integer comparisons (no FP math), so it should be faster than label-linear's 64×10 multiply-adds. Hyperparameter (max_depth) is chosen on the last 20% of train.in. If dev accuracy ≤ 0.9613 or Ir/item ≥ 5058, the family does not improve on label-linear.
```
```python
"""label-dtree: single CART decision tree for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seed), single thread, stdlib + numpy only.
Splits last 20% of train for tuning max_depth, then retrains on all data.
Exports: magic 0x44545245 ('DTRE'), n_nodes, max_depth, then nodes:
  feature (u8, 255=leaf), threshold (u8), class (u8), pad (u8), left (i32), right (i32)."""
import json, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_LEAF = 10
DEPTH_CANDIDATES = [4, 5, 6, 7, 8, 9, 10, 12, 15]


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


def best_split(X, y, rng):
    n, d = X.shape
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1
    parent_gini = gini(y)
    for feat in range(d):
        vals = np.unique(X[:, feat])
        if len(vals) <= 1:
            continue
        for thresh in vals[:-1]:
            left = X[:, feat] <= thresh
            right = ~left
            if left.sum() < MIN_SAMPLES_LEAF or right.sum() < MIN_SAMPLES_LEAF:
                continue
            gain = parent_gini - (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / n
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh, best_gain


class Node:
    __slots__ = ("feat", "thresh", "cls", "left", "right")
    def __init__(self, feat=255, thresh=0, cls=0, left=-1, right=-1):
        self.feat = feat      # 0..63, 255 = leaf
        self.thresh = thresh  # 0..15
        self.cls = cls        # 0..9
        self.left = left      # index in nodes list
        self.right = right


def build_tree(X, y, max_depth, rng, depth=0):
    node = Node()
    # stopping criteria
    if depth >= max_depth or len(y) < 2 * MIN_SAMPLES_LEAF or gini(y) == 0.0:
        node.feat = 255
        node.cls = int(np.bincount(y).argmax())
        return node
    feat, thresh, gain = best_split(X, y, rng)
    if feat == -1 or gain <= 1e-12:
        node.feat = 255
        node.cls = int(np.bincount(y).argmax())
        return node
    node.feat = feat
    node.thresh = thresh
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    nodes = [node]
    left_idx = len(nodes)
    nodes.append(build_tree(X[left_mask], y[left_mask], max_depth, rng, depth + 1))
    node.left = left_idx
    right_idx = len(nodes)
    nodes.append(build_tree(X[right_mask], y[right_mask], max_depth, rng, depth + 1))
    node.right = right_idx
    return nodes[0]  # root is at index 0 after flattening? Wait, we need to flatten properly.


def flatten_tree(root):
    """Return list of nodes in pre-order, with left/right as indices into this list."""
    nodes = []
    def dfs(node):
        idx = len(nodes)
        nodes.append(node)
        if node.feat != 255:
            l = dfs(node.left)
            r = dfs(node.right)
            node.left = l
            node.right = r
        return idx
    dfs(root)
    return nodes


def train_tree(X, y, max_depth):
    rng = np.random.default_rng(SEED)
    root = build_tree(X, y, max_depth, rng)
    return flatten_tree(root)


def eval_tree(nodes, X):
    preds = np.zeros(len(X), dtype=np.int64)
    for i, x in enumerate(X):
        idx = 0
        while True:
            node = nodes[idx]
            if node.feat == 255:
                preds[i] = node.cls
                break
            if x[node.feat] <= node.thresh:
                idx = node.left
            else:
                idx = node.right
    return preds


def nodes_to_bytes(nodes):
    out = bytearray()
    for n in nodes:
        out += struct.pack("<BBBBii", n.feat, n.thresh, n.cls, 0, n.left, n.right)
    return bytes(out)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_depth = DEPTH_CANDIDATES[0]
    best_acc = -1.0
    for d in DEPTH_CANDIDATES:
        nodes = train_tree(X_tr, y_tr, d)
        pred = eval_tree(nodes, X_val)
        acc = (pred == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_depth = d

    # retrain on all data
    final_nodes = train_tree(X, y, best_depth)
    magic = 0x44545245  # 'DTRE'
    header = struct.pack("<3i", magic, len(final_nodes), best_depth)
    body = nodes_to_bytes(final_nodes)
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* label-dtree inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels 0..16). Output record: 1 byte (class 0..9).
 * Weights format: magic 0x44545245, n_nodes (i32), max_depth (i32), then n_nodes * 16 bytes:
 *   u8 feature (255=leaf), u8 threshold, u8 class, u8 pad, i32 left, i32 right. */
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

typedef struct { uint8_t feat, thresh, cls, pad; int32_t left, right; } Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x44545245) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_nodes = hd[1], max_depth = hd[2];
    if (n_nodes <= 0 || nw != 12 + (long)n_nodes * 16) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 12);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int32_t idx = 0;
        while (1) {
            Node nd = nodes[idx];
            if (nd.feat == 255) { put_rec(stdout, &nd.cls, 1); break; }
            idx = (xi[nd.feat] <= nd.thresh) ? nd.left : nd.right;
        }
    }
    return 0;
}
```