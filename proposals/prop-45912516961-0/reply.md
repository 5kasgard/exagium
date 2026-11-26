```hypothesis
Family: label-tree-linear (decision tree + linear model). A depth-3 CART on the raw 14 byte features captures low-order interactions; its leaf index (≤8 leaves) is added as an extra one-hot feature to the same logistic regression used by label-linear. This adds minimal inference cost (a few integer comparisons) while giving the linear model non-linear feature crosses. If dev accuracy does not exceed 0.855 (label-linear 0.8524 ± 0.0044) the extra complexity is not justified.
```

```python
"""label-tree-linear: depth-3 CART on raw bytes + logistic regression on original encoding plus leaf one-hot.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x54524C31, d, H=0, C, tree_nodes, num_leaves;
then tree_nodes * (uint8 feat, uint8 thresh, int16 left, int16 right, int16 leaf);
then d * 256 * C float32 (per-column lookup tables), num_leaves * C float32 (leaf lookup), C float32 bias."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
MAX_DEPTH = 3
MIN_LEAF = 10


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class Node:
    __slots__ = ("feat", "thresh", "left", "right", "leaf", "idx")
    def __init__(self):
        self.feat = self.thresh = 0
        self.left = self.right = -1
        self.leaf = -1
        self.idx = -1


def gini(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)


def build_tree(X, y, node, depth, rng):
    n = len(y)
    if n < MIN_LEAF or depth == MAX_DEPTH:
        node.leaf = 0  # temporary, will be renumbered later
        return
    best_gini = 1.0
    best_feat = best_thresh = -1
    for f in range(X.shape[1]):
        vals = X[:, f]
        # histogram of class counts per value
        hist = np.zeros((256, 2), dtype=np.int32)
        np.add.at(hist, (vals, y), 1)
        left_counts = np.zeros(2, dtype=np.int32)
        right_counts = hist.sum(0)
        for t in range(255):
            left_counts += hist[t]
            right_counts -= hist[t]
            if left_counts.sum() == 0 or right_counts.sum() == 0:
                continue
            g = (left_counts.sum() * gini(left_counts) + right_counts.sum() * gini(right_counts)) / n
            if g < best_gini:
                best_gini = g
                best_feat = f
                best_thresh = t
    if best_feat == -1:
        node.leaf = 0
        return
    node.feat = best_feat
    node.thresh = best_thresh
    mask = X[:, best_feat] <= best_thresh
    node.left = Node()
    node.right = Node()
    build_tree(X[mask], y[mask], node.left, depth + 1, rng)
    build_tree(X[~mask], y[~mask], node.right, depth + 1, rng)


def assign_leaves(node, leaf_counter):
    if node.leaf != -1:
        node.leaf = leaf_counter[0]
        leaf_counter[0] += 1
        return
    assign_leaves(node.left, leaf_counter)
    assign_leaves(node.right, leaf_counter)


def get_leaf(node, x):
    while node.leaf == -1:
        if x[node.feat] <= node.thresh:
            node = node.left
        else:
            node = node.right
    return node.leaf


def collect_nodes(node, nodes):
    nodes.append(node)
    if node.left != -1:
        collect_nodes(node.left, nodes)
        collect_nodes(node.right, nodes)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Build deterministic CART
    root = Node()
    build_tree(X, y, root, 0, rng)
    assign_leaves(root, [0])
    num_leaves = max(n.leaf for n in [root] if n.leaf != -1) + 1  # actually need all leaves
    # collect all nodes to get correct num_leaves
    all_nodes = []
    collect_nodes(root, all_nodes)
    leaves = [n for n in all_nodes if n.leaf != -1]
    num_leaves = len(leaves)

    # Get leaf index for each training example
    leaf_idx = np.array([get_leaf(root, x) for x in X], dtype=np.int32)

    # Encoders for original columns (same as label-linear)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    # Encoder for leaf index: one-hot
    enc.append(np.eye(num_leaves, dtype=np.float32))

    # Concatenate features
    Z_list = []
    for j, E in enumerate(enc[:-1]):
        Z_list.append(E[X[:, j]])
    Z_list.append(enc[-1][leaf_idx])
    Z = np.concatenate(Z_list, axis=1).astype(np.float32)
    n, k = Z.shape

    # Logistic regression with Adam
    W1 = np.zeros((k, C), np.float32)
    b1 = np.zeros(C, np.float32)
    params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def sg(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            g = sg(z @ W1 + b1, yy)
            grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Prepare per-column lookup tables T[j] = E_j @ W1[off:off+E_j.shape[1]]
    T = []
    off = 0
    for E in enc[:-1]:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T_leaf = (enc[-1].astype(np.float32) @ W1[off:]).astype(np.float32)

    # Serialize tree nodes in pre-order (root first)
    nodes = []
    collect_nodes(root, nodes)
    for i, n in enumerate(nodes):
        n.idx = i
    # fix left/right indices
    for n in nodes:
        if n.left != -1:
            n.left = n.left.idx
            n.right = n.right.idx

    # Write weights
    magic = 0x54524C31
    tree_nodes = len(nodes)
    header = struct.pack("<6I", magic, d, 0, C, tree_nodes, num_leaves)
    node_bytes = bytearray()
    for n in nodes:
        node_bytes += struct.pack("<BBhhh", n.feat, n.thresh, n.left, n.right, n.leaf)
    lin_bytes = b"".join(t.tobytes() for t in T) + T_leaf.tobytes() + b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + node_bytes + lin_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-tree-linear inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output: 1 byte class (0/1). */
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

typedef struct { uint8_t feat, thresh; int16_t left, right, leaf; } Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    uint32_t *hd = (uint32_t *)wb;
    if (nw < 24 || hd[0] != 0x54524C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], tree_nodes = hd[4], num_leaves = hd[5];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad weights config\n"); return 3; }
    Node *nodes = (Node *)(wb + 24);
    long expected = 24L + tree_nodes * 8L + (long)d * 256 * C * 4L + (long)num_leaves * C * 4L + (long)C * 4L;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *T = (const float *)(wb + 24 + tree_nodes * 8);
    const float *T_leaf = T + (long)d * 256 * C;
    const float *bias = T_leaf + (long)num_leaves * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* traverse tree */
        int node_idx = 0;
        while (nodes[node_idx].leaf == -1) {
            if (xi[nodes[node_idx].feat] <= nodes[node_idx].thresh)
                node_idx = nodes[node_idx].left;
            else
                node_idx = nodes[node_idx].right;
        }
        int leaf = nodes[node_idx].leaf;
        /* linear score */
        for (int c = 0; c < C; c++) sc[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) sc[c] += t[c];
        }
        const float *tl = T_leaf + (long)leaf * C;
        for (int c = 0; c < C; c++) sc[c] += tl[c];
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```