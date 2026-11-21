```hypothesis
Family: label-tree-linear. A depth-3 decision tree (7 splits) on the 14 binned features feeds its leaf index (0–7) as a 15th categorical feature into a logistic regression. The tree captures low-order interactions that linear models miss; the linear layer calibrates and adds the remaining additive signal. Training is deterministic (fixed tie-breaking, seed=1). If dev accuracy does not exceed label-linear (0.8524) by at least 0.005, the hybrid adds complexity without sufficient gain.
```
```python
"""label-tree-linear: depth-3 decision tree + logistic regression.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
MAX_DEPTH = 3
MIN_LEAF = 100


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
    p = np.mean(y)
    return 2.0 * p * (1.0 - p)


def best_split(X, y, feat_idx):
    """Return (threshold, gain) for best binary split on one feature."""
    best_gain, best_thr = -1.0, 0
    parent = gini(y)
    n = len(y)
    for thr in range(255):  # split: <= thr vs > thr
        mask = X[:, feat_idx] <= thr
        nL, nR = mask.sum(), (~mask).sum()
        if nL < MIN_LEAF or nR < MIN_LEAF:
            continue
        gain = parent - (nL * gini(y[mask]) + nR * gini(y[~mask])) / n
        if gain > best_gain + 1e-12:
            best_gain, best_thr = gain, thr
    return best_thr, best_gain


def build_tree(X, y, depth=0):
    """Recursively build a deterministic binary tree. Returns node dict or leaf index."""
    if depth == MAX_DEPTH or len(y) < 2 * MIN_LEAF or gini(y) == 0.0:
        return {"leaf": True, "index": 0}  # index filled later

    best_feat, best_thr, best_gain = -1, 0, -1.0
    for f in range(X.shape[1]):
        thr, gain = best_split(X, y, f)
        if gain > best_gain + 1e-12:
            best_gain, best_feat, best_thr = gain, f, thr

    if best_gain <= 0.0:
        return {"leaf": True, "index": 0}

    mask = X[:, best_feat] <= best_thr
    left = build_tree(X[mask], y[mask], depth + 1)
    right = build_tree(X[~mask], y[~mask], depth + 1)
    return {"leaf": False, "feat": best_feat, "thr": best_thr, "left": left, "right": right}


def assign_leaves(node, next_idx=0):
    """Assign consecutive leaf indices (0..L-1) in deterministic order (left first)."""
    if node["leaf"]:
        node["index"] = next_idx
        return next_idx + 1
    next_idx = assign_leaves(node["left"], next_idx)
    next_idx = assign_leaves(node["right"], next_idx)
    return next_idx


def tree_to_arrays(node, nodes, leaf_count):
    """Flatten tree to arrays for C inference. Returns (nodes_list, leaf_count)."""
    if node["leaf"]:
        return
    nodes.append((node["feat"], node["thr"], None, None))  # left/right filled later
    idx = len(nodes) - 1
    left_idx = len(nodes)
    tree_to_arrays(node["left"], nodes, leaf_count)
    right_idx = len(nodes)
    tree_to_arrays(node["right"], nodes, leaf_count)
    nodes[idx] = (node["feat"], node["thr"], left_idx, right_idx)


def predict_leaf(tree_nodes, x):
    """Traverse flattened tree (list of (feat, thr, left, right)) for one sample x (14 bytes)."""
    idx = 0
    while True:
        feat, thr, left, right = tree_nodes[idx]
        if left is None:  # leaf
            return idx  # we'll map node index to leaf index later; simpler: store leaf index in node
        if x[feat] <= thr:
            idx = left
        else:
            idx = right


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # --- Build deterministic tree ---
    rng = np.random.default_rng(SEED)
    # Shuffle once for tie-breaking stability in split search? Not needed; we iterate features in order.
    root = build_tree(X, y)
    n_leaves = assign_leaves(root)
    # Flatten tree: we need a mapping from node index to leaf index for leaves.
    # Let's create a flat array where each node is either a split or a leaf.
    # We'll store: for split nodes: feat (u8), thr (u8), left (u16), right (u16)
    # for leaf nodes: feat=255, thr=leaf_index, left=0, right=0
    nodes = []
    def flatten(node):
        if node["leaf"]:
            nodes.append((255, node["index"], 0, 0))
        else:
            idx = len(nodes)
            nodes.append((node["feat"], node["thr"], 0, 0))  # placeholder
            left_idx = len(nodes)
            flatten(node["left"])
            right_idx = len(nodes)
            flatten(node["right"])
            nodes[idx] = (node["feat"], node["thr"], left_idx, right_idx)
    flatten(root)
    n_nodes = len(nodes)

    # --- Augment data with leaf index ---
    leaf_idx = np.zeros(len(X), dtype=np.uint8)
    for i, x in enumerate(X):
        idx = 0
        while True:
            feat, thr, left, right = nodes[idx]
            if feat == 255:
                leaf_idx[i] = thr
                break
            if x[feat] <= thr:
                idx = left
            else:
                idx = right

    X_aug = np.column_stack((X, leaf_idx))  # 15 features
    d_aug = d + 1

    # --- Encode features (same as label-linear) ---
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X_aug[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X_aug[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    # The new leaf feature is categorical with up to 8 values (0..n_leaves-1)
    vals = np.arange(n_leaves)
    E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
    enc.append(E)

    Z = np.concatenate([E[X_aug[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    # --- Logistic regression (Adam) ---
    W1 = np.zeros((k, C), np.float32); b1 = np.zeros(C, np.float32)
    params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            S = z @ W1 + b1
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # --- Pack weights ---
    # Format: magic(0x54524532), d_aug, H=0, C, n_nodes, tree_nodes, T, b1
    # tree_nodes: each 6 bytes (u8 feat, u8 thr, u16 left, u16 right)
    # T: d_aug * 256 * C float32, b1: C float32
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    magic = 0x54524532  # "TRE2"
    tree_bytes = b""
    for feat, thr, left, right in nodes:
        tree_bytes += struct.pack("<BBHH", feat, thr, left, right)

    out = struct.pack("<5i", magic, d_aug, 0, C, n_nodes)
    out += tree_bytes
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-tree-linear inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output: 1 byte class (0 or 1).
 * Weights format: magic(0x54524532), d_aug, H=0, C, n_nodes,
 *                 tree_nodes[n_nodes] (u8 feat, u8 thr, u16 left, u16 right),
 *                 T[d_aug][256][C] float32, b1[C] float32. */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }

    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x54524532) { fprintf(stderr, "bad magic\n"); return 3; }
    int d_aug = hd[1], H = hd[2], C = hd[3], n_nodes = hd[4];
    if (H != 0 || C != 2 || d_aug != 15) { fprintf(stderr, "unexpected header\n"); return 3; }

    // tree nodes start at byte 20
    unsigned char *tree_ptr = wb + 20;
    if (nw < 20 + 6L * n_nodes) { fprintf(stderr, "weights too small for tree\n"); return 3; }

    // T and b1 after tree
    const float *T = (const float *)(tree_ptr + 6L * n_nodes);
    const float *b1 = T + (long)d_aug * 256 * C;
    long expected = 20 + 6L * n_nodes + 4L * ((long)d_aug * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    unsigned char leaf_idx;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }

        // traverse tree to get leaf index
        int node = 0;
        while (1) {
            unsigned char feat = tree_ptr[6 * node];
            unsigned char thr  = tree_ptr[6 * node + 1];
            uint16_t left  = tree_ptr[6 * node + 2] | (tree_ptr[6 * node + 3] << 8);
            uint16_t right = tree_ptr[6 * node + 4] | (tree_ptr[6 * node + 5] << 8);
            if (feat == 255) { leaf_idx = thr; break; }
            if (xi[feat] <= thr) node = left;
            else node = right;
        }

        // linear model: features 0..13 are xi[0..13], feature 14 is leaf_idx
        acc[0] = b1[0]; acc[1] = b1[1];
        for (int j = 0; j < 14; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            acc[0] += t[0]; acc[1] += t[1];
        }
        const float *t = T + ((long)14 * 256 + leaf_idx) * C;
        acc[0] += t[0]; acc[1] += t[1];

        unsigned char best = (acc[1] > acc[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```