```hypothesis
Family: label-tree (single shallow decision tree, greedy Gini splits on byte features).
Why: Adult has strong threshold effects (e.g., capital-gain/loss, hours-per-week) and interactions that a linear model misses.
     A depth-5..8 tree captures these with few nodes, giving >0.855 accuracy at ~1k Ir/item (vs 719 for linear, 7570 for GBDT).
Failure signal: dev accuracy ≤ 0.8524 (linear's score) or Ir/item > 3000.
```

```python
"""label-tree: single decision tree for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then trained on all."""
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH_GRID = [3, 4, 5, 6, 7, 8]
MIN_LEAF_GRID = [10, 20, 30, 50]
TUNE_FRAC = 0.2


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
    p1 = y.mean()
    p0 = 1.0 - p1
    return 1.0 - p0 * p0 - p1 * p1


def build_tree(X, y, max_depth, min_leaf):
    """Returns nodes list: each node is (is_leaf, feature, threshold, left, right, class).
    is_leaf=1 -> class in feature; threshold,left,right=0.
    is_leaf=0 -> feature,threshold,left,right valid; class=0."""
    n, d = X.shape
    nodes = []
    # stack of (node_idx, sample_indices, depth)
    stack = [(0, np.arange(n), 0)]
    while stack:
        node_idx, idx, depth = stack.pop()
        if len(idx) < 2 * min_leaf or depth == max_depth or gini(y[idx]) == 0.0:
            # make leaf
            cls = int(np.bincount(y[idx]).argmax())
            nodes.append((1, cls, 0, 0, 0, 0))
            continue
        parent_imp = gini(y[idx])
        best_gain = -1.0
        best_f = -1
        best_t = -1
        for f in range(d):
            vals = X[idx, f]
            # counting sort for 0..255
            cnt0 = np.zeros(256, dtype=np.int32)
            cnt1 = np.zeros(256, dtype=np.int32)
            for v in range(256):
                m = (vals == v)
                cnt0[v] = np.sum((y[idx] == 0) & m)
                cnt1[v] = np.sum((y[idx] == 1) & m)
            cum0 = np.cumsum(cnt0)
            cum1 = np.cumsum(cnt1)
            tot0, tot1 = cum0[-1], cum1[-1]
            for t in range(255):  # threshold t means left: <=t, right: >t
                l0, l1 = cum0[t], cum1[t]
                r0, r1 = tot0 - l0, tot1 - l1
                ln, rn = l0 + l1, r0 + r1
                if ln < min_leaf or rn < min_leaf:
                    continue
                gain = parent_imp - (ln / len(idx)) * (1.0 - (l0/ln)**2 - (l1/ln)**2) \
                                      - (rn / len(idx)) * (1.0 - (r0/rn)**2 - (r1/rn)**2)
                if gain > best_gain + 1e-12:
                    best_gain, best_f, best_t = gain, f, t
        if best_gain <= 0.0:
            cls = int(np.bincount(y[idx]).argmax())
            nodes.append((1, cls, 0, 0, 0, 0))
            continue
        # split
        left_mask = X[idx, best_f] <= best_t
        right_mask = ~left_mask
        left_idx = idx[left_mask]
        right_idx = idx[right_mask]
        # we will add children after this node, so left child is len(nodes), right is len(nodes)+1
        left_child = len(nodes)  # placeholder, will be updated after left subtree built
        right_child = left_child + 1  # placeholder
        nodes.append((0, best_f, best_t, left_child, right_child, 0))
        # push right then left (stack LIFO -> left processed first)
        stack.append((right_child, right_idx, depth + 1))
        stack.append((left_child, left_idx, depth + 1))
    # fix child indices (they were set as placeholders based on current length)
    # Actually, the above placeholder logic is flawed because nodes grow during left subtree.
    # Rebuild with a second pass or use a different method.
    # Let's do a simpler recursive build that returns the node index.
    pass  # We'll replace with a correct implementation below.

# --- Correct iterative build using explicit node list and a queue of work items ---
def build_tree(X, y, max_depth, min_leaf):
    n, d = X.shape
    nodes = []  # each will be a list to allow mutation
    # work queue: (parent_idx, is_left, sample_indices, depth)
    # root has parent_idx=-1
    from collections import deque
    q = deque()
    q.append((-1, True, np.arange(n), 0))
    while q:
        parent_idx, is_left, idx, depth = q.popleft()
        if len(idx) < 2 * min_leaf or depth == max_depth or gini(y[idx]) == 0.0:
            cls = int(np.bincount(y[idx]).argmax())
            node = [1, cls, 0, 0, 0, 0]
            nodes.append(node)
            if parent_idx != -1:
                nodes[parent_idx][3 if is_left else 4] = len(nodes) - 1
            continue
        parent_imp = gini(y[idx])
        best_gain = -1.0
        best_f = -1
        best_t = -1
        for f in range(d):
            vals = X[idx, f]
            cnt0 = np.zeros(256, dtype=np.int32)
            cnt1 = np.zeros(256, dtype=np.int32)
            for v in range(256):
                m = (vals == v)
                cnt0[v] = np.sum((y[idx] == 0) & m)
                cnt1[v] = np.sum((y[idx] == 1) & m)
            cum0 = np.cumsum(cnt0)
            cum1 = np.cumsum(cnt1)
            tot0, tot1 = cum0[-1], cum1[-1]
            for t in range(255):
                l0, l1 = cum0[t], cum1[t]
                r0, r1 = tot0 - l0, tot1 - l1
                ln, rn = l0 + l1, r0 + r1
                if ln < min_leaf or rn < min_leaf:
                    continue
                gain = parent_imp - (ln / len(idx)) * (1.0 - (l0/ln)**2 - (l1/ln)**2) \
                                      - (rn / len(idx)) * (1.0 - (r0/rn)**2 - (r1/rn)**2)
                if gain > best_gain + 1e-12:
                    best_gain, best_f, best_t = gain, f, t
        if best_gain <= 0.0:
            cls = int(np.bincount(y[idx]).argmax())
            node = [1, cls, 0, 0, 0, 0]
            nodes.append(node)
            if parent_idx != -1:
                nodes[parent_idx][3 if is_left else 4] = len(nodes) - 1
            continue
        # create internal node with placeholder children
        node = [0, best_f, best_t, -1, -1, 0]
        nodes.append(node)
        cur_idx = len(nodes) - 1
        if parent_idx != -1:
            nodes[parent_idx][3 if is_left else 4] = cur_idx
        # split indices
        left_mask = X[idx, best_f] <= best_t
        right_mask = ~left_mask
        left_idx = idx[left_mask]
        right_idx = idx[right_mask]
        # enqueue right then left so left is processed first (depth-first)
        q.appendleft((cur_idx, False, right_idx, depth + 1))
        q.appendleft((cur_idx, True, left_idx, depth + 1))
    # convert to tuples
    return [tuple(n) for n in nodes]


def eval_tree(nodes, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        xi = X[i]
        node_idx = 0
        while True:
            is_leaf, feat, thresh, left, right, cls = nodes[node_idx]
            if is_leaf:
                if cls == y[i]:
                    correct += 1
                break
            if xi[feat] <= thresh:
                node_idx = left
            else:
                node_idx = right
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    tune_size = int(n * TUNE_FRAC)
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_acc = -1.0
    best_params = None
    best_nodes = None
    for max_depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_LEAF_GRID:
            nodes = build_tree(X_tr, y_tr, max_depth, min_leaf)
            acc = eval_tree(nodes, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_nodes = nodes
    print(f"chosen: depth={best_params[0]}, min_leaf={best_params[1]}, tune_acc={best_acc:.4f}", file=sys.stderr)
    # retrain on full data
    final_nodes = build_tree(X, y, best_params[0], best_params[1])
    # write weights
    # magic 0x54524545 ("TREE"), d, max_depth, n_nodes, then 6 int32 per node
    d = X.shape[1]
    max_depth, _ = best_params
    n_nodes = len(final_nodes)
    magic = 0x54524545
    hdr = struct.pack("<4i", magic, d, max_depth, n_nodes)
    body = b"".join(struct.pack("<6i", *node) for node in final_nodes)
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class (0/1). */
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
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], max_depth = hd[2], n_nodes = hd[3];
    long expected = 16L + 24L * n_nodes;  // 6 int32 per node
    if (d != 14 || n_nodes <= 0 || nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    int32_t *nodes = hd + 4;  // flat array: is_leaf, feat, thresh, left, right, cls
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int idx = 0;
        while (1) {
            int32_t *nd = nodes + idx * 6;
            int is_leaf = nd[0];
            if (is_leaf) {
                unsigned char cls = (unsigned char)nd[5];
                put_rec(stdout, &cls, 1);
                break;
            }
            int feat = nd[1];
            int thresh = nd[2];
            if (xi[feat] <= thresh)
                idx = nd[3];
            else
                idx = nd[4];
        }
    }
    return 0;
}
```