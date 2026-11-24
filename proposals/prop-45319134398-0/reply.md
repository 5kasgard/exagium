```hypothesis
Family: label-tree. A single deterministic CART decision tree (max depth 3) on the raw 14 byte features.
It should capture non-linear feature interactions that linear models miss, while inference requires only a few integer comparisons per item (extremely low CPU cost). If dev accuracy does not exceed label-linear's 0.8524, the hypothesis fails.
```

```python
"""label-tree: single CART decision tree (max depth 3) on raw byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: fixed seed, no randomness, tie-breaking by smallest feature then threshold.
Exports: uint32 magic 0x54524545, uint8 max_depth, uint8 n_features, uint16 pad, uint32 n_nodes,
then per node (12 bytes): uint8 feature, uint8 threshold, uint8 leaf_class, uint8 pad,
int32 left_child, int32 right_child (indices, -1 for none). Leaf nodes have left_child=-1."""
import json, struct, sys
import numpy as np

MAX_DEPTH = 3
MIN_SAMPLES_SPLIT = 20
MIN_SAMPLES_LEAF = 10
SEED = 1
MAGIC = 0x54524545

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(counts):
    # counts: [c0, c1]
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)

def build_tree(X, y, indices, depth, nodes):
    # returns node index
    n = len(indices)
    if n == 0:
        leaf_class = 0
    else:
        classes, counts = np.unique(y[indices], return_counts=True)
        leaf_class = int(classes[np.argmax(counts)])
    # stopping criteria
    if depth >= MAX_DEPTH or n < MIN_SAMPLES_SPLIT or len(classes) == 1:
        node = {
            'feature': 0, 'threshold': 0, 'leaf_class': leaf_class,
            'left': -1, 'right': -1
        }
        nodes.append(node)
        return len(nodes) - 1

    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    best_left_idx = None
    best_right_idx = None

    # For each feature, compute histograms
    for feat in range(X.shape[1]):
        vals = X[indices, feat]
        # histogram per class per value (0-255)
        hist = np.zeros((256, 2), dtype=np.int64)
        for v, c in zip(vals, y[indices]):
            hist[v, c] += 1
        cum = np.cumsum(hist, axis=0)  # cum[t] = sum_{v<=t} hist[v]
        total = cum[-1]
        # try thresholds 0..254
        for t in range(255):
            left_counts = cum[t]
            right_counts = total - left_counts
            n_left = left_counts.sum()
            n_right = right_counts.sum()
            if n_left < MIN_SAMPLES_LEAF or n_right < MIN_SAMPLES_LEAF:
                continue
            g_left = gini(left_counts)
            g_right = gini(right_counts)
            weighted_gini = (n_left * g_left + n_right * g_right) / n
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feat = feat
                best_thresh = t
                best_left_idx = (vals <= t)
                best_right_idx = ~best_left_idx

    if best_feat == -1:
        # no valid split
        node = {
            'feature': 0, 'threshold': 0, 'leaf_class': leaf_class,
            'left': -1, 'right': -1
        }
        nodes.append(node)
        return len(nodes) - 1

    # split
    left_indices = indices[best_left_idx]
    right_indices = indices[best_right_idx]
    # create internal node (placeholder, will fill children after recursion)
    node_idx = len(nodes)
    nodes.append(None)  # placeholder
    left_child = build_tree(X, y, left_indices, depth + 1, nodes)
    right_child = build_tree(X, y, right_indices, depth + 1, nodes)
    node = {
        'feature': best_feat,
        'threshold': best_thresh,
        'leaf_class': 0,  # unused for internal
        'left': left_child,
        'right': right_child
    }
    nodes[node_idx] = node
    return node_idx

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    assert d == 14
    np.random.seed(SEED)  # for any potential tie-breaking (not used)

    nodes = []
    build_tree(X, y, np.arange(n), 0, nodes)

    # flatten nodes in pre-order (already in pre-order because we appended root first, then recursively)
    # but our build_tree returns index and fills nodes list in post-order? Actually we create placeholder, then recurse, then fill.
    # The list order is: root placeholder, then left subtree nodes, then right subtree nodes, then root filled.
    # So the root is at index 0, but its children are at indices >0. The order is not pre-order.
    # We need to output in a flat array where children indices refer to positions in that array.
    # Let's do a BFS or DFS to reorder into pre-order with correct indices.
    # Simpler: we can store the tree as a list of nodes with children indices as they are in the list.
    # The current list has the property that for any internal node, its left and right children are at higher indices (since we appended them before filling the parent).
    # So the list is in post-order. We can just write the list as is, and in infer.c we traverse using the stored indices.
    # That works as long as we write the nodes in the same order (the list order) and the indices are correct.
    # The list order is the order of creation (post-order). The indices we stored (left_child, right_child) are indices into this list.
    # So we can just write the list sequentially.

    # Validate indices
    for i, node in enumerate(nodes):
        if node['left'] != -1:
            assert 0 <= node['left'] < len(nodes)
            assert 0 <= node['right'] < len(nodes)

    # Write weights.bin
    out = bytearray()
    out += struct.pack("<I", MAGIC)
    out += struct.pack("<BBH", MAX_DEPTH, d, 0)  # max_depth, n_features, pad
    out += struct.pack("<I", len(nodes))
    for node in nodes:
        out += struct.pack("<BBBBii", node['feature'], node['threshold'],
                           node['leaf_class'], 0, node['left'], node['right'])
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output: 1 byte class (0 or 1). */
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
    uint8_t feature;
    uint8_t threshold;
    uint8_t leaf_class;
    uint8_t pad;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    uint8_t max_depth = wb[4];
    uint8_t n_features = wb[5];
    // uint16_t pad = wb[6] | wb[7] << 8;
    uint32_t n_nodes = wb[8] | wb[9] << 8 | wb[10] << 16 | (uint32_t)wb[11] << 24;
    if (n_features != 14) { fprintf(stderr, "expected 14 features\n"); return 3; }
    long expected = 12 + (long)n_nodes * 12;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }
    Node *nodes = (Node *)(wb + 12);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 14) { fprintf(stderr, "item %ld is %u bytes, not 14\n", i, k); return 3; }
        int idx = 0;
        while (1) {
            Node *nd = &nodes[idx];
            if (nd->left == -1) {  // leaf
                put_rec(stdout, &nd->leaf_class, 1);
                break;
            }
            uint8_t val = xi[nd->feature];
            if (val <= nd->threshold) idx = nd->left;
            else idx = nd->right;
        }
    }
    return 0;
}
```