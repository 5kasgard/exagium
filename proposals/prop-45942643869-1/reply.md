```hypothesis
Family: label-rf (Random Forest with 10 trees, max depth 10).
Why: Optdigits has non-linear pixel interactions that linear models miss; a small forest captures these with few parameters. Each tree uses 8 random features per split (sqrt(64)) and bootstrapping for diversity. Inference is ~100 node visits per sample (10 trees × depth 10), competitive with linear's 640 adds.
Failure signal: If dev accuracy ≤ 0.961 (label-linear) or instructions/item ≥ 5000, the forest is too small or the task is truly linear.
```
```python
"""Random Forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 42), single thread. Writes weights.bin with magic 0x52460001."""
import json, struct, sys
import numpy as np

SEED = 42
N_TREES = 10
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 8  # sqrt(64)

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

def best_split(X, y, idx, features, rng):
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1
    parent_gini = gini(y[idx])
    n = len(idx)
    for f in features:
        vals = X[idx, f]
        # thresholds between 0..15 (since pixel values 0..16)
        for t in range(16):
            left_mask = vals <= t
            right_mask = ~left_mask
            if left_mask.sum() < MIN_SAMPLES_LEAF or right_mask.sum() < MIN_SAMPLES_LEAF:
                continue
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            gain = parent_gini - (n_left/n)*gini(y[idx[left_mask]]) - (n_right/n)*gini(y[idx[right_mask]])
            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thresh = t
    return best_gain, best_feat, best_thresh

def build_tree(X, y, idx, depth, rng):
    # stopping criteria
    if depth >= MAX_DEPTH or len(idx) < MIN_SAMPLES_SPLIT or len(np.unique(y[idx])) == 1:
        counts = np.bincount(y[idx], minlength=10).astype(np.int32)
        return ('leaf', counts)
    # select random features
    features = rng.choice(64, size=MAX_FEATURES, replace=False)
    gain, feat, thresh = best_split(X, y, idx, features, rng)
    if gain <= 0:
        counts = np.bincount(y[idx], minlength=10).astype(np.int32)
        return ('leaf', counts)
    left_mask = X[idx, feat] <= thresh
    right_mask = ~left_mask
    left_idx = idx[left_mask]
    right_idx = idx[right_mask]
    left_node = build_tree(X, y, left_idx, depth+1, rng)
    right_node = build_tree(X, y, right_idx, depth+1, rng)
    return ('node', feat, thresh, left_node, right_node)

def flatten_tree(node, nodes_list):
    """Convert tree to list of nodes in DFS order, return index of this node."""
    if node[0] == 'leaf':
        counts = node[1]
        nodes_list.append((1, 0, 0, 0, 0, 0, counts))
        return len(nodes_list) - 1
    else:
        _, feat, thresh, left_node, right_node = node
        left_idx = flatten_tree(left_node, nodes_list)
        right_idx = flatten_tree(right_node, nodes_list)
        nodes_list.append((0, feat, thresh, left_idx, right_idx, 0, np.zeros(10, np.int32)))
        return len(nodes_list) - 1

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    all_trees_nodes = []
    for t in range(N_TREES):
        # bootstrap sample
        boot_idx = rng.integers(0, n, size=n)
        tree_root = build_tree(X, y, boot_idx, 0, rng)
        nodes = []
        flatten_tree(tree_root, nodes)
        all_trees_nodes.append(nodes)

    # write weights.bin
    # Header: magic (uint32), n_trees (uint32)
    # Per tree: n_nodes (uint32), then nodes...
    # Node: is_leaf (uint8), feature (uint8), threshold (uint8), padding (uint8),
    #       left (int32), right (int32), class_counts[10] (int32)
    out = bytearray()
    out += struct.pack("<II", 0x52460001, N_TREES)
    for nodes in all_trees_nodes:
        out += struct.pack("<I", len(nodes))
        for nd in nodes:
            is_leaf, feat, thresh, left, right, _, counts = nd
            out += struct.pack("<BBBBii", is_leaf, feat, thresh, 0, left, right)
            out += struct.pack("<10i", *counts)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Random Forest inference: ./infer weights.bin in.rec N
 * Each input record is 64 bytes; output is one byte (class 0..9). */
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
    uint8_t feature;
    uint8_t threshold;
    uint8_t padding;
    int32_t left;
    int32_t right;
    int32_t counts[10];
} Node;

typedef struct {
    int n_nodes;
    Node *nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    const uint32_t *u32 = (const uint32_t *)wb;
    if (nw < 8 || u32[0] != 0x52460001) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int n_trees = u32[1];
    const unsigned char *ptr = wb + 8;

    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (int t = 0; t < n_trees; t++) {
        if (ptr + 4 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        int n_nodes = *(const uint32_t *)ptr; ptr += 4;
        trees[t].n_nodes = n_nodes;
        trees[t].nodes = malloc(n_nodes * sizeof(Node));
        if (ptr + (long)n_nodes * 52 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        for (int i = 0; i < n_nodes; i++) {
            Node *nd = &trees[t].nodes[i];
            nd->is_leaf = ptr[0];
            nd->feature = ptr[1];
            nd->threshold = ptr[2];
            nd->padding = ptr[3];
            nd->left = *(const int32_t *)(ptr + 4);
            nd->right = *(const int32_t *)(ptr + 8);
            memcpy(nd->counts, ptr + 12, 10 * sizeof(int32_t));
            ptr += 52;
        }
    }

    int votes[10];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        memset(votes, 0, sizeof(votes));
        for (int t = 0; t < n_trees; t++) {
            int node_idx = 0;
            while (1) {
                Node *nd = &trees[t].nodes[node_idx];
                if (nd->is_leaf) {
                    for (int c = 0; c < 10; c++) votes[c] += nd->counts[c];
                    break;
                }
                if (xi[nd->feature] <= nd->threshold)
                    node_idx = nd->left;
                else
                    node_idx = nd->right;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (votes[c] > votes[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```