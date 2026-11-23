```hypothesis
Propose "label-dtree": a single CART decision tree (max depth 8, min samples split 5) trained on raw pixel counts.
Decision trees capture pixel interactions (e.g., strokes) that linear models miss, and inference is just a few integer compares.
If dev accuracy ≤ 0.965 (label-linear is 0.9613), the hypothesis fails.
```

```python
"""Decision tree family for label tasks with fixed-size ordinal bytes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Tunes max_depth and min_samples_split on last 20% of train."""
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning
MAX_DEPTH_GRID = [5, 6, 7, 8, 9, 10]
MIN_SAMPLES_SPLIT_GRID = [2, 5, 10, 20]

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

def best_split(X, y, indices, min_samples_split):
    """Return (feature, threshold, left_indices, right_indices, gain) or None."""
    n = len(indices)
    if n < min_samples_split:
        return None
    parent_gini = gini(y[indices])
    best_gain = 0.0
    best_feat = -1
    best_thresh = 0.0
    best_left = best_right = None

    for feat in range(X.shape[1]):
        vals = X[indices, feat]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        # thresholds are midpoints between consecutive unique values
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        for thresh in thresholds:
            left_mask = vals <= thresh
            right_mask = ~left_mask
            n_left = np.count_nonzero(left_mask)
            n_right = n - n_left
            if n_left == 0 or n_right == 0:
                continue
            left_idx = indices[left_mask]
            right_idx = indices[right_mask]
            gain = parent_gini - (n_left / n) * gini(y[left_idx]) - (n_right / n) * gini(y[right_idx])
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh
                best_left = left_idx
                best_right = right_idx
    if best_gain <= 0:
        return None
    return best_feat, best_thresh, best_left, best_right, best_gain

def build_tree(X, y, max_depth, min_samples_split):
    """Build tree iteratively (BFS). Returns list of nodes.
    Each node: (feature_idx, threshold, left_child, right_child, class_label).
    Leaf: feature_idx = -1, threshold = 0.0, left_child = right_child = -1, class_label = majority class.
    """
    n_samples = X.shape[0]
    root_indices = np.arange(n_samples)
    nodes = []  # each node is a dict during building
    queue = [(root_indices, 0)]  # (indices, depth)
    # We'll store parent index and whether left/right to link later
    # Instead, we build nodes in BFS order: when we split, we append children to nodes and record their indices.
    node_info = []  # list of dicts with keys: indices, depth, parent, is_left
    node_info.append({'indices': root_indices, 'depth': 0, 'parent': -1, 'is_left': False})
    # We'll fill nodes after we know the structure? Simpler: build nodes list as we go, but we need to know child indices.
    # Let's do: nodes list will be built in BFS order. We process queue of node indices.
    nodes = []  # final node tuples
    # Start with root node placeholder
    nodes.append(None)  # root at index 0
    queue = [0]  # indices in nodes list to process
    node_indices = {0: root_indices}  # map node index -> sample indices
    node_depth = {0: 0}

    while queue:
        idx = queue.pop(0)
        indices = node_indices[idx]
        depth = node_depth[idx]
        n = len(indices)
        # Check stopping criteria
        if depth == max_depth or n < min_samples_split or len(np.unique(y[indices])) == 1:
            # leaf
            classes, counts = np.unique(y[indices], return_counts=True)
            majority = classes[np.argmax(counts)]  # ties: smallest class because unique returns sorted
            nodes[idx] = (-1, 0.0, -1, -1, int(majority))
            continue
        split = best_split(X, y, indices, min_samples_split)
        if split is None:
            classes, counts = np.unique(y[indices], return_counts=True)
            majority = classes[np.argmax(counts)]
            nodes[idx] = (-1, 0.0, -1, -1, int(majority))
            continue
        feat, thresh, left_idx, right_idx, _ = split
        left_child = len(nodes)
        right_child = len(nodes) + 1
        nodes.append(None)
        nodes.append(None)
        node_indices[left_child] = left_idx
        node_indices[right_child] = right_idx
        node_depth[left_child] = depth + 1
        node_depth[right_child] = depth + 1
        queue.append(left_child)
        queue.append(right_child)
        nodes[idx] = (int(feat), float(thresh), left_child, right_child, -1)
    return nodes

def evaluate_tree(nodes, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        node = 0
        while True:
            feat, thresh, left, right, cls = nodes[node]
            if feat == -1:
                if cls == y[i]:
                    correct += 1
                break
            if X[i, feat] <= thresh:
                node = left
            else:
                node = right
    return correct / n

def serialize_tree(nodes, d, C):
    """Pack nodes into bytes: magic, n_nodes, d, C, then per node: i32 feat, f32 thresh, i32 left, i32 right, i32 class."""
    magic = 0x44545231  # "DTR1"
    n_nodes = len(nodes)
    header = struct.pack("<iiii", magic, n_nodes, d, C)
    body = bytearray()
    for (feat, thresh, left, right, cls) in nodes:
        body += struct.pack("<ifiii", feat, thresh, left, right, cls)
    return header + body

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # Tuning split: last 20% in file order
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    best_acc = -1.0
    best_params = None
    for max_depth in MAX_DEPTH_GRID:
        for min_samples_split in MIN_SAMPLES_SPLIT_GRID:
            nodes = build_tree(X[train_idx], y[train_idx], max_depth, min_samples_split)
            acc = evaluate_tree(nodes, X[tune_idx], y[tune_idx])
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_params = (max_depth, min_samples_split)
    # Retrain on all data
    final_nodes = build_tree(X, y, best_params[0], best_params[1])
    out = serialize_tree(final_nodes, d, C)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Decision tree inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class index. */
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
    int32_t feat;
    float thresh;
    int32_t left;
    int32_t right;
    int32_t cls;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x44545231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t n_nodes = hd[1], d = hd[2], C = hd[3];
    if (n_nodes <= 0 || d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 16 + (size_t)n_nodes * 20;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (1) {
            Node nd = nodes[node];
            if (nd.feat == -1) {
                unsigned char cls = (unsigned char)nd.cls;
                put_rec(stdout, &cls, 1);
                break;
            }
            float val = (float)xi[nd.feat];
            if (val <= nd.thresh) node = nd.left;
            else node = nd.right;
        }
    }
    return 0;
}
```