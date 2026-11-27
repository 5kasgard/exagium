```hypothesis
Family: label-dtree (single decision tree with histogram-based splits on ordinal features).
Why: Optdigits pixels have local interactions (strokes) that axis-aligned splits can capture; a depth-limited tree adds non-linearity over linear models while staying extremely fast at inference (O(depth) comparisons). Histogram splits on 0..16 values are deterministic and training is O(n * d * bins).
Failure mode: If the data is essentially linearly separable after standardization, the tree will not beat label-linear and may overfit with depth > 10.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
TUNE_FRAC = 0.2
MAX_DEPTH_CANDIDATES = [5, 10, 15]
MIN_SAMPLES_LEAF_CANDIDATES = [1, 5, 10]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return 1.0 - np.sum(probs * probs)

def find_best_split(X, y, indices, n_classes, n_bins=17):
    n = len(indices)
    if n == 0:
        return None, None, -1.0, None, None
    parent_counts = np.bincount(y[indices], minlength=n_classes)
    parent_gini = gini_from_counts(parent_counts)
    best_gain = 0.0
    best_feature = -1
    best_threshold = -1.0
    best_left = None
    best_right = None
    for feat in range(X.shape[1]):
        vals = X[indices, feat]
        # histogram per class per bin
        hist = np.zeros((n_bins, n_classes), dtype=np.int32)
        for v, c in zip(vals, y[indices]):
            hist[v, c] += 1
        # cumulative sums for left side
        left_counts = np.zeros(n_classes, dtype=np.int32)
        right_counts = parent_counts.copy()
        for split in range(n_bins - 1):  # split between split and split+1
            left_counts += hist[split]
            right_counts -= hist[split]
            left_n = left_counts.sum()
            right_n = right_counts.sum()
            if left_n == 0 or right_n == 0:
                continue
            left_gini = gini_from_counts(left_counts)
            right_gini = gini_from_counts(right_counts)
            weighted_gini = (left_n * left_gini + right_n * right_gini) / n
            gain = parent_gini - weighted_gini
            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                best_threshold = split + 0.5  # midpoint
                best_left = indices[vals <= split]
                best_right = indices[vals > split]
    return best_feature, best_threshold, best_gain, best_left, best_right

def build_tree(X, y, n_classes, max_depth, min_samples_leaf):
    n_samples = X.shape[0]
    nodes = []  # each node: (is_leaf, class, feature, threshold, left_idx, right_idx)
    # queue entries: (node_index, indices, depth)
    root_idx = 0
    nodes.append(None)  # placeholder
    queue = [(root_idx, np.arange(n_samples), 0)]
    while queue:
        node_idx, indices, depth = queue.pop(0)
        if depth >= max_depth or len(indices) <= min_samples_leaf:
            # make leaf
            counts = np.bincount(y[indices], minlength=n_classes)
            leaf_class = int(np.argmax(counts))
            nodes[node_idx] = (True, leaf_class, 0, 0.0, -1, -1)
            continue
        feat, thresh, gain, left_idx, right_idx = find_best_split(X, y, indices, n_classes)
        if feat == -1 or gain <= 0:
            counts = np.bincount(y[indices], minlength=n_classes)
            leaf_class = int(np.argmax(counts))
            nodes[node_idx] = (True, leaf_class, 0, 0.0, -1, -1)
            continue
        # internal node
        left_node_idx = len(nodes)
        right_node_idx = len(nodes) + 1
        nodes[node_idx] = (False, 0, feat, thresh, left_node_idx, right_node_idx)
        nodes.append(None); nodes.append(None)
        queue.append((left_node_idx, left_idx, depth + 1))
        queue.append((right_node_idx, right_idx, depth + 1))
    return nodes

def evaluate_tree(nodes, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        xi = X[i]
        node = 0
        while True:
            is_leaf, cls, feat, thresh, left, right = nodes[node]
            if is_leaf:
                if cls == y[i]:
                    correct += 1
                break
            if xi[feat] <= thresh:
                node = left
            else:
                node = right
    return correct / n

def write_weights(nodes, n_features, n_classes, path):
    # header: magic, n_nodes, n_features, n_classes
    magic = 0x44545245  # "DTRE"
    n_nodes = len(nodes)
    header = struct.pack("<4I", magic, n_nodes, n_features, n_classes)
    body = bytearray()
    for node in nodes:
        is_leaf, cls, feat, thresh, left, right = node
        if is_leaf:
            body.append(1)
            body.append(cls & 0xFF)
            body.extend(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')  # padding to fixed size? We'll use fixed 16 bytes per node for simplicity.
            # Actually we can use variable but easier: fixed 16 bytes: uint8 is_leaf, uint8 class, uint8 feat, uint8 pad, float32 thresh, int32 left, int32 right
        else:
            body.append(0)
            body.append(0)  # class unused
            body.append(feat & 0xFF)
            body.append(0)  # pad
            body.extend(struct.pack("<f", thresh))
            body.extend(struct.pack("<i", left))
            body.extend(struct.pack("<i", right))
    # Ensure each node is 16 bytes
    # Re-pack properly:
    body = bytearray()
    for node in nodes:
        is_leaf, cls, feat, thresh, left, right = node
        body.append(1 if is_leaf else 0)
        body.append(cls if is_leaf else 0)
        body.append(feat if not is_leaf else 0)
        body.append(0)  # padding
        body.extend(struct.pack("<f", thresh))
        body.extend(struct.pack("<i", left if not is_leaf else -1))
        body.extend(struct.pack("<i", right if not is_leaf else -1))
    with open(path, "wb") as f:
        f.write(header + body)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_classes = card["classes"]
    n_features = card["input"]["bytes"]

    # tuning split: last 20% of train.in in file order
    n = X.shape[0]
    tune_size = int(n * TUNE_FRAC)
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    best_acc = -1.0
    best_params = None
    for max_depth in MAX_DEPTH_CANDIDATES:
        for min_leaf in MIN_SAMPLES_LEAF_CANDIDATES:
            nodes = build_tree(X[train_idx], y[train_idx], n_classes, max_depth, min_leaf)
            acc = evaluate_tree(nodes, X[tune_idx], y[tune_idx])
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
    # retrain on all data with best params
    max_depth, min_leaf = best_params
    nodes = build_tree(X, y, n_classes, max_depth, min_leaf)
    write_weights(nodes, n_features, n_classes, sys.argv[4])

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
    uint8_t cls;
    uint8_t feat;
    uint8_t pad;
    float thresh;
    int32_t left;
    int32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x44545245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t n_nodes = wb[4] | wb[5]<<8 | wb[6]<<16 | wb[7]<<24;
    uint32_t n_features = wb[8] | wb[9]<<8 | wb[10]<<16 | wb[11]<<24;
    uint32_t n_classes = wb[12] | wb[13]<<8 | wb[14]<<16 | wb[15]<<24;
    if (nw != 16 + (long)n_nodes * 16) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, n_features); return 3; }
        int node_idx = 0;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.is_leaf) {
                put_rec(stdout, &nd.cls, 1);
                break;
            }
            float val = (float)xi[nd.feat];
            if (val <= nd.thresh) node_idx = nd.left;
            else node_idx = nd.right;
        }
    }
    return 0;
}
```