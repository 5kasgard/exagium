```hypothesis
Family: label-rf (random forest). Optdigits is a classic tabular vision task where tree ensembles excel; a shallow random forest captures non-linear pixel interactions that linear models miss, while inference remains fast (only comparisons, no multiplications). We expect dev accuracy >0.9613 (label-linear) with lower or comparable Ir/item. Failure: dev accuracy ≤0.9613 or Ir/item ≥5058 (label-linear cost).
```

```python
"""Random forest for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning (last 20% of train.in)
GRID = [
    (5, 5), (5, 10), (5, 15), (5, 20),
    (10, 5), (10, 10), (10, 15), (10, 20),
    (20, 5), (20, 10), (20, 15), (20, 20),
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(y):
    counts = np.bincount(y, minlength=10)
    probs = counts / len(y)
    return 1.0 - np.sum(probs * probs)

def find_best_split(X, y, feat_indices, rng):
    best_gini = 1.0
    best_feat = None
    best_thresh = None
    best_left = None
    best_right = None
    for f in feat_indices:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        # try midpoints between consecutive unique values
        for i in range(len(uniq) - 1):
            thresh = (uniq[i] + uniq[i + 1]) * 0.5
            left = vals <= thresh
            right = ~left
            if left.sum() == 0 or right.sum() == 0:
                continue
            gini_left = gini(y[left])
            gini_right = gini(y[right])
            weighted = (left.sum() * gini_left + right.sum() * gini_right) / len(y)
            if weighted < best_gini:
                best_gini = weighted
                best_feat = f
                best_thresh = thresh
                best_left = left
                best_right = right
    return best_feat, best_thresh, best_left, best_right

def build_tree(X, y, depth, max_depth, max_features, rng):
    if depth == max_depth or len(np.unique(y)) == 1 or len(y) < 2:
        cls = int(np.bincount(y).argmax())
        return {'leaf': True, 'class': cls}
    n_features = X.shape[1]
    feat_indices = rng.choice(n_features, max_features, replace=False)
    best_feat, best_thresh, best_left, best_right = find_best_split(X, y, feat_indices, rng)
    if best_feat is None:
        cls = int(np.bincount(y).argmax())
        return {'leaf': True, 'class': cls}
    left_tree = build_tree(X[best_left], y[best_left], depth + 1, max_depth, max_features, rng)
    right_tree = build_tree(X[best_right], y[best_right], depth + 1, max_depth, max_features, rng)
    return {'leaf': False, 'feature': int(best_feat), 'threshold': float(best_thresh),
            'left': left_tree, 'right': right_tree}

def train_forest(X, y, n_trees, max_depth, rng):
    trees = []
    n = X.shape[0]
    max_features = int(math.sqrt(X.shape[1]))
    for i in range(n_trees):
        tree_rng = np.random.default_rng(rng.integers(2**32))
        idx = tree_rng.choice(n, n, replace=True)
        tree = build_tree(X[idx], y[idx], 0, max_depth, max_features, tree_rng)
        trees.append(tree)
    return trees

def flatten_tree(tree):
    nodes = []
    def build(node):
        idx = len(nodes)
        if node['leaf']:
            nodes.append({'leaf': True, 'class': node['class'],
                          'feature': -1, 'threshold': 0.0, 'left': -1, 'right': -1})
        else:
            nodes.append({'leaf': False, 'feature': node['feature'],
                          'threshold': node['threshold'], 'left': -1, 'right': -1, 'class': 0})
            left_idx = build(node['left'])
            right_idx = build(node['right'])
            nodes[idx]['left'] = left_idx
            nodes[idx]['right'] = right_idx
        return idx
    build(tree)
    return nodes

def predict_tree(tree, x):
    node = tree
    while not node['leaf']:
        if x[node['feature']] <= node['threshold']:
            node = node['left']
        else:
            node = node['right']
    return node['class']

def predict_forest(trees, X):
    preds = []
    for x in X:
        votes = np.zeros(10, dtype=int)
        for tree in trees:
            votes[predict_tree(tree, x)] += 1
        preds.append(int(np.argmax(votes)))
    return np.array(preds)

def main():
    card = json.load(open(sys.argv[1]))
    train_recs = records(sys.argv[2])
    key_recs = records(sys.argv[3])
    n = len(train_recs)
    tune_size = n // 5
    train_size = n - tune_size

    # Split into train and tune (last 20% for tuning)
    X_all = np.frombuffer(b"".join(train_recs), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in key_recs], dtype=np.int64)
    X_train, X_tune = X_all[:train_size], X_all[train_size:]
    y_train, y_tune = y_all[:train_size], y_all[train_size:]

    # Normalization on training portion
    mean = X_train.mean(axis=0).astype(np.float64)
    std = X_train.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0
    X_train_n = (X_train - mean) / std
    X_tune_n = (X_tune - mean) / std

    # Grid search on tuning split
    best_acc = -1.0
    best_params = None
    grid_rng = np.random.default_rng(12345)  # fixed seed for grid search
    for n_trees, max_depth in GRID:
        # deterministic seed per config
        seed = (n_trees << 16) ^ max_depth ^ 0xABCD
        forest = train_forest(X_train_n, y_train, n_trees, max_depth,
                              np.random.default_rng(seed))
        preds = predict_forest(forest, X_tune_n)
        acc = (preds == y_tune).mean()
        if acc > best_acc or (acc == best_acc and (n_trees < best_params[0] or
                                                    (n_trees == best_params[0] and max_depth < best_params[1]))):
            best_acc = acc
            best_params = (n_trees, max_depth)

    n_trees, max_depth = best_params
    # Retrain on full data with final seed
    mean_full = X_all.mean(axis=0).astype(np.float64)
    std_full = X_all.std(axis=0).astype(np.float64)
    std_full[std_full == 0] = 1.0
    X_full_n = (X_all - mean_full) / std_full
    final_forest = train_forest(X_full_n, y_all, n_trees, max_depth,
                                np.random.default_rng(SEED))

    # Serialize
    magic = 0x52465354  # "RFST"
    n_features = card["input"]["bytes"]
    n_classes = card["classes"]
    out = bytearray()
    out += struct.pack("<I", magic)
    out += struct.pack("<I", n_trees)
    out += struct.pack("<I", max_depth)
    out += struct.pack("<I", n_features)
    out += struct.pack("<I", n_classes)
    out += np.array(mean_full, dtype=np.float32).tobytes()
    out += np.array(std_full, dtype=np.float32).tobytes()
    for tree in final_forest:
        nodes = flatten_tree(tree)
        out += struct.pack("<I", len(nodes))
        for node in nodes:
            feat = node['feature']
            thresh = node['threshold']
            left = node['left']
            right = node['right']
            cls = node['class']
            out += struct.pack("<i", feat)
            out += struct.pack("<f", thresh)
            out += struct.pack("<i", left)
            out += struct.pack("<i", right)
            out += struct.pack("<B", cls)
            out += b'\x00\x00\x00'  # padding to 20 bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Random forest inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the majority-vote class. */
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

typedef struct __attribute__((packed)) {
    int32_t feature;      // -1 for leaf
    float threshold;
    int32_t left;
    int32_t right;
    uint8_t class_label;
    uint8_t padding[3];
} Node;

typedef struct {
    uint32_t num_nodes;
    Node *nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    uint32_t *hd = (uint32_t *)wbuf;
    if (nw < 20 || hd[0] != 0x52465354) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t n_trees = hd[1];
    uint32_t max_depth = hd[2];      // unused but validated
    uint32_t n_features = hd[3];
    uint32_t n_classes = hd[4];
    if (n_features != 64 || n_classes != 10) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    float *mean = (float *)(hd + 5);
    float *std = mean + n_features;
    unsigned char *ptr = (unsigned char *)(std + n_features);
    Tree *trees = malloc(n_trees * sizeof(Tree));
    for (uint32_t t = 0; t < n_trees; t++) {
        if (ptr + 4 > wbuf + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t num_nodes = *(uint32_t *)ptr; ptr += 4;
        if (ptr + (long)num_nodes * 20 > wbuf + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        trees[t].num_nodes = num_nodes;
        trees[t].nodes = (Node *)ptr;
        ptr += num_nodes * 20;
    }
    if (ptr != wbuf + nw) { fprintf(stderr, "extra data in weights\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float features[64];
    int votes[10];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, n_features); return 3; }
        for (int j = 0; j < 64; j++) {
            float val = xi[j];
            float m = mean[j];
            float s = std[j];
            features[j] = (val - m) / (s == 0.0f ? 1.0f : s);
        }
        memset(votes, 0, sizeof(votes));
        for (uint32_t t = 0; t < n_trees; t++) {
            uint32_t idx = 0;
            while (1) {
                Node *node = &trees[t].nodes[idx];
                if (node->feature == -1) {
                    votes[node->class_label]++;
                    break;
                }
                if (features[node->feature] <= node->threshold)
                    idx = node->left;
                else
                    idx = node->right;
            }
        }
        int best = 0;
        for (int c = 1; c < 10; c++)
            if (votes[c] > votes[best]) best = c;
        unsigned char out = (unsigned char)best;
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```