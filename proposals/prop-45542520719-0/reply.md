```hypothesis
The "label-tree" family uses a single decision tree with target-based ordinal encoding for categorical features. It should capture non-linear interactions better than label-linear while maintaining very low inference cost (a few branch comparisons per sample). If dev accuracy does not exceed 0.8524 (label-linear) or inference cost exceeds label-linear by more than 2x, the hypothesis is disproven.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
np.random.seed(SEED)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def compute_mappings(X, y, types):
    """Return list of mappings (uint8[256] or None) and cat_mask."""
    mappings = []
    cat_mask = 0
    for j, t in enumerate(types):
        if t == "categorical":
            cat_mask |= (1 << j)
            vals = X[:, j]
            unique_vals = np.unique(vals)
            means = {}
            for v in unique_vals:
                means[v] = y[vals == v].mean()
            sorted_vals = sorted(unique_vals, key=lambda v: means[v])
            rank = {v: i for i, v in enumerate(sorted_vals)}
            unseen_rank = len(sorted_vals)
            mapping = np.zeros(256, dtype=np.uint8)
            for v in range(256):
                mapping[v] = rank.get(v, unseen_rank)
            mappings.append(mapping)
        else:
            mappings.append(None)
    return mappings, cat_mask

def transform(X, mappings, cat_mask):
    Xt = X.copy()
    for j in range(X.shape[1]):
        if cat_mask & (1 << j):
            Xt[:, j] = mappings[j][X[:, j]]
    return Xt

class TreeBuilder:
    def __init__(self, X, y, max_depth, min_samples_leaf):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.nodes = []

    def build(self, indices, depth=0):
        n = len(indices)
        if depth >= self.max_depth or n <= self.min_samples_leaf:
            return self._make_leaf(indices)
        # purity check
        if np.all(self.y[indices] == self.y[indices[0]]):
            return self._make_leaf(indices)

        best_gini = 1.0
        best_feat = -1
        best_thresh = -1

        for feat in range(self.X.shape[1]):
            vals = self.X[indices, feat]
            labs = self.y[indices]
            c0 = np.bincount(vals[labs == 0], minlength=256)
            c1 = np.bincount(vals[labs == 1], minlength=256)
            tot0, tot1 = c0.sum(), c1.sum()
            cum0, cum1 = np.cumsum(c0), np.cumsum(c1)
            for t in range(255):
                l0, l1 = cum0[t], cum1[t]
                lt = l0 + l1
                rt = n - lt
                if lt < self.min_samples_leaf or rt < self.min_samples_leaf:
                    continue
                gini_l = 1.0 - (l0/lt)**2 - (l1/lt)**2 if lt else 0.0
                r0, r1 = tot0 - l0, tot1 - l1
                gini_r = 1.0 - (r0/rt)**2 - (r1/rt)**2 if rt else 0.0
                wg = (lt * gini_l + rt * gini_r) / n
                if wg < best_gini:
                    best_gini, best_feat, best_thresh = wg, feat, t

        if best_feat == -1:
            return self._make_leaf(indices)

        node_idx = len(self.nodes)
        self.nodes.append({'leaf': False, 'feat': best_feat, 'thresh': best_thresh,
                           'left': None, 'right': None})
        left_mask = self.X[indices, best_feat] <= best_thresh
        left_idx = self.build(indices[left_mask], depth + 1)
        right_idx = self.build(indices[~left_mask], depth + 1)
        self.nodes[node_idx]['left'] = left_idx
        self.nodes[node_idx]['right'] = right_idx
        return node_idx

    def _make_leaf(self, indices):
        labs = self.y[indices]
        cls = 1 if np.sum(labs) > len(labs) // 2 else 0
        node_idx = len(self.nodes)
        self.nodes.append({'leaf': True, 'cls': cls})
        return node_idx

def evaluate_tree(nodes, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        node = 0
        while True:
            nd = nodes[node]
            if nd['leaf']:
                if nd['cls'] == y[i]:
                    correct += 1
                break
            if X[i, nd['feat']] <= nd['thresh']:
                node = nd['left']
            else:
                node = nd['right']
    return correct / n

def serialize_tree(nodes, mappings, cat_mask, d, out_path):
    n_nodes = len(nodes)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I B I H", 0x54524545, d, n_nodes, cat_mask))
        for j in range(d):
            if cat_mask & (1 << j):
                f.write(mappings[j].tobytes())
        for nd in nodes:
            if nd['leaf']:
                f.write(struct.pack("<BBBBII", 1, 0, 0, nd['cls'], 0, 0))
            else:
                f.write(struct.pack("<BBBBII", 0, nd['feat'], nd['thresh'], 0,
                                    nd['left'], nd['right']))

def main():
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)

    n = len(X)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # hyperparameter grid
    best_acc = -1.0
    best_params = (5, 20)
    for max_depth in [3, 4, 5, 6]:
        for min_leaf in [10, 20, 30, 50]:
            maps, mask = compute_mappings(X_tr, y_tr, types)
            Xtr_t = transform(X_tr, maps, mask)
            Xtu_t = transform(X_tune, maps, mask)
            builder = TreeBuilder(Xtr_t, y_tr, max_depth, min_leaf)
            builder.build(np.arange(len(X_tr)))
            acc = evaluate_tree(builder.nodes, Xtu_t, y_tune)
            if acc > best_acc:
                best_acc, best_params = acc, (max_depth, min_leaf)

    # retrain on full data with best params
    maps, mask = compute_mappings(X, y, types)
    X_t = transform(X, maps, mask)
    builder = TreeBuilder(X_t, y, best_params[0], best_params[1])
    builder.build(np.arange(n))
    serialize_tree(builder.nodes, maps, mask, X.shape[1], sys.argv[4])

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
    uint8_t feature;
    uint8_t threshold;
    uint8_t leaf_class;
    uint32_t left;
    uint32_t right;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]);
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *p = wb;

    if (nw < 11) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t*)p; p += 4;
    if (magic != 0x54524545) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = *p++;
    uint32_t n_nodes = *(uint32_t*)p; p += 4;
    uint16_t cat_mask = *(uint16_t*)p; p += 2;

    unsigned char *mappings[14] = {0};
    for (int j = 0; j < d; j++) {
        if (cat_mask & (1 << j)) {
            mappings[j] = p;
            p += 256;
        }
    }
    Node *nodes = (Node*)p;
    size_t expected = (size_t)(p - wb) + n_nodes * sizeof(Node);
    if (expected != (size_t)nw) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;
    unsigned char feat[14];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *rec = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            feat[j] = mappings[j] ? mappings[j][rec[j]] : rec[j];
        }
        uint32_t node_idx = 0;
        while (1) {
            Node *nd = &nodes[node_idx];
            if (nd->is_leaf) {
                put_rec(stdout, &nd->leaf_class, 1);
                break;
            }
            if (feat[nd->feature] <= nd->threshold)
                node_idx = nd->left;
            else
                node_idx = nd->right;
        }
    }
    return 0;
}
```