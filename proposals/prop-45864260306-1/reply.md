```hypothesis
Family: label-dtree (single CART decision tree). Decision trees naturally handle ordinal pixel intensities and capture non-linear feature interactions that linear models miss. With max_depth tuned on the 20% validation split, a shallow tree (depth 5–10) should exceed label-linear's 96.13% accuracy while keeping inference extremely fast (only a few integer comparisons per sample). If dev accuracy ≤ 0.9613 or inference cost ≥ label-mlp16 (8685 Ir/item), the family is not worthwhile.
```

```python
"""label-dtree: CART decision tree for fixed-size byte label tasks.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters (max_depth, min_samples_leaf) chosen on last 20% of train.in (file order),
then final model trained on all train.in."""
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


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        # tree arrays
        self.feature_idx = []   # int16
        self.threshold = []     # float32
        self.left = []          # int32
        self.right = []         # int32
        self.leaf_class = []    # uint8 (0 for internal nodes)

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p * p)

    def _best_split(self, X, y, idx):
        """Return (feature, threshold, left_idx, right_idx) or None if no split."""
        n = len(idx)
        if n < 2 * self.min_samples_leaf:
            return None
        parent_gini = self._gini(y[idx])
        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0
        best_left = None
        best_right = None

        for f in range(X.shape[1]):
            vals = X[idx, f]
            uniq = np.unique(vals)
            if len(uniq) == 1:
                continue
            # thresholds at midpoints
            threshes = (uniq[:-1] + uniq[1:]) / 2.0
            for th in threshes:
                left_mask = vals <= th
                left_idx = idx[left_mask]
                right_idx = idx[~left_mask]
                if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                    continue
                gini_left = self._gini(y[left_idx])
                gini_right = self._gini(y[right_idx])
                weighted_gini = (len(left_idx) * gini_left + len(right_idx) * gini_right) / n
                gain = parent_gini - weighted_gini
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feat = f
                    best_thresh = th
                    best_left = left_idx
                    best_right = right_idx
                elif abs(gain - best_gain) < 1e-12:
                    # deterministic tie-break: smaller feature index, then smaller threshold
                    if f < best_feat or (f == best_feat and th < best_thresh):
                        best_feat = f
                        best_thresh = th
                        best_left = left_idx
                        best_right = right_idx
        if best_gain <= 0.0:
            return None
        return best_feat, best_thresh, best_left, best_right

    def _build(self, X, y, idx, depth):
        node_idx = len(self.feature_idx)
        # placeholder
        self.feature_idx.append(0)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.leaf_class.append(0)

        # stopping criteria
        if depth >= self.max_depth:
            # make leaf
            cls = np.argmax(np.bincount(y[idx]))
            self.feature_idx[node_idx] = -1
            self.leaf_class[node_idx] = cls
            return node_idx

        split = self._best_split(X, y, idx)
        if split is None:
            cls = np.argmax(np.bincount(y[idx]))
            self.feature_idx[node_idx] = -1
            self.leaf_class[node_idx] = cls
            return node_idx

        feat, thresh, left_idx, right_idx = split
        self.feature_idx[node_idx] = feat
        self.threshold[node_idx] = thresh
        left_node = self._build(X, y, left_idx, depth + 1)
        right_node = self._build(X, y, right_idx, depth + 1)
        self.left[node_idx] = left_node
        self.right[node_idx] = right_node
        return node_idx

    def fit(self, X, y):
        idx = np.arange(len(y))
        self._build(X, y, idx, 0)

    def to_bytes(self, n_features, n_classes):
        n_nodes = len(self.feature_idx)
        # header: magic, n_nodes, n_features, n_classes
        magic = 0x44545245  # 'DTRE' little-endian
        header = struct.pack("<4i", magic, n_nodes, n_features, n_classes)
        # arrays
        feat_arr = np.array(self.feature_idx, dtype=np.int16)
        thresh_arr = np.array(self.threshold, dtype=np.float32)
        left_arr = np.array(self.left, dtype=np.int32)
        right_arr = np.array(self.right, dtype=np.int32)
        leaf_arr = np.array(self.leaf_class, dtype=np.uint8)
        return header + feat_arr.tobytes() + thresh_arr.tobytes() + left_arr.tobytes() + right_arr.tobytes() + leaf_arr.tobytes()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n, d = X.shape
    C = card["classes"]

    # split: last 20% for tuning
    split_idx = int(n * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    # hyperparameter grid
    best_acc = -1.0
    best_params = None
    for max_depth in [5, 6, 7, 8, 9, 10]:
        for min_samples_leaf in [1, 2, 5, 10]:
            tree = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            tree.fit(X_train, y_train)
            # predict on val
            preds = []
            for xi in X_val:
                node = 0
                while tree.feature_idx[node] != -1:
                    f = tree.feature_idx[node]
                    if xi[f] <= tree.threshold[node]:
                        node = tree.left[node]
                    else:
                        node = tree.right[node]
                preds.append(tree.leaf_class[node])
            acc = np.mean(np.array(preds) == y_val)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)

    # retrain on full data
    tree = DecisionTree(max_depth=best_params[0], min_samples_leaf=best_params[1])
    tree.fit(X, y)

    out_bytes = tree.to_bytes(d, C)
    open(sys.argv[4], "wb").write(out_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-dtree inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (uint8 pixels). Output record: 1 byte class (0-9). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x44545245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int n_nodes = hd[1];
    int n_features = hd[2];
    int n_classes = hd[3];
    if (n_features != 64 || n_classes != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + n_nodes * (2L + 4L + 4L + 4L + 1L);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    int16_t *feat = (int16_t *)(wb + 16);
    float *thresh = (float *)(wb + 16 + n_nodes * 2);
    int32_t *left = (int32_t *)(wb + 16 + n_nodes * 2 + n_nodes * 4);
    int32_t *right = (int32_t *)(wb + 16 + n_nodes * 2 + n_nodes * 4 + n_nodes * 4);
    uint8_t *leaf_class = (uint8_t *)(wb + 16 + n_nodes * 2 + n_nodes * 4 + n_nodes * 4 + n_nodes * 4);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_features); return 3; }
        int node = 0;
        while (feat[node] != -1) {
            int f = feat[node];
            if (xi[f] <= thresh[node])
                node = left[node];
            else
                node = right[node];
        }
        unsigned char out = leaf_class[node];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```