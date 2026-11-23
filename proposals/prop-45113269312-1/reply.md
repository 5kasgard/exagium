```hypothesis
Random forests (label-forest) are strong on tabular ordinal data and can capture feature interactions that linear models miss. With 64 features (0..16) and 10 classes, a forest of ~100 trees of depth ≤10 should exceed label-linear's 0.9613 accuracy while keeping inference cost low (≈1000 node visits/item). If dev accuracy ≤0.965, the extra complexity is not justified.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameter grid for tuning split (last 20% of train.in)
PARAM_GRID = [
    {"n_trees": 50, "max_depth": 5},
    {"n_trees": 50, "max_depth": 10},
    {"n_trees": 50, "max_depth": 15},
    {"n_trees": 100, "max_depth": 5},
    {"n_trees": 100, "max_depth": 10},
    {"n_trees": 100, "max_depth": 15},
]
MAX_FEATURES = 8          # sqrt(64)
MIN_SAMPLES_SPLIT = 2
BASE_SEED = 12345         # deterministic seed for the whole process


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=8, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.class_ = None
        self.n_classes = 10

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.feature = []
        self.threshold = []
        self.left = []
        self.right = []
        self.class_ = []
        rng = np.random.default_rng(self.random_state)
        self._build(0, X, y, 0, rng)
        self.feature = np.array(self.feature, dtype=np.uint8)
        self.threshold = np.array(self.threshold, dtype=np.float32)
        self.left = np.array(self.left, dtype=np.int32)
        self.right = np.array(self.right, dtype=np.int32)
        self.class_ = np.array(self.class_, dtype=np.uint8)

    def _gini_impurity(self, y_left, y_right):
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        if n_total == 0:
            return 0.0
        def gini(y):
            if len(y) == 0:
                return 0.0
            counts = np.bincount(y, minlength=self.n_classes)
            probs = counts / len(y)
            return 1.0 - np.sum(probs ** 2)
        return (n_left * gini(y_left) + n_right * gini(y_right)) / n_total

    def _build(self, node_idx, X, y, depth, rng):
        n_samples = X.shape[0]
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            counts = np.bincount(y, minlength=self.n_classes)
            cls = np.argmax(counts)
            self.feature.append(255)
            self.threshold.append(0.0)
            self.left.append(-1)
            self.right.append(-1)
            self.class_.append(cls)
            return

        feat_indices = rng.choice(self.n_features, self.max_features, replace=False)
        best_gini = 1.0
        best_feat = -1
        best_thresh = -1.0
        for feat in feat_indices:
            values = X[:, feat]
            unique_vals = np.unique(values)
            if len(unique_vals) == 1:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gini = self._gini_impurity(y[left_mask], y[right_mask])
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat
                    best_thresh = thresh

        if best_feat == -1:
            counts = np.bincount(y, minlength=self.n_classes)
            cls = np.argmax(counts)
            self.feature.append(255)
            self.threshold.append(0.0)
            self.left.append(-1)
            self.right.append(-1)
            self.class_.append(cls)
            return

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        cur_idx = len(self.feature)
        self.feature.append(best_feat)
        self.threshold.append(best_thresh)
        self.left.append(0)
        self.right.append(0)
        self.class_.append(0)

        left_child_idx = len(self.feature)
        self.left[cur_idx] = left_child_idx
        self._build(left_child_idx, X[left_mask], y[left_mask], depth + 1, rng)

        right_child_idx = len(self.feature)
        self.right[cur_idx] = right_child_idx
        self._build(right_child_idx, X[right_mask], y[right_mask], depth + 1, rng)

    def predict(self, X):
        n = X.shape[0]
        preds = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            node = 0
            while True:
                feat = self.feature[node]
                if feat == 255:
                    preds[i] = self.class_[node]
                    break
                thresh = self.threshold[node]
                if X[i, feat] <= thresh:
                    node = self.left[node]
                else:
                    node = self.right[node]
        return preds


class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, max_features=8, seed=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.seed)
        for i in range(self.n_trees):
            idx = rng.choice(n_samples, n_samples, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            tree_seed = int(rng.integers(0, 2**32))
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_features=self.max_features,
                                random_state=tree_seed)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        n = X.shape[0]
        votes = np.zeros((n, 10), dtype=np.int32)
        for tree in self.trees:
            preds = tree.predict(X)
            for i, p in enumerate(preds):
                votes[i, p] += 1
        return np.argmax(votes, axis=1)

    def write_weights(self, path, num_features, num_classes):
        with open(path, "wb") as f:
            f.write(struct.pack("<IIIII", 0x464F5245, 1, self.n_trees, num_classes, num_features))
            for tree in self.trees:
                num_nodes = len(tree.feature)
                f.write(struct.pack("<I", num_nodes))
                f.write(tree.feature.astype("<u1").tobytes())
                f.write(tree.threshold.astype("<f4").tobytes())
                f.write(tree.left.astype("<i4").tobytes())
                f.write(tree.right.astype("<i4").tobytes())
                f.write(tree.class_.astype("<u1").tobytes())


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc = -1.0
    best_params = None
    for params in PARAM_GRID:
        rf = RandomForest(n_trees=params["n_trees"],
                          max_depth=params["max_depth"],
                          min_samples_split=MIN_SAMPLES_SPLIT,
                          max_features=MAX_FEATURES,
                          seed=BASE_SEED)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_val)
        acc = np.mean(preds == y_val)
        if acc > best_acc:
            best_acc = acc
            best_params = params

    # Retrain on all data with best parameters
    rf_final = RandomForest(n_trees=best_params["n_trees"],
                            max_depth=best_params["max_depth"],
                            min_samples_split=MIN_SAMPLES_SPLIT,
                            max_features=MAX_FEATURES,
                            seed=BASE_SEED)
    rf_final.fit(X, y)
    rf_final.write_weights(sys.argv[4], card["input"]["bytes"], card["classes"])


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
    uint32_t num_nodes;
    uint8_t *feature;
    float *threshold;
    int32_t *left;
    int32_t *right;
    uint8_t *class_;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x464F5245) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t version = wb[4] | wb[5]<<8 | wb[6]<<16 | wb[7]<<24;
    if (version != 1) { fprintf(stderr, "bad version\n"); return 3; }
    uint32_t num_trees = wb[8] | wb[9]<<8 | wb[10]<<16 | wb[11]<<24;
    uint32_t num_classes = wb[12] | wb[13]<<8 | wb[14]<<16 | wb[15]<<24;
    uint32_t num_features = wb[16] | wb[17]<<8 | wb[18]<<16 | wb[19]<<24;
    if (num_classes < 1 || num_classes > 256 || num_features < 1) {
        fprintf(stderr, "bad header values\n"); return 3;
    }

    Tree *trees = calloc(num_trees, sizeof(Tree));
    unsigned char *ptr = wb + 20;
    unsigned char *end = wb + nw;
    for (uint32_t t = 0; t < num_trees; t++) {
        if (ptr + 4 > end) { fprintf(stderr, "truncated tree header\n"); return 3; }
        trees[t].num_nodes = ptr[0] | ptr[1]<<8 | ptr[2]<<16 | ptr[3]<<24;
        ptr += 4;
        uint32_t nn = trees[t].num_nodes;
        size_t need = nn * (1 + 4 + 4 + 4 + 1);
        if (ptr + need > end) { fprintf(stderr, "truncated tree data\n"); return 3; }
        trees[t].feature = ptr; ptr += nn;
        trees[t].threshold = (float *)ptr; ptr += nn * 4;
        trees[t].left = (int32_t *)ptr; ptr += nn * 4;
        trees[t].right = (int32_t *)ptr; ptr += nn * 4;
        trees[t].class_ = ptr; ptr += nn;
    }

    int *votes = calloc(num_classes, sizeof(int));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != num_features) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, num_features); return 3; }
        memset(votes, 0, num_classes * sizeof(int));
        for (uint32_t t = 0; t < num_trees; t++) {
            uint32_t node = 0;
            while (1) {
                uint8_t feat = trees[t].feature[node];
                if (feat == 255) {
                    votes[trees[t].class_[node]]++;
                    break;
                }
                float thresh = trees[t].threshold[node];
                if (xi[feat] <= thresh)
                    node = trees[t].left[node];
                else
                    node = trees[t].right[node];
            }
        }
        unsigned char best = 0;
        for (uint32_t c = 1; c < num_classes; c++)
            if (votes[c] > votes[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```