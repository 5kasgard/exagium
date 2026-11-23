```hypothesis
A single decision tree (max depth tuned on the last 20% of train) captures pixel interactions
that linear models miss, while inference is just a handful of integer comparisons per sample.
It should beat label-linear (0.9613) on dev accuracy; if it stays ≤0.96 or costs more than
~6000 Ir/item, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH_CANDIDATES = [5, 6, 7, 8, 9, 10, 11, 12]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return -np.sum(probs * np.log2(probs + 1e-12))


class DecisionTree:
    def __init__(self, max_depth, n_classes):
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.nodes = []  # each node: dict with keys

    def _best_split(self, X, y, indices):
        n_samples = len(indices)
        if n_samples <= 1:
            return None, None, None
        parent_entropy = entropy(np.bincount(y[indices], minlength=self.n_classes))
        best_gain = -1.0
        best_feat = -1
        best_thresh = -1.0
        n_features = X.shape[1]
        for feat in range(n_features):
            vals = X[indices, feat]
            unique_vals = np.unique(vals)
            if len(unique_vals) <= 1:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]
                n_left, n_right = len(left_indices), len(right_indices)
                left_entropy = entropy(np.bincount(y[left_indices], minlength=self.n_classes))
                right_entropy = entropy(np.bincount(y[right_indices], minlength=self.n_classes))
                gain = parent_entropy - (n_left / n_samples) * left_entropy - (n_right / n_samples) * right_entropy
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
        if best_gain <= 1e-12:
            return None, None, None
        return best_feat, best_thresh, best_gain

    def _build(self, X, y, indices, depth):
        node_idx = len(self.nodes)
        self.nodes.append({})  # placeholder
        counts = np.bincount(y[indices], minlength=self.n_classes)
        majority_class = int(np.argmax(counts))
        if depth == self.max_depth or len(indices) <= 1 or np.all(y[indices] == y[indices[0]]):
            self.nodes[node_idx] = {"is_leaf": True, "class": majority_class}
            return node_idx
        feat, thresh, _ = self._best_split(X, y, indices)
        if feat is None:
            self.nodes[node_idx] = {"is_leaf": True, "class": majority_class}
            return node_idx
        left_mask = X[indices, feat] <= thresh
        right_mask = ~left_mask
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]
        left_child = self._build(X, y, left_indices, depth + 1)
        right_child = self._build(X, y, right_indices, depth + 1)
        self.nodes[node_idx] = {
            "is_leaf": False,
            "feature": feat,
            "threshold": float(thresh),
            "left": left_child,
            "right": right_child,
        }
        return node_idx

    def fit(self, X, y):
        self.nodes = []
        indices = np.arange(len(y))
        self._build(X, y, indices, 0)

    def predict_one(self, x):
        node_idx = 0
        while True:
            node = self.nodes[node_idx]
            if node["is_leaf"]:
                return node["class"]
            if x[node["feature"]] <= node["threshold"]:
                node_idx = node["left"]
            else:
                node_idx = node["right"]

    def serialize(self, d):
        # header: magic, num_nodes, d
        magic = 0x44545245  # "TREE"
        num_nodes = len(self.nodes)
        header = struct.pack("<3I", magic, num_nodes, d)
        body = bytearray()
        for node in self.nodes:
            if node["is_leaf"]:
                # feature=-1, threshold=0, left=-1, right=-1, class
                body += struct.pack("<iffii", -1, 0.0, -1, -1, node["class"])
            else:
                body += struct.pack("<iffii", node["feature"], node["threshold"],
                                    node["left"], node["right"], -1)
        return header + body


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(y)
    # split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_depth = MAX_DEPTH_CANDIDATES[0]
    best_acc = -1.0
    for depth in MAX_DEPTH_CANDIDATES:
        tree = DecisionTree(depth, C)
        tree.fit(X_train, y_train)
        correct = 0
        for i in range(len(X_tune)):
            if tree.predict_one(X_tune[i]) == y_tune[i]:
                correct += 1
        acc = correct / len(X_tune)
        if acc > best_acc + 1e-12:
            best_acc = acc
            best_depth = depth

    # retrain on full data
    final_tree = DecisionTree(best_depth, C)
    final_tree.fit(X, y)
    weights = final_tree.serialize(d)
    open(sys.argv[4], "wb").write(weights)


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
    int32_t feature;
    float threshold;
    int32_t left;
    int32_t right;
    int32_t class_label;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x44545245) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t num_nodes = hdr[1];
    uint32_t d = hdr[2];
    if (nw != 12 + (long)num_nodes * 20) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 12);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != (int)d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        int node_idx = 0;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.feature == -1) {
                unsigned char out = (unsigned char)nd.class_label;
                put_rec(stdout, &out, 1);
                break;
            }
            if (xi[nd.feature] <= nd.threshold)
                node_idx = nd.left;
            else
                node_idx = nd.right;
        }
    }
    return 0;
}
```