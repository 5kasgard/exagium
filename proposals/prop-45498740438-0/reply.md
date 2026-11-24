```hypothesis
Family: label-dt (single decision tree with tuned max depth).
Why: Adult data has non-linear feature interactions; a depth-tuned CART captures them with minimal inference cost (few comparisons per sample).
Failure: If dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-linear significantly.
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

def gini(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)

def build_tree(X, y, max_depth, min_leaf=1):
    n, d = X.shape
    # Node arrays
    feature = []
    threshold = []
    left = []
    right = []
    leaf_class = []
    # Queue of (node_idx, indices, depth)
    queue = [(0, np.arange(n), 0)]
    feature.append(-1)   # placeholder for root
    threshold.append(0)
    left.append(-1)
    right.append(-1)
    leaf_class.append(0)

    while queue:
        node_idx, idx, depth = queue.pop(0)
        if len(idx) < min_leaf or depth >= max_depth:
            # make leaf
            feature[node_idx] = -1
            leaf_class[node_idx] = np.bincount(y[idx], minlength=2).argmax()
            continue
        # Check if pure
        if np.all(y[idx] == y[idx[0]]):
            feature[node_idx] = -1
            leaf_class[node_idx] = y[idx[0]]
            continue

        # Find best split
        best_gain = -1.0
        best_f = -1
        best_t = 0
        parent_gini = gini(np.bincount(y[idx], minlength=2))
        for f in range(d):
            vals = X[idx, f]
            # Histogram per class
            hist = np.zeros((2, 256), dtype=np.int32)
            for c in (0, 1):
                mask = (y[idx] == c)
                if np.any(mask):
                    hist[c] = np.bincount(vals[mask], minlength=256)
            # Cumulative sums for thresholds 0..254
            cum = hist.cumsum(axis=1)
            total_left = cum[:, :-1].sum(axis=0)
            total_right = len(idx) - total_left
            # Avoid empty splits
            valid = (total_left > 0) & (total_right > 0)
            if not np.any(valid):
                continue
            # Gini for left and right
            left_gini = np.zeros(255)
            right_gini = np.zeros(255)
            for c in (0, 1):
                p_left = cum[c, :-1] / np.maximum(total_left, 1)
                p_right = (hist[c].sum() - cum[c, :-1]) / np.maximum(total_right, 1)
                left_gini += p_left * (1 - p_left)
                right_gini += p_right * (1 - p_right)
            left_gini = 1 - left_gini
            right_gini = 1 - right_gini
            weighted_gini = (total_left * left_gini + total_right * right_gini) / len(idx)
            gain = parent_gini - weighted_gini
            # Only consider valid splits
            gain[~valid] = -1
            local_best = gain.argmax()
            if gain[local_best] > best_gain:
                best_gain = gain[local_best]
                best_f = f
                best_t = local_best
        if best_gain <= 0:
            feature[node_idx] = -1
            leaf_class[node_idx] = np.bincount(y[idx], minlength=2).argmax()
            continue
        # Split
        feature[node_idx] = best_f
        threshold[node_idx] = best_t
        mask = X[idx, best_f] <= best_t
        left_idx = idx[mask]
        right_idx = idx[~mask]
        # Create children
        left_child = len(feature)
        right_child = left_child + 1
        left.append(-1); right.append(-1)
        feature.append(-1); feature.append(-1)
        threshold.append(0); threshold.append(0)
        leaf_class.append(0); leaf_class.append(0)
        left[node_idx] = left_child
        right[node_idx] = right_child
        queue.append((left_child, left_idx, depth + 1))
        queue.append((right_child, right_idx, depth + 1))

    return np.array(feature, dtype=np.int8), np.array(threshold, dtype=np.uint8), \
           np.array(left, dtype=np.int32), np.array(right, dtype=np.int32), \
           np.array(leaf_class, dtype=np.uint8)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = X.shape[0]
    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_depth = 3
    best_acc = 0.0
    for depth in range(3, 11):
        feat, thresh, lft, rgt, leaf = build_tree(X_train, y_train, depth)
        # Predict on validation
        preds = np.zeros(len(X_val), dtype=np.uint8)
        for i, x in enumerate(X_val):
            node = 0
            while feat[node] != -1:
                f = feat[node]
                t = thresh[node]
                if x[f] <= t:
                    node = lft[node]
                else:
                    node = rgt[node]
            preds[i] = leaf[node]
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # Retrain on full data
    feat, thresh, lft, rgt, leaf = build_tree(X, y, best_depth)
    num_nodes = len(feat)
    # Write weights.bin
    magic = 0x314C5444  # "DTL1" little-endian
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<IIII", magic, num_nodes, X.shape[1], 2))
        f.write(feat.tobytes())
        f.write(thresh.tobytes())
        f.write(lft.tobytes())
        f.write(rgt.tobytes())
        f.write(leaf.tobytes())

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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x314C5444) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t num_nodes = hdr[1];
    uint32_t d = hdr[2];
    uint32_t C = hdr[3];
    if (C != 2 || d != 14) { fprintf(stderr, "bad header dims\n"); return 3; }
    size_t expected = 16
        + num_nodes * (sizeof(int8_t) + sizeof(uint8_t) + sizeof(int32_t) + sizeof(int32_t) + sizeof(uint8_t));
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *p = wb + 16;
    int8_t *feature = (int8_t *)p; p += num_nodes * sizeof(int8_t);
    uint8_t *threshold = (uint8_t *)p; p += num_nodes * sizeof(uint8_t);
    int32_t *left = (int32_t *)p; p += num_nodes * sizeof(int32_t);
    int32_t *right = (int32_t *)p; p += num_nodes * sizeof(int32_t);
    uint8_t *leaf_class = (uint8_t *)p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        int node = 0;
        while (feature[node] != -1) {
            int f = feature[node];
            uint8_t t = threshold[node];
            if (xi[f] <= t) node = left[node];
            else node = right[node];
        }
        unsigned char out = leaf_class[node];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```