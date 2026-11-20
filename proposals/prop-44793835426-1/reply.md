```hypothesis
We propose a "label-tree" family: a single deterministic decision tree trained with Gini impurity.
It should do well on optdigits because the 8×8 pixel counts (0..16) have local structure that axis-aligned splits can capture, and a tree models non-linear boundaries without heavy computation.
Inference is just a handful of integer comparisons per sample, so it should be faster than label-linear while beating its 0.9613 accuracy.
If dev accuracy falls below 0.9613 or inference cost exceeds label-linear, the hypothesis fails.
```

```python
"""label-tree: single decision tree for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Trains on full train.in after choosing hyperparameters on the last 20% (tuning split).
Exports a binary tree: magic 'TREE', version, num_nodes, num_features, num_classes, max_depth;
then int32 node_feature[num_nodes], float32 node_threshold[num_nodes],
int32 node_left[num_nodes], int32 node_right[num_nodes], uint8 node_class[num_nodes]."""
import json, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
MAX_DEPTH_GRID = [5, 10, 15, 20, 25, 30]
MIN_SAMPLES_SPLIT_GRID = [2, 5, 10]
MIN_SAMPLES_LEAF_GRID = [1, 2, 4]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def gini(counts, total):
    if total == 0:
        return 0.0
    probs = counts / total
    return 1.0 - np.sum(probs * probs)


def best_split(X, y, indices, min_samples_leaf):
    """Return (feature, threshold, left_indices, right_indices) or (None,)*4."""
    total = len(indices)
    parent_counts = np.bincount(y[indices], minlength=10)
    best_gini = gini(parent_counts, total)
    best_feat = -1
    best_thresh = 0.0

    for feat in range(X.shape[1]):
        vals = X[indices, feat]
        # histogram of class counts per value (0..16)
        hist = np.zeros((17, 10), dtype=np.int32)
        for idx in indices:
            v = X[idx, feat]
            c = y[idx]
            hist[v, c] += 1
        present = np.where(hist.sum(axis=1) > 0)[0]
        if len(present) <= 1:
            continue
        cum = np.cumsum(hist, axis=0)
        for i in range(len(present) - 1):
            v = present[i]
            left_count = int(cum[v].sum())
            right_count = total - left_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue
            left_counts = cum[v]
            right_counts = parent_counts - left_counts
            weighted_gini = (left_count * gini(left_counts, left_count) +
                             right_count * gini(right_counts, right_count)) / total
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feat = feat
                best_thresh = (v + present[i + 1]) / 2.0

    if best_feat == -1:
        return None, None, None, None

    left_mask = X[indices, best_feat] <= best_thresh
    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]
    return best_feat, best_thresh, left_indices, right_indices


def build_tree(X, y, max_depth, min_samples_split, min_samples_leaf):
    """Build tree iteratively, return arrays for export."""
    n = len(y)
    # Pre-allocate lists
    node_feature = []
    node_threshold = []
    node_left = []
    node_right = []
    node_class = []

    # Root node
    node_feature.append(-2)  # placeholder
    node_threshold.append(0.0)
    node_left.append(-1)
    node_right.append(-1)
    node_class.append(0)

    # Stack of (node_idx, indices, depth)
    stack = [(0, np.arange(n), 0)]

    while stack:
        node_idx, indices, depth = stack.pop()
        classes, counts = np.unique(y[indices], return_counts=True)
        majority = classes[np.argmax(counts)]

        # Stopping criteria
        if depth >= max_depth or len(indices) < min_samples_split or len(classes) == 1:
            node_feature[node_idx] = -1
            node_class[node_idx] = majority
            continue

        feat, thresh, left_idx, right_idx = best_split(X, y, indices, min_samples_leaf)
        if feat is None:
            node_feature[node_idx] = -1
            node_class[node_idx] = majority
            continue

        node_feature[node_idx] = feat
        node_threshold[node_idx] = thresh

        # Create children
        left_node = len(node_feature)
        node_feature.append(-2)
        node_threshold.append(0.0)
        node_left.append(-1)
        node_right.append(-1)
        node_class.append(0)

        right_node = len(node_feature)
        node_feature.append(-2)
        node_threshold.append(0.0)
        node_left.append(-1)
        node_right.append(-1)
        node_class.append(0)

        node_left[node_idx] = left_node
        node_right[node_idx] = right_node

        stack.append((right_node, right_idx, depth + 1))
        stack.append((left_node, left_idx, depth + 1))

    return (np.array(node_feature, dtype=np.int32),
            np.array(node_threshold, dtype=np.float32),
            np.array(node_left, dtype=np.int32),
            np.array(node_right, dtype=np.int32),
            np.array(node_class, dtype=np.uint8))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = len(X)

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)

    best_acc = -1.0
    best_params = None

    for max_depth in MAX_DEPTH_GRID:
        for min_split in MIN_SAMPLES_SPLIT_GRID:
            for min_leaf in MIN_SAMPLES_LEAF_GRID:
                # Build on train_idx
                nf, nt, nl, nr, nc = build_tree(X[train_idx], y[train_idx],
                                                max_depth, min_split, min_leaf)
                # Predict on val_idx
                preds = []
                for idx in val_idx:
                    node = 0
                    while nf[node] != -1:
                        if X[idx, nf[node]] <= nt[node]:
                            node = nl[node]
                        else:
                            node = nr[node]
                    preds.append(nc[node])
                acc = np.mean(np.array(preds) == y[val_idx])
                if acc > best_acc:
                    best_acc = acc
                    best_params = (max_depth, min_split, min_leaf)

    # Retrain on full data with best params
    max_depth, min_split, min_leaf = best_params
    nf, nt, nl, nr, nc = build_tree(X, y, max_depth, min_split, min_leaf)
    num_nodes = len(nf)

    # Export weights
    out = bytearray()
    out += struct.pack("<4sIIIII", b"TREE", 1, num_nodes, X.shape[1], C, max_depth)
    out += nf.tobytes()
    out += nt.tobytes()
    out += nl.tobytes()
    out += nr.tobytes()
    out += nc.tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Each input record is 64 bytes (8x8 pixels). Output is one byte: class index.
 * Weights format: magic "TREE", version, num_nodes, num_features, num_classes, max_depth;
 * then int32 node_feature[num_nodes], float32 node_threshold[num_nodes],
 * int32 node_left[num_nodes], int32 node_right[num_nodes], uint8 node_class[num_nodes]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END); *size = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*size ? *size : 1);
    if (fread(buf, 1, *size, f) != (size_t)*size) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long N = atol(argv[3]);
    long wsize, isize = 0;
    unsigned char *wbuf = slurp(argv[1], &wsize);
    unsigned char *ibuf = N ? slurp(argv[2], &isize) : NULL;

    // Parse header
    if (wsize < 24) { fprintf(stderr, "weights too small\n"); return 3; }
    if (memcmp(wbuf, "TREE", 4) != 0) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t *hdr = (uint32_t *)(wbuf + 4);
    uint32_t version = hdr[0];
    uint32_t num_nodes = hdr[1];
    uint32_t num_features = hdr[2];
    uint32_t num_classes = hdr[3];
    uint32_t max_depth = hdr[4];
    (void)version; (void)max_depth; (void)num_classes;

    if (num_features != 64) { fprintf(stderr, "expected 64 features\n"); return 3; }

    size_t expected = 24
        + num_nodes * (sizeof(int32_t) + sizeof(float) + sizeof(int32_t) + sizeof(int32_t) + sizeof(uint8_t));
    if (wsize != (long)expected) { fprintf(stderr, "bad weights size: got %ld expected %zu\n", wsize, expected); return 3; }

    // Pointers into the weight buffer
    unsigned char *p = wbuf + 24;
    int32_t *node_feature = (int32_t *)p; p += num_nodes * sizeof(int32_t);
    float *node_threshold = (float *)p; p += num_nodes * sizeof(float);
    int32_t *node_left = (int32_t *)p; p += num_nodes * sizeof(int32_t);
    int32_t *node_right = (int32_t *)p; p += num_nodes * sizeof(int32_t);
    uint8_t *node_class = (uint8_t *)p;

    long ipos = 0;
    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *rec = next_rec(ibuf, isize, &ipos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int node = 0;
        while (node_feature[node] != -1) {
            int feat = node_feature[node];
            float thresh = node_threshold[node];
            if (rec[feat] <= thresh)
                node = node_left[node];
            else
                node = node_right[node];
        }
        unsigned char out = node_class[node];
        put_rec(stdout, &out, 1);
    }
    return 0;
}
```