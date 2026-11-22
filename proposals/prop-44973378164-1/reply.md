```hypothesis
Family: label-dtree (single deterministic decision tree).
Why: Decision trees capture non-linear pixel interactions that linear models miss, and inference is just a few integer comparisons per sample (very low CPU cost). The UCI optdigits pixels (0..16) are ordinal and low-cardinality, ideal for axis-aligned splits.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds label-linear by >2×, the hypothesis is not supported.
```

```python
import json, struct, sys
import numpy as np

SEED = 1
MAX_DEPTHS = [5, 10, 15, 20]
MIN_SAMPLES_LEAF = [1, 5, 10]
MIN_SAMPLES_SPLIT = 10

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

def train_tree(X, y, max_depth, min_samples_leaf, min_samples_split):
    nodes = []
    def build(X_sub, y_sub, depth):
        n = len(y_sub)
        if depth >= max_depth or n < min_samples_split or len(np.unique(y_sub)) == 1:
            cls = int(np.bincount(y_sub).argmax())
            nodes.append((-1, -1, -1, -1, cls))
            return len(nodes) - 1
        best_gini_val = gini(y_sub)
        best_feature = -1
        best_threshold = -1
        best_mask = None
        for feat in range(X_sub.shape[1]):
            for thr in range(1, 17):
                mask = X_sub[:, feat] <= thr
                left_count = np.sum(mask)
                right_count = n - left_count
                if left_count < min_samples_leaf or right_count < min_samples_leaf:
                    continue
                left_y = y_sub[mask]
                right_y = y_sub[~mask]
                g = (left_count * gini(left_y) + right_count * gini(right_y)) / n
                if g < best_gini_val:
                    best_gini_val = g
                    best_feature = feat
                    best_threshold = thr
                    best_mask = mask
        if best_feature == -1:
            cls = int(np.bincount(y_sub).argmax())
            nodes.append((-1, -1, -1, -1, cls))
            return len(nodes) - 1
        left_idx = build(X_sub[best_mask], y_sub[best_mask], depth + 1)
        right_idx = build(X_sub[~best_mask], y_sub[~best_mask], depth + 1)
        nodes.append((best_feature, best_threshold, left_idx, right_idx, -1))
        return len(nodes) - 1
    root_idx = build(X, y, 0)
    return nodes, root_idx

def evaluate(nodes, root_idx, X, y):
    correct = 0
    for i in range(X.shape[0]):
        xi = X[i]
        node_idx = root_idx
        while True:
            feat, thr, left, right, cls = nodes[node_idx]
            if feat == -1:
                if cls == y[i]:
                    correct += 1
                break
            if xi[feat] <= thr:
                node_idx = left
            else:
                node_idx = right
    return correct / X.shape[0]

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    n = X.shape[0]
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]

    best_acc = -1.0
    best_params = (MAX_DEPTHS[0], MIN_SAMPLES_LEAF[0])
    for max_depth in MAX_DEPTHS:
        for min_leaf in MIN_SAMPLES_LEAF:
            nodes, root = train_tree(train_X, train_y, max_depth, min_leaf, MIN_SAMPLES_SPLIT)
            acc = evaluate(nodes, root, tune_X, tune_y)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)

    max_depth, min_leaf = best_params
    nodes, root = train_tree(X, y, max_depth, min_leaf, MIN_SAMPLES_SPLIT)

    magic = 0x44545231  # 'DTR1'
    n_nodes = len(nodes)
    out = struct.pack("<IIII", magic, d, n_nodes, root)
    for node in nodes:
        feat, thr, left, right, cls = node
        out += struct.pack("<iiiii", feat, thr, left, right, cls)
    open(sys.argv[4], "wb").write(out)

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
    int32_t threshold;
    int32_t left;
    int32_t right;
    int32_t class_;
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)w;
    uint32_t magic = hdr[0];
    int32_t d = (int32_t)hdr[1];
    int32_t n_nodes = (int32_t)hdr[2];
    int32_t root_idx = (int32_t)hdr[3];
    if (magic != 0x44545231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    if (d != 64) { fprintf(stderr, "unexpected input dimension %d\n", d); return 3; }
    long expected = 16 + (long)n_nodes * 20;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(w + 16);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int32_t node_idx = root_idx;
        while (1) {
            Node nd = nodes[node_idx];
            if (nd.feature == -1) {
                unsigned char cls = (unsigned char)nd.class_;
                put_rec(stdout, &cls, 1);
                break;
            }
            if (xi[nd.feature] <= nd.threshold) node_idx = nd.left;
            else node_idx = nd.right;
        }
    }
    return 0;
}
```