```hypothesis
We propose a decision tree family (label-dt). Decision trees naturally handle ordinal pixel features and capture local interactions (e.g., strokes) that are crucial for digit recognition. Inference is extremely fast (only integer comparisons) and uses no floating-point arithmetic. If the model fails to beat label-linear, dev accuracy will fall below 0.96.
```
```python
import json, struct, sys
import numpy as np

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, feat_indices):
    best_gini = float('inf')
    best_feat = -1
    best_thresh = -1
    n = len(y)
    for feat in feat_indices:
        vals = np.unique(X[:, feat])
        for v in vals[:-1]:
            thresh = v
            left_mask = X[:, feat] <= thresh
            right_mask = ~left_mask
            if not np.any(left_mask) or not np.any(right_mask):
                continue
            g = (np.sum(left_mask) * gini(y[left_mask]) + np.sum(right_mask) * gini(y[right_mask])) / n
            if g < best_gini:
                best_gini = g
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh, best_gini

def build_tree(X, y, depth, max_depth):
    if depth >= max_depth or len(np.unique(y)) == 1:
        return int(np.bincount(y).argmax())
    feat, thresh, _ = best_split(X, y, range(X.shape[1]))
    if feat == -1:
        return int(np.bincount(y).argmax())
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
    return {'feat': feat, 'thresh': thresh, 'left': left, 'right': right}

def predict_tree(tree, x):
    while isinstance(tree, dict):
        if x[tree['feat']] <= tree['thresh']:
            tree = tree['left']
        else:
            tree = tree['right']
    return tree

def tree_to_arrays(tree):
    nodes = []
    def traverse(node):
        if isinstance(node, dict):
            left_idx = len(nodes) + 1
            traverse(node['left'])
            right_idx = len(nodes)
            traverse(node['right'])
            nodes.append((0, node['feat'], node['thresh'], left_idx, right_idx))
        else:
            nodes.append((1, 0, node, 0, 0))
    traverse(tree)
    return nodes

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X)
    val_size = n // 5
    train_X, train_y = X[:n - val_size], y[:n - val_size]
    val_X, val_y = X[n - val_size:], y[n - val_size:]

    best_depth = 5
    best_acc = 0.0
    for max_depth in range(3, 11):
        tree = build_tree(train_X, train_y, 0, max_depth)
        preds = np.array([predict_tree(tree, x) for x in val_X])
        acc = np.mean(preds == val_y)
        if acc > best_acc:
            best_acc = acc
            best_depth = max_depth

    tree = build_tree(X, y, 0, best_depth)
    nodes = tree_to_arrays(tree)
    num_nodes = len(nodes)

    out = struct.pack("<Iiii", 0x54445431, d, C, num_nodes)
    for node in nodes:
        out += struct.pack("<BBBii", node[0], node[1], node[2], node[3], node[4])
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x54445431) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t d = *(int32_t*)(wb + 4);
    int32_t C = *(int32_t*)(wb + 8);
    int32_t num_nodes = *(int32_t*)(wb + 12);
    if (d != 64 || C != 10 || num_nodes <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + (long)num_nodes * 11;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    unsigned char *node_type = malloc(num_nodes);
    unsigned char *node_feat = malloc(num_nodes);
    unsigned char *node_thresh = malloc(num_nodes);
    int32_t *node_left = malloc(num_nodes * sizeof(int32_t));
    int32_t *node_right = malloc(num_nodes * sizeof(int32_t));

    unsigned char *ptr = wb + 16;
    for (int i = 0; i < num_nodes; i++) {
        node_type[i] = ptr[0];
        node_feat[i] = ptr[1];
        node_thresh[i] = ptr[2];
        node_left[i] = *(int32_t*)(ptr + 3);
        node_right[i] = *(int32_t*)(ptr + 7);
        ptr += 11;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (node_type[node] == 0) {
            unsigned char feat = node_feat[node];
            unsigned char thresh = node_thresh[node];
            if (xi[feat] <= thresh) node = node_left[node];
            else node = node_right[node];
        }
        unsigned char cls = node_thresh[node];
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```