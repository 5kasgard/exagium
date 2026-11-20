```hypothesis
Family: label-rf (Random Forest). Optdigits pixels have local spatial correlations that linear models miss; an ensemble of shallow decision trees captures non-linear interactions while keeping inference fast (simple threshold comparisons). Expect dev accuracy > 0.9613 (label-linear) at comparable or lower Ir/item. Failure if accuracy ≤ 0.9613 or Ir/item ≥ 5058.
```
```python
import json, struct, sys
import numpy as np

N_TREES = 10
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 2
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p * p)

def best_split(X, y, features, rng):
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1
    parent_gini = gini(y)
    n_samples = len(y)
    for feat in features:
        vals = X[:, feat]
        for t in range(16):
            left_mask = vals <= t
            right_mask = ~left_mask
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            gain = parent_gini - (n_left/n_samples)*gini(y[left_mask]) - (n_right/n_samples)*gini(y[right_mask])
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = t
    return best_feat, best_thresh, best_gain

def build_tree(X, y, depth, rng, n_features):
    if depth >= MAX_DEPTH or len(y) < MIN_SAMPLES_SPLIT or len(np.unique(y)) == 1:
        cls = int(np.bincount(y).argmax())
        return {'leaf': True, 'class': cls}
    mtry = max(1, int(np.sqrt(n_features)))
    features = rng.choice(n_features, mtry, replace=False)
    feat, thresh, gain = best_split(X, y, features, rng)
    if feat == -1 or gain <= 0:
        cls = int(np.bincount(y).argmax())
        return {'leaf': True, 'class': cls}
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    left_tree = build_tree(X[left_mask], y[left_mask], depth+1, rng, n_features)
    right_tree = build_tree(X[right_mask], y[right_mask], depth+1, rng, n_features)
    return {'leaf': False, 'feature': int(feat), 'threshold': int(thresh), 'left': left_tree, 'right': right_tree}

def flatten_tree(tree, nodes_list):
    if tree['leaf']:
        idx = len(nodes_list)
        nodes_list.append({'feature': 0, 'threshold': 0, 'left': -1, 'right': -1, 'class': tree['class']})
        return idx
    else:
        left_idx = flatten_tree(tree['left'], nodes_list)
        right_idx = flatten_tree(tree['right'], nodes_list)
        idx = len(nodes_list)
        nodes_list.append({'feature': tree['feature'], 'threshold': tree['threshold'], 'left': left_idx, 'right': right_idx, 'class': 255})
        return idx

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_samples, n_features = X.shape
    n_classes = card["classes"]
    rng = np.random.default_rng(SEED)
    all_nodes = []
    tree_roots = []
    tree_node_counts = []
    for t in range(N_TREES):
        idx = rng.choice(n_samples, n_samples, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        tree = build_tree(X_boot, y_boot, 0, rng, n_features)
        start_idx = len(all_nodes)
        root_idx = flatten_tree(tree, all_nodes)
        tree_roots.append(root_idx)
        tree_node_counts.append(len(all_nodes) - start_idx)
    total_nodes = len(all_nodes)
    features_arr = np.array([n['feature'] for n in all_nodes], dtype=np.uint8)
    thresholds_arr = np.array([n['threshold'] for n in all_nodes], dtype=np.uint8)
    lefts_arr = np.array([n['left'] for n in all_nodes], dtype=np.int32)
    rights_arr = np.array([n['right'] for n in all_nodes], dtype=np.int32)
    classes_arr = np.array([n['class'] for n in all_nodes], dtype=np.uint8)
    roots_arr = np.array(tree_roots, dtype=np.int32)
    counts_arr = np.array(tree_node_counts, dtype=np.int32)
    magic = 0x52463100
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<IIIII", magic, N_TREES, n_features, n_classes, total_nodes))
        f.write(roots_arr.astype("<i4").tobytes())
        f.write(counts_arr.astype("<i4").tobytes())
        f.write(features_arr.astype("<u1").tobytes())
        f.write(thresholds_arr.astype("<u1").tobytes())
        f.write(lefts_arr.astype("<i4").tobytes())
        f.write(rights_arr.astype("<i4").tobytes())
        f.write(classes_arr.astype("<u1").tobytes())

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
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x52463100) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    int32_t n_trees = *(int32_t*)(wb+4);
    int32_t n_features = *(int32_t*)(wb+8);
    int32_t n_classes = *(int32_t*)(wb+12);
    int32_t total_nodes = *(int32_t*)(wb+16);
    if (n_trees <= 0 || n_features != 64 || n_classes != 10 || total_nodes <= 0) {
        fprintf(stderr, "bad weights: params\n"); return 3; }
    long expected = 20 + 8L * n_trees + 11L * total_nodes;
    if (nw != expected) { fprintf(stderr, "bad weights: size mismatch %ld != %ld\n", nw, expected); return 3; }
    int32_t *roots = (int32_t*)(wb + 20);
    int32_t *counts = (int32_t*)(wb + 20 + 4L * n_trees);
    uint8_t *features = wb + 20 + 8L * n_trees;
    uint8_t *thresholds = features + total_nodes;
    int32_t *lefts = (int32_t*)(thresholds + total_nodes);
    int32_t *rights = lefts + total_nodes;
    uint8_t *classes = (uint8_t*)(rights + total_nodes);
    for (int t = 0; t < n_trees; t++) {
        if (roots[t] < 0 || roots[t] >= total_nodes) { fprintf(stderr, "bad root index\n"); return 3; }
        if (counts[t] <= 0) { fprintf(stderr, "bad count\n"); return 3; }
    }
    for (int i = 0; i < total_nodes; i++) {
        if (features[i] >= n_features) { fprintf(stderr, "bad feature index\n"); return 3; }
        if (classes[i] != 255 && classes[i] >= n_classes) { fprintf(stderr, "bad class\n"); return 3; }
        if (lefts[i] != -1 && (lefts[i] < 0 || lefts[i] >= total_nodes)) { fprintf(stderr, "bad left index\n"); return 3; }
        if (rights[i] != -1 && (rights[i] < 0 || rights[i] >= total_nodes)) { fprintf(stderr, "bad right index\n"); return 3; }
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char *votes = malloc(n_classes);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_features); return 3; }
        memset(votes, 0, n_classes);
        for (int t = 0; t < n_trees; t++) {
            int32_t node = roots[t];
            while (1) {
                uint8_t cls = classes[node];
                if (cls != 255) { votes[cls]++; break; }
                uint8_t feat = features[node];
                uint8_t thresh = thresholds[node];
                node = (xi[feat] <= thresh) ? lefts[node] : rights[node];
                if (node < 0 || node >= total_nodes) { fprintf(stderr, "traversal error\n"); return 3; }
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < n_classes; c++) if (votes[c] > votes[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```