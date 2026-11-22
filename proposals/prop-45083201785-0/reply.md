```hypothesis
Family: label-tree (single deterministic decision tree on expanded one-hot/ordinal features).
Why: Adult has strong feature interactions (e.g., education × occupation) that linear models miss; a tree captures these non-linearities with few splits. Trees are extremely fast at inference (only a few comparisons per sample) and require no matrix multiplies, so they should beat label-linear on accuracy while being cheaper per item.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear, the hypothesis is wrong.
```

```python
import json
import struct
import sys

import numpy as np

SEED = 1
MAX_DEPTHS = [5, 6, 7, 8, 9, 10]
MIN_SAMPLES_LEAVES = [1, 5, 10, 20, 50]

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
    p1 = y.mean()
    p0 = 1.0 - p1
    return 1.0 - p0 * p0 - p1 * p1

def find_best_split(X, y, indices, min_samples_leaf):
    best_gain = -1.0
    best_feature = -1
    best_threshold = 0.0
    parent_gini = gini(y[indices])
    n_features = X.shape[1]
    for f in range(n_features):
        vals = X[indices, f]
        unique_vals = np.unique(vals)
        if len(unique_vals) <= 1:
            continue
        thresholds = (unique_vals[:-1] + unique_vals[1:]) * 0.5
        for thr in thresholds:
            left_mask = vals <= thr
            right_mask = ~left_mask
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            left_gini = gini(y[indices[left_mask]])
            right_gini = gini(y[indices[right_mask]])
            n_total = n_left + n_right
            gain = parent_gini - (n_left / n_total) * left_gini - (n_right / n_total) * right_gini
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = thr
    return best_gain, best_feature, best_threshold

def build_tree(X, y, max_depth, min_samples_leaf):
    n_samples = X.shape[0]
    nodes = [None]
    stack = [(0, np.arange(n_samples), 0)]
    while stack:
        node_idx, sample_indices, depth = stack.pop()
        if depth == max_depth or len(sample_indices) < 2 * min_samples_leaf:
            prob1 = y[sample_indices].mean() if len(sample_indices) > 0 else 0.0
            nodes[node_idx] = (1, 0, 0.0, 0, 0, prob1)
            continue
        if np.all(y[sample_indices] == y[sample_indices[0]]):
            prob1 = y[sample_indices[0]]
            nodes[node_idx] = (1, 0, 0.0, 0, 0, prob1)
            continue
        gain, feat, thr = find_best_split(X, y, sample_indices, min_samples_leaf)
        if gain <= 0 or feat == -1:
            prob1 = y[sample_indices].mean()
            nodes[node_idx] = (1, 0, 0.0, 0, 0, prob1)
            continue
        left_mask = X[sample_indices, feat] <= thr
        left_indices = sample_indices[left_mask]
        right_indices = sample_indices[~left_mask]
        left_idx = len(nodes)
        right_idx = left_idx + 1
        nodes.append(None)
        nodes.append(None)
        nodes[node_idx] = (0, feat, thr, left_idx, right_idx, 0.0)
        stack.append((right_idx, right_indices, depth + 1))
        stack.append((left_idx, left_indices, depth + 1))
    return nodes

def expand_features(X, encodings):
    n = X.shape[0]
    total_expanded = sum(1 if t == 'ordinal' else K for t, K in encodings)
    X_exp = np.zeros((n, total_expanded), dtype=np.float32)
    col = 0
    for j, (t, params) in enumerate(encodings):
        if t == 'ordinal':
            mean, std = params
            X_exp[:, col] = (X[:, j].astype(np.float32) - mean) / std
            col += 1
        else:
            mapping, K = params
            indices = mapping[X[:, j]]
            rows = np.arange(n)
            valid = indices != 255
            X_exp[rows[valid], col + indices[valid]] = 1.0
            col += K
    return X_exp

def compute_encodings(X, types):
    encodings = []
    for j, t in enumerate(types):
        if t == 'ordinal':
            v = X[:, j].astype(np.float64)
            mu = v.mean()
            sd = v.std()
            if sd == 0:
                sd = 1.0
            encodings.append(('ordinal', (np.float32(mu), np.float32(sd))))
        else:
            vals = np.unique(X[:, j])
            K = len(vals)
            mapping = np.full(256, 255, dtype=np.uint8)
            mapping[vals] = np.arange(K, dtype=np.uint8)
            encodings.append(('categorical', (mapping, K)))
    return encodings

def evaluate_tree(nodes, X_exp):
    n = X_exp.shape[0]
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        x = X_exp[i]
        node_idx = 0
        while True:
            is_leaf, feat, thr, left, right, prob1 = nodes[node_idx]
            if is_leaf:
                preds[i] = 1 if prob1 >= 0.5 else 0
                break
            if x[feat] <= thr:
                node_idx = left
            else:
                node_idx = right
    return preds

def main():
    np.random.seed(SEED)
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_all.shape[0]
    split_idx = int(0.8 * n)
    X_train, X_tune = X_all[:split_idx], X_all[split_idx:]
    y_train, y_tune = y_all[:split_idx], y_all[split_idx:]

    encodings_train = compute_encodings(X_train, types)
    X_train_exp = expand_features(X_train, encodings_train)
    X_tune_exp = expand_features(X_tune, encodings_train)

    best_acc = -1.0
    best_params = None
    for max_depth in MAX_DEPTHS:
        for min_samples_leaf in MIN_SAMPLES_LEAVES:
            nodes = build_tree(X_train_exp, y_train, max_depth, min_samples_leaf)
            preds = evaluate_tree(nodes, X_tune_exp)
            acc = (preds == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)

    encodings_full = compute_encodings(X_all, types)
    X_full_exp = expand_features(X_all, encodings_full)
    max_depth, min_samples_leaf = best_params
    final_nodes = build_tree(X_full_exp, y_all, max_depth, min_samples_leaf)

    total_expanded = sum(1 if t == 'ordinal' else K for t, K in encodings_full)
    d = card["input"]["bytes"]
    magic = 0x54524545
    out = bytearray()
    out += struct.pack("<III", magic, d, total_expanded)
    for j, t in enumerate(types):
        if t == 'ordinal':
            mean, std = encodings_full[j][1]
            out += struct.pack("<Bff", 0, mean, std)
        else:
            mapping, K = encodings_full[j][1]
            out += struct.pack("<BH", 1, K)
            out += mapping.tobytes()
    num_nodes = len(final_nodes)
    out += struct.pack("<i", num_nodes)
    for node in final_nodes:
        is_leaf, feat, thr, left, right, prob1 = node
        if is_leaf:
            out += struct.pack("<Bf", 1, prob1)
        else:
            out += struct.pack("<Biffi", 0, feat, thr, left, right)
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
    unsigned char *weights = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    const unsigned char *p = weights;
    #define READ_U32 (p[0] | p[1]<<8 | p[2]<<16 | (uint32_t)p[3]<<24); p += 4
    #define READ_I32 ((int32_t)READ_U32)
    #define READ_U16 (p[0] | p[1]<<8); p += 2
    #define READ_U8 (*p++)
    #define READ_FLOAT (*(const float*)p); p += 4

    uint32_t magic = READ_U32;
    if (magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t d = READ_I32;
    int32_t total_expanded = READ_I32;

    uint8_t feat_type[16];
    float feat_mean[16], feat_std[16];
    uint16_t feat_K[16];
    uint8_t *feat_mapping[16];
    int32_t feat_offset[16];

    int32_t offset = 0;
    for (int j = 0; j < d; j++) {
        uint8_t type = READ_U8;
        feat_type[j] = type;
        if (type == 0) {
            feat_mean[j] = READ_FLOAT;
            feat_std[j] = READ_FLOAT;
            feat_offset[j] = offset;
            offset += 1;
        } else {
            uint16_t K = READ_U16;
            feat_K[j] = K;
            feat_mapping[j] = p;
            p += 256;
            feat_offset[j] = offset;
            offset += K;
        }
    }
    if (offset != total_expanded) { fprintf(stderr, "offset mismatch\n"); return 3; }

    int32_t num_nodes = READ_I32;
    uint8_t *node_is_leaf = malloc(num_nodes);
    int32_t *node_feature = malloc(num_nodes * sizeof(int32_t));
    float *node_threshold = malloc(num_nodes * sizeof(float));
    int32_t *node_left = malloc(num_nodes * sizeof(int32_t));
    int32_t *node_right = malloc(num_nodes * sizeof(int32_t));
    float *node_prob1 = malloc(num_nodes * sizeof(float));

    for (int i = 0; i < num_nodes; i++) {
        uint8_t is_leaf = READ_U8;
        node_is_leaf[i] = is_leaf;
        if (is_leaf) {
            node_prob1[i] = READ_FLOAT;
            node_feature[i] = node_left[i] = node_right[i] = 0;
            node_threshold[i] = 0.0f;
        } else {
            node_feature[i] = READ_I32;
            node_threshold[i] = READ_FLOAT;
            node_left[i] = READ_I32;
            node_right[i] = READ_I32;
            node_prob1[i] = 0.0f;
        }
    }

    if (p - weights != nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    float *expanded = malloc(total_expanded * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(expanded, 0, total_expanded * sizeof(float));

        for (int j = 0; j < d; j++) {
            uint8_t byte = xi[j];
            if (feat_type[j] == 0) {
                expanded[feat_offset[j]] = (byte - feat_mean[j]) / feat_std[j];
            } else {
                uint8_t idx = feat_mapping[j][byte];
                if (idx != 255) {
                    expanded[feat_offset[j] + idx] = 1.0f;
                }
            }
        }

        int32_t node_idx = 0;
        while (1) {
            if (node_is_leaf[node_idx]) {
                unsigned char best = (node_prob1[node_idx] >= 0.5f) ? 1 : 0;
                put_rec(stdout, &best, 1);
                break;
            }
            float val = expanded[node_feature[node_idx]];
            if (val <= node_threshold[node_idx]) {
                node_idx = node_left[node_idx];
            } else {
               