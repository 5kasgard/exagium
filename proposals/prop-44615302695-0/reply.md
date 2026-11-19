```hypothesis
A single decision tree (label-tree) on the same one-hot/z-scored features as label-linear, with depth tuned on the last 20% of train. Trees capture non-linear interactions missed by linear models, and a shallow tree adds minimal inference cost. If dev accuracy does not exceed label-linear's 0.8524, the family is not worthwhile.
```

```python
"""label-tree: single decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH_CANDIDATES = list(range(1, 11))
MIN_SAMPLES_LEAF = 1


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


def build_tree(X, y, max_depth, is_binary, rng, depth=0):
    n = len(y)
    # leaf?
    if depth == max_depth or n < MIN_SAMPLES_LEAF or gini(y) == 0.0:
        _, counts = np.unique(y, return_counts=True)
        prob = np.zeros(2, dtype=np.float32)
        prob[_] = counts / n
        return {"leaf": True, "prob": prob}

    best_gini = 1.0
    best_feat = -1
    best_thresh = 0.0

    for feat in range(X.shape[1]):
        vals = X[:, feat]
        if is_binary[feat]:
            thresholds = [0.5]
        else:
            uniq = np.unique(vals)
            if len(uniq) <= 1:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) * 0.5

        for th in thresholds:
            left_mask = vals <= th
            right_mask = ~left_mask
            nl, nr = left_mask.sum(), right_mask.sum()
            if nl == 0 or nr == 0:
                continue
            g = (nl * gini(y[left_mask]) + nr * gini(y[right_mask])) / n
            if g < best_gini:
                best_gini = g
                best_feat = feat
                best_thresh = th

    if best_feat == -1:
        _, counts = np.unique(y, return_counts=True)
        prob = np.zeros(2, dtype=np.float32)
        prob[_] = counts / n
        return {"leaf": True, "prob": prob}

    left_mask = X[:, best_feat] <= best_thresh
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], max_depth, is_binary, rng, depth + 1)
    right = build_tree(X[right_mask], y[right_mask], max_depth, is_binary, rng, depth + 1)
    return {"leaf": False, "feat": best_feat, "thresh": best_thresh, "left": left, "right": right}


def flatten_tree(node, nodes_list):
    """Pre-order flattening, returns index of this node."""
    idx = len(nodes_list)
    if node["leaf"]:
        nodes_list.append((-1, 0.0, -1, -1, node["prob"][0], node["prob"][1]))
    else:
        # placeholder, will fill children after they are created
        nodes_list.append((node["feat"], node["thresh"], -1, -1, 0.0, 0.0))
        left_idx = flatten_tree(node["left"], nodes_list)
        right_idx = flatten_tree(node["right"], nodes_list)
        nodes_list[idx] = (node["feat"], node["thresh"], left_idx, right_idx, 0.0, 0.0)
    return idx


def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_raw.shape[0]
    split = int(n * 0.8)
    idx_train = np.arange(split)
    idx_val = np.arange(split, n)

    rng = np.random.default_rng(SEED)

    # Preprocessing on training portion only
    enc = []
    is_binary = []  # per expanded feature
    feat_start = []  # start index in expanded feature vector for each original column
    for j, t in enumerate(card["input"]["types"]):
        v = X_raw[idx_train, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(("ordinal", mu, sd))
            is_binary.append(False)
            feat_start.append(len(is_binary) - 1)
        else:
            vals = np.unique(v)
            mapping = {val: i for i, val in enumerate(vals)}
            enc.append(("categorical", mapping))
            for _ in vals:
                is_binary.append(True)
            feat_start.append(len(is_binary) - len(vals))

    D = len(is_binary)

    def transform(X_part):
        Z = np.zeros((X_part.shape[0], D), dtype=np.float32)
        for j, e in enumerate(enc):
            if e[0] == "ordinal":
                mu, sd = e[1], e[2]
                col = feat_start[j]
                Z[:, col] = (X_part[:, j].astype(np.float32) - mu) / sd
            else:
                mapping = e[1]
                base = feat_start[j]
                for val, idx in mapping.items():
                    mask = X_part[:, j] == val
                    Z[mask, base + idx] = 1.0
        return Z

    Z_train = transform(X_raw[idx_train])
    Z_val = transform(X_raw[idx_val])
    y_train = y[idx_train]
    y_val = y[idx_val]

    best_depth = 1
    best_acc = 0.0
    for depth in MAX_DEPTH_CANDIDATES:
        tree = build_tree(Z_train, y_train, depth, is_binary, rng)
        nodes = []
        flatten_tree(tree, nodes)
        # evaluate on validation
        correct = 0
        for i in range(len(y_val)):
            z = Z_val[i]
            node = 0
            while nodes[node][0] != -1:
                feat, thresh, left, right, _, _ = nodes[node]
                if z[feat] <= thresh:
                    node = left
                else:
                    node = right
            _, _, _, _, p0, p1 = nodes[node]
            pred = 0 if p0 >= p1 else 1
            if pred == y_val[i]:
                correct += 1
        acc = correct / len(y_val)
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    # Retrain on all data with best_depth
    # Recompute preprocessing on all data
    enc_full = []
    is_binary_full = []
    feat_start_full = []
    for j, t in enumerate(card["input"]["types"]):
        v = X_raw[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc_full.append(("ordinal", mu, sd))
            is_binary_full.append(False)
            feat_start_full.append(len(is_binary_full) - 1)
        else:
            vals = np.unique(v)
            mapping = {val: i for i, val in enumerate(vals)}
            enc_full.append(("categorical", mapping))
            for _ in vals:
                is_binary_full.append(True)
            feat_start_full.append(len(is_binary_full) - len(vals))
    D_full = len(is_binary_full)

    def transform_full(X_part):
        Z = np.zeros((X_part.shape[0], D_full), dtype=np.float32)
        for j, e in enumerate(enc_full):
            if e[0] == "ordinal":
                mu, sd = e[1], e[2]
                col = feat_start_full[j]
                Z[:, col] = (X_part[:, j].astype(np.float32) - mu) / sd
            else:
                mapping = e[1]
                base = feat_start_full[j]
                for val, idx in mapping.items():
                    mask = X_part[:, j] == val
                    Z[mask, base + idx] = 1.0
        return Z

    Z_all = transform_full(X_raw)
    tree = build_tree(Z_all, y, best_depth, is_binary_full, rng)
    nodes = []
    flatten_tree(tree, nodes)

    # Pack weights.bin
    # Header: magic, d, D, C, max_depth, num_nodes
    magic = 0x54524545  # 'TREE'
    d = card["input"]["bytes"]
    C = card["classes"]
    num_nodes = len(nodes)
    header = struct.pack("<6i", magic, d, D_full, C, best_depth, num_nodes)

    # Column info
    col_bytes = bytearray()
    for e in enc_full:
        if e[0] == "ordinal":
            col_bytes.append(0)  # ordinal
            col_bytes.extend(struct.pack("<2f", e[1], e[2]))
        else:
            col_bytes.append(1)  # categorical
            mapping = e[1]
            # 256-byte map: feature index or 255
            mp = bytearray([255]) * 256
            for val, idx in mapping.items():
                mp[val] = idx
            col_bytes.extend(mp)

    # Nodes
    node_bytes = bytearray()
    for feat, thresh, left, right, p0, p1 in nodes:
        node_bytes.extend(struct.pack("<i f i i f f", feat, thresh, left, right, p0, p1))

    out = header + col_bytes + node_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-tree inference: ./infer weights.bin in.rec N
 * Input record: d bytes (original columns). Output: 1 byte class.
 * Weights format: see train.py. */
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
    int32_t feat;
    float thresh;
    int32_t left;
    int32_t right;
    float prob[2];
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    if (nw < 24) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3], max_depth = hd[4], num_nodes = hd[5];
    if (C != 2 || d != 14 || D <= 0 || num_nodes <= 0) { fprintf(stderr, "bad header\n"); return 3; }

    unsigned char *p = wb + 24;
    // Column preprocessing
    float *mean = malloc(d * sizeof(float));
    float *std = malloc(d * sizeof(float));
    unsigned char **cat_map = malloc(d * sizeof(unsigned char *));
    int *col_type = malloc(d * sizeof(int)); // 0 ordinal, 1 categorical
    int *feat_start = malloc(d * sizeof(int));
    int feat_idx = 0;
    for (int j = 0; j < d; j++) {
        col_type[j] = p[0];
        if (col_type[j] == 0) { // ordinal
            mean[j] = *(float *)(p + 1);
            std[j] = *(float *)(p + 5);
            feat_start[j] = feat_idx++;
            p += 9;
        } else { // categorical
            cat_map[j] = p + 1;
            feat_start[j] = feat_idx;
            // count unique categories (non-255 entries)
            int cnt = 0;
            for (int v = 0; v < 256; v++) if (cat_map[j][v] != 255) cnt++;
            feat_idx += cnt;
            p += 257;
        }
    }
    if (feat_idx != D) { fprintf(stderr, "feature count mismatch\n"); return 3; }

    // Nodes
    if (p + num_nodes * 24 != wb + nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }
    Node *nodes = (Node *)p;

    float *features = malloc(D * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(features, 0, D * sizeof(float));
        for (int j = 0; j < d; j++) {
            if (col_type[j] == 0) {
                features[feat_start[j]] = (xi[j] - mean[j]) / std[j];
            } else {
                unsigned char idx = cat_map[j][xi[j]];
                if (idx != 255) features[feat_start[j] + idx] = 1.0f;
            }
        }
        int node = 0;
        while (nodes[node].feat != -1) {
            if (features[nodes[node].feat] <= nodes[node].thresh)
                node = nodes[node].left;
            else
                node = nodes[node].right;
        }
        unsigned char best = nodes[node].prob[0] >= nodes[node].prob[1] ? 0 : 1;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```