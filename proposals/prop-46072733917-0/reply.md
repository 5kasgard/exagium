```hypothesis
Family: label-bonsai (tiny CART decision tree). The Adult data has strong feature interactions (e.g., education×occupation, hours×age) that linear models miss. A depth-5 tree on the 255-bin features captures these with ~5 integer comparisons per sample, far cheaper than linear's 14×256 float adds. Hyperparameters (max_depth ∈ {3,4,5,6,7}, min_leaf) are chosen on the last 20% tuning split, then the tree is retrained on all data. If dev accuracy ≤ 0.852 or Ir/item ≥ 500, the approach fails.
```
```python
"""label-bonsai: tiny CART decision tree for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encodes: ordinal as-is (0-255), categorical as-is (0-255, 255=unseen).
Splits: ordinal uses <= threshold; categorical uses == value.
Hyperparameters (max_depth, min_leaf) tuned on last 20% of train.in (file order),
then final tree trained on all data. Exports compact tree for fast inference."""
import json, struct, sys
import numpy as np

SEED = 1
MIN_SAMPLES_SPLIT = 20
MIN_LEAF_CANDIDATES = [5, 10, 20]
DEPTH_CANDIDATES = [3, 4, 5, 6, 7]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gini(n0, n1):
    n = n0 + n1
    if n == 0: return 0.0
    p0, p1 = n0 / n, n1 / n
    return 1.0 - p0*p0 - p1*p1

def best_split(X, y, idx, types, min_leaf):
    n = len(idx)
    if n < MIN_SAMPLES_SPLIT: return None, None, None, None, -1.0
    y_idx = y[idx]
    tot0 = np.count_nonzero(y_idx == 0)
    tot1 = n - tot0
    parent_gini = gini(tot0, tot1)
    best_gain = -1.0
    best_feat = best_thresh = best_is_cat = -1
    for j, t in enumerate(types):
        vals = X[idx, j]
        if t == "ordinal":
            # histogram per class
            cnt0 = np.bincount(vals[y_idx == 0], minlength=256)
            cnt1 = np.bincount(vals[y_idx == 1], minlength=256)
            cum0 = np.cumsum(cnt0)
            cum1 = np.cumsum(cnt1)
            for th in range(255):  # split <= th
                l0, l1 = cum0[th], cum1[th]
                r0, r1 = tot0 - l0, tot1 - l1
                if l0 + l1 < min_leaf or r0 + r1 < min_leaf: continue
                gain = parent_gini - ((l0+l1)/n)*gini(l0, l1) - ((r0+r1)/n)*gini(r0, r1)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh, best_is_cat = gain, j, th, 0
        else:  # categorical: split == value
            cnt0 = np.bincount(vals[y_idx == 0], minlength=256)
            cnt1 = np.bincount(vals[y_idx == 1], minlength=256)
            for v in range(256):
                l0, l1 = cnt0[v], cnt1[v]
                r0, r1 = tot0 - l0, tot1 - l1
                if l0 + l1 < min_leaf or r0 + r1 < min_leaf: continue
                gain = parent_gini - ((l0+l1)/n)*gini(l0, l1) - ((r0+r1)/n)*gini(r0, r1)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh, best_is_cat = gain, j, v, 1
    if best_gain <= 1e-9: return None, None, None, None, -1.0
    return best_feat, best_thresh, best_is_cat, best_gain

def build_tree(X, y, idx, types, max_depth, min_leaf, depth=0):
    y_idx = y[idx]
    n0 = np.count_nonzero(y_idx == 0)
    n1 = len(idx) - n0
    prob1 = n1 / (n0 + n1) if (n0 + n1) else 0.0
    if depth == max_depth or n0 == 0 or n1 == 0:
        return {"leaf": True, "prob1": prob1}
    feat, thresh, is_cat, gain = best_split(X, y, idx, types, min_leaf)
    if feat is None:
        return {"leaf": True, "prob1": prob1}
    if is_cat == 0:
        left_mask = X[idx, feat] <= thresh
    else:
        left_mask = X[idx, feat] == thresh
    right_mask = ~left_mask
    left_idx = idx[left_mask]
    right_idx = idx[right_mask]
    if len(left_idx) < min_leaf or len(right_idx) < min_leaf:
        return {"leaf": True, "prob1": prob1}
    return {
        "leaf": False, "feat": feat, "thresh": thresh, "is_cat": is_cat,
        "left": build_tree(X, y, left_idx, types, max_depth, min_leaf, depth+1),
        "right": build_tree(X, y, right_idx, types, max_depth, min_leaf, depth+1),
        "prob1": prob1  # not used for internal, but stored for uniformity
    }

def flatten_tree(node, nodes):
    idx = len(nodes)
    if node["leaf"]:
        nodes.append((1, 0, 0, -1, -1, node["prob1"]))
    else:
        left_idx = len(nodes) + 1
        # placeholder for right, will be updated after left subtree
        nodes.append((0, node["feat"], node["thresh"], node["is_cat"], left_idx, -1, node["prob1"]))
        flatten_tree(node["left"], nodes)
        right_idx = len(nodes)
        nodes[idx] = (0, node["feat"], node["thresh"], node["is_cat"], left_idx, right_idx, node["prob1"])
        flatten_tree(node["right"], nodes)
    return idx

def eval_tree(X, y, nodes):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        xi = X[i]
        node = 0
        while True:
            is_leaf, feat, thresh, is_cat, left, right, prob1 = nodes[node]
            if is_leaf:
                pred = 1 if prob1 > 0.5 else 0
                correct += (pred == y[i])
                break
            if is_cat == 0:
                go_left = xi[feat] <= thresh
            else:
                go_left = xi[feat] == thresh
            node = left if go_left else right
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    types = card["input"]["types"]
    n = X.shape[0]
    tune_start = int(0.8 * n)
    idx_all = np.arange(n)
    idx_train = idx_all[:tune_start]
    idx_tune = idx_all[tune_start:]
    X_train, y_train = X[idx_train], y[idx_train]
    X_tune, y_tune = X[idx_tune], y[idx_tune]
    rng = np.random.default_rng(SEED)
    # deterministic order: shuffle train indices once for any internal randomness (not used here)
    # but we keep seed fixed for reproducibility
    best_acc = -1.0
    best_params = None
    best_nodes = None
    for max_depth in DEPTH_CANDIDATES:
        for min_leaf in MIN_LEAF_CANDIDATES:
            tree = build_tree(X_train, y_train, np.arange(len(X_train)), types, max_depth, min_leaf)
            nodes = []
            flatten_tree(tree, nodes)
            acc = eval_tree(X_tune, y_tune, nodes)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_nodes = nodes
    # retrain on all data with best params
    max_depth, min_leaf = best_params
    tree = build_tree(X, y, np.arange(n), types, max_depth, min_leaf)
    nodes = []
    flatten_tree(tree, nodes)
    # pack weights: magic, d, max_depth, num_nodes, then arrays
    num_nodes = len(nodes)
    is_leaf_arr = np.array([n[0] for n in nodes], dtype=np.uint8)
    feat_arr = np.array([n[1] for n in nodes], dtype=np.uint8)
    thresh_arr = np.array([n[2] for n in nodes], dtype=np.uint8)
    is_cat_arr = np.array([n[3] for n in nodes], dtype=np.uint8)
    left_arr = np.array([n[4] for n in nodes], dtype=np.int32)
    right_arr = np.array([n[5] for n in nodes], dtype=np.int32)
    prob1_arr = np.array([n[6] for n in nodes], dtype=np.float32)
    out = struct.pack("<4i", 0x424F4E53, X.shape[1], max_depth, num_nodes)
    out += is_leaf_arr.tobytes() + feat_arr.tobytes() + thresh_arr.tobytes() + is_cat_arr.tobytes()
    out += left_arr.astype("<i4").tobytes() + right_arr.astype("<i4").tobytes() + prob1_arr.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-bonsai inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8 features). Output: 1 byte class (0 or 1).
 * Weights format (little-endian):
 *   uint32 magic=0x424F4E53 ('BONS')
 *   int32 d, max_depth, num_nodes
 *   uint8 is_leaf[num_nodes]
 *   uint8 feat[num_nodes]
 *   uint8 thresh[num_nodes]
 *   uint8 is_cat[num_nodes]   (0: <=, 1: ==)
 *   int32 left[num_nodes]     (child index, -1 for leaf)
 *   int32 right[num_nodes]
 *   float32 prob1[num_nodes]  (P(class=1), used at leaves)
 */
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
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x424F4E53) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    int32_t *hd = (int32_t *)(wb + 4);
    int d = hd[0], max_depth = hd[1], num_nodes = hd[2];
    if (d <= 0 || d > 256 || num_nodes <= 0 || num_nodes > 10000) { fprintf(stderr, "bad weights: header\n"); return 3; }
    size_t off = 16;
    if (off + 4 * num_nodes > (size_t)nw) { fprintf(stderr, "bad weights: uint8 arrays\n"); return 3; }
    uint8_t *is_leaf = wb + off; off += num_nodes;
    uint8_t *feat = wb + off; off += num_nodes;
    uint8_t *thresh = wb + off; off += num_nodes;
    uint8_t *is_cat = wb + off; off += num_nodes;
    if (off + 2 * num_nodes * 4 > (size_t)nw) { fprintf(stderr, "bad weights: int32 arrays\n"); return 3; }
    int32_t *left = (int32_t *)(wb + off); off += num_nodes * 4;
    int32_t *right = (int32_t *)(wb + off); off += num_nodes * 4;
    if (off + num_nodes * 4 > (size_t)nw) { fprintf(stderr, "bad weights: float array\n"); return 3; }
    float *prob1 = (float *)(wb + off);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        int node = 0;
        while (1) {
            if (is_leaf[node]) {
                unsigned char pred = (prob1[node] > 0.5f) ? 1 : 0;
                put_rec(stdout, &pred, 1);
                break;
            }
            uint8_t f = feat[node], t = thresh[node], cat = is_cat[node];
            uint8_t xv = xi[f];
            int go_left = (cat == 0) ? (xv <= t) : (xv == t);
            node = go_left ? left[node] : right[node];
        }
    }
    return 0;
}
```