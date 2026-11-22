```hypothesis
Family: label-cart (single CART decision tree on encoded features).  
Why: The Adult dataset has 14 mixed-type features; a shallow tree captures non-linear interactions that linear models miss, while staying far cheaper than GBDT/MLP at inference. Depth 5 with min-leaf 20 gives ~30 nodes, so inference is a few comparisons per record.  
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear, the family does not meet its goal.
```

```python
"""label-cart: single CART decision tree for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib + numpy only.
Encodes ordinal as z-scored, categorical as one-hot (like label-linear).
Chooses max_depth and min_samples_leaf on last 20% of train.in (tuning split),
then retrains on all train.in. Exports tree + encoding parameters to weights.bin."""
import json, math, struct, sys

import numpy as np

SEED = 1
MAX_DEPTH_GRID = [3, 4, 5, 6]
MIN_LEAF_GRID = [10, 20, 50, 100]
TUNING_FRAC = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types, rng):
    """Return encoded matrix Z (n x d), and encoding info for inference."""
    n, n_orig = X.shape
    enc_info = []  # list of dict per original feature
    Z_parts = []
    for j, t in enumerate(types):
        v = X[:, j]
        if t == "ordinal":
            v_f = v.astype(np.float64)
            mu, sd = v_f.mean(), v_f.std()
            if sd == 0:
                sd = 1.0
            z = (v_f - mu) / sd
            Z_parts.append(z.reshape(-1, 1))
            enc_info.append({"type": "ordinal", "mean": float(mu), "std": float(sd)})
        else:
            vals = np.unique(v)
            cat_to_idx = {val: idx for idx, val in enumerate(vals)}
            one_hot = np.zeros((n, len(vals)), dtype=np.float32)
            for idx, val in enumerate(vals):
                one_hot[v == val, idx] = 1.0
            Z_parts.append(one_hot)
            enc_info.append({"type": "categorical", "categories": vals.astype(np.uint8)})
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    return Z, enc_info


def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    return 2 * p * (1 - p)


def best_split(Z, y, feature_indices, min_leaf):
    n, d = Z.shape
    best_gini = float('inf')
    best_feat = -1
    best_thresh = 0.0
    for feat in feature_indices:
        col = Z[:, feat]
        unique_vals = np.unique(col)
        if len(unique_vals) <= 1:
            continue
        # For binary features, only one split point (0.5)
        if len(unique_vals) == 2 and set(unique_vals) <= {0.0, 1.0}:
            thresholds = [0.5]
        else:
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        for thresh in thresholds:
            left = col <= thresh
            right = ~left
            if left.sum() < min_leaf or right.sum() < min_leaf:
                continue
            g = (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / n
            if g < best_gini:
                best_gini = g
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh, best_gini


def build_tree(Z, y, max_depth, min_leaf, depth=0, feature_indices=None):
    """Returns list of nodes in pre-order. Each node is a dict."""
    if feature_indices is None:
        feature_indices = list(range(Z.shape[1]))
    nodes = []
    def recurse(node_idx):
        # Check stopping conditions
        if depth >= max_depth or len(y) < 2 * min_leaf or gini(y) == 0.0:
            # leaf
            cls = 1 if np.mean(y) >= 0.5 else 0
            nodes.append({"leaf": True, "class": int(cls)})
            return
        feat, thresh, _ = best_split(Z, y, feature_indices, min_leaf)
        if feat == -1:
            cls = 1 if np.mean(y) >= 0.5 else 0
            nodes.append({"leaf": True, "class": int(cls)})
            return
        left_mask = Z[:, feat] <= thresh
        right_mask = ~left_mask
        # internal node
        nodes.append({"leaf": False, "feat": int(feat), "thresh": float(thresh),
                      "left": None, "right": None})
        cur_idx = len(nodes) - 1
        # build left
        left_nodes_before = len(nodes)
        recurse(cur_idx)  # this will append left subtree nodes
        left_child_idx = cur_idx + 1 if len(nodes) > left_nodes_before else -1
        # build right
        right_nodes_before = len(nodes)
        recurse(cur_idx)  # this will append right subtree nodes
        right_child_idx = left_child_idx + (len(nodes) - left_nodes_before) if len(nodes) > right_nodes_before else -1
        # update current node with child indices
        nodes[cur_idx]["left"] = left_child_idx
        nodes[cur_idx]["right"] = right_child_idx
    recurse(0)
    return nodes


def train_and_eval(Z_train, y_train, Z_tune, y_tune, max_depth, min_leaf):
    nodes = build_tree(Z_train, y_train, max_depth, min_leaf)
    # evaluate on tune
    def predict_one(z):
        idx = 0
        while True:
            node = nodes[idx]
            if node["leaf"]:
                return node["class"]
            if z[node["feat"]] <= node["thresh"]:
                idx = node["left"]
            else:
                idx = node["right"]
    preds = np.array([predict_one(z) for z in Z_tune])
    acc = np.mean(preds == y_tune)
    return acc, nodes


def write_weights(enc_info, nodes, n_orig, n_encoded, path):
    # magic: 0x43415254 ('CART')
    buf = bytearray()
    buf += struct.pack("<I", 0x43415254)
    buf += struct.pack("<B", n_orig)
    buf += struct.pack("<H", n_encoded)
    # encoding info
    for info in enc_info:
        if info["type"] == "ordinal":
            buf += struct.pack("<B", 0)  # type ordinal
            buf += struct.pack("<f", info["mean"])
            buf += struct.pack("<f", info["std"])
        else:
            cats = info["categories"]
            buf += struct.pack("<B", 1)  # type categorical
            buf += struct.pack("<B", len(cats))
            buf += cats.tobytes()
    # tree nodes
    buf += struct.pack("<I", len(nodes))
    for node in nodes:
        if node["leaf"]:
            buf += struct.pack("<B", 1)
            buf += struct.pack("<B", node["class"])
        else:
            buf += struct.pack("<B", 0)
            buf += struct.pack("<H", node["feat"])
            buf += struct.pack("<f", node["thresh"])
            buf += struct.pack("<i", node["left"])
            buf += struct.pack("<i", node["right"])
    with open(path, "wb") as f:
        f.write(buf)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    types = card["input"]["types"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Encode all data
    Z, enc_info = encode_features(X, types, rng)
    n_encoded = Z.shape[1]

    # Split tuning (last 20% in file order)
    tune_size = int(n * TUNING_FRAC)
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_tune, y_tune = Z[tune_idx], y[tune_idx]

    # Grid search
    best_acc = -1.0
    best_params = None
    best_nodes = None
    for max_depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_LEAF_GRID:
            acc, nodes = train_and_eval(Z_train, y_train, Z_tune, y_tune, max_depth, min_leaf)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_leaf)
                best_nodes = nodes

    # Retrain on full data with best params
    max_depth, min_leaf = best_params
    final_nodes = build_tree(Z, y, max_depth, min_leaf)

    # Write weights
    write_weights(enc_info, final_nodes, card["input"]["bytes"], n_encoded, sys.argv[4])


if __name__ == "__main__":
    main()
```

```c
/* label-cart inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (original features). Output record: 1 byte (class 0/1). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    uint8_t type;        // 0 ordinal, 1 categorical
    float mean, std;     // ordinal
    uint8_t n_cats;      // categorical
    uint8_t *cats;       // categorical values (sorted)
    uint16_t start_idx;  // start index in encoded feature vector
    uint16_t dim;        // dimension (1 for ordinal, n_cats for categorical)
} FeatureEnc;

typedef struct {
    uint8_t is_leaf;
    uint8_t cls;         // if leaf
    uint16_t feat;       // if internal: encoded feature index
    float thresh;        // if internal
    int32_t left, right; // child indices
} Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    const unsigned char *p = wb;
    if (nw < 4) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24;
    if (magic != 0x43415254) { fprintf(stderr, "bad magic\n"); return 3; }
    p += 4;

    uint8_t n_orig = *p++; 
    uint16_t n_encoded = p[0] | p[1]<<8; p += 2;

    FeatureEnc *enc = calloc(n_orig, sizeof(FeatureEnc));
    uint16_t encoded_idx = 0;
    for (int i = 0; i < n_orig; i++) {
        enc[i].type = *p++;
        if (enc[i].type == 0) { // ordinal
            enc[i].mean = *(float*)p; p += 4;
            enc[i].std  = *(float*)p; p += 4;
            enc[i].dim = 1;
        } else { // categorical
            enc[i].n_cats = *p++;
            enc[i].cats = malloc(enc[i].n_cats);
            memcpy(enc[i].cats, p, enc[i].n_cats); p += enc[i].n_cats;
            enc[i].dim = enc[i].n_cats;
        }
        enc[i].start_idx = encoded_idx;
        encoded_idx += enc[i].dim;
    }
    if (encoded_idx != n_encoded) { fprintf(stderr, "encoded dim mismatch\n"); return 3; }

    uint32_t n_nodes = *(uint32_t*)p; p += 4;
    Node *nodes = malloc(n_nodes * sizeof(Node));
    for (uint32_t i = 0; i < n_nodes; i++) {
        nodes[i].is_leaf = *p++;
        if (nodes[i].is_leaf) {
            nodes[i].cls = *p++;
        } else {
            nodes[i].feat = p[0] | p[1]<<8; p += 2;
            nodes[i].thresh = *(float*)p; p += 4;
            nodes[i].left = *(int32_t*)p; p += 4;
            nodes[i].right = *(int32_t*)p; p += 4;
        }
    }

    float *encoded = malloc(n_encoded * sizeof(float));
    for (long rec_i = 0; rec_i < N; rec_i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != n_orig) { fprintf(stderr, "item %ld is %u bytes, not %d\n", rec_i, k, n_orig); return 3; }

        // Encode this record
        for (int i = 0; i < n_orig; i++) {
            uint8_t val = xi[i];
            if (enc[i].type == 0) {
                encoded[enc[i].start_idx] = (val - enc[i].mean) / enc[i].std;
            } else {
                // one-hot
                for (int d = 0; d < enc[i].dim; d++)
                    encoded[enc[i].start_idx + d] = 0.0f;
                // binary search in cats (they are sorted)
                int lo = 0, hi = enc[i].n_cats - 1, found = -1;
                while (lo <= hi) {
                    int mid = (lo + hi) >> 1;
                    if (enc[i].cats[mid] == val) { found = mid; break; }
                    else if (enc[i].cats[mid] < val) lo = mid + 1;
                    else hi = mid - 1;
                }
                if (found >= 0)
                    encoded[enc[i].start_idx + found] = 1.0f;
                // unseen category (255) -> all zeros (handled by init)
            }
        }

        // Traverse tree
        int32_t idx = 0;
        while (1) {
            Node *nd = &nodes[idx];
            if (nd->is_leaf) {
                unsigned char out = nd->cls;
                put_rec(stdout, &out, 1);
                break;
            }
            float fval = encoded[nd->feat];
            if (fval <= nd->thresh) idx = nd->left;
            else idx = nd->right;
        }
    }
    return 0;
}
```