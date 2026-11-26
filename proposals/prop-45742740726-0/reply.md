```hypothesis
Family: label-gbst (gradient boosted shallow trees). The adult dataset has mixed ordinal/categorical features; shallow trees capture interactions that linear models miss. With 50 trees of depth 3, we expect >0.855 accuracy at ~1500 Ir/item. If accuracy ≤0.852 or Ir/item >2500, the model does not justify its complexity.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 0.1
MAX_DEPTH = 3
MIN_SAMPLES = 10
MAX_TREES = 200
VAL_FRACTION = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Tree:
    __slots__ = ("feat", "thresh", "left", "right", "value")
    def __init__(self, feat=-1, thresh=0.0, left=-1, right=-1, value=0.0):
        self.feat = feat
        self.thresh = thresh
        self.left = left
        self.right = right
        self.value = value

def build_tree(Z, g, h, idxs, depth):
    n = len(idxs)
    if n < MIN_SAMPLES or depth == MAX_DEPTH:
        sum_g = g[idxs].sum()
        sum_h = h[idxs].sum()
        return Tree(value = float(sum_g / sum_h) if sum_h != 0 else 0.0)
    best_gain = 0.0
    best_feat = -1
    best_thresh = 0.0
    sum_g = g[idxs].sum()
    sum_h = h[idxs].sum()
    parent_score = sum_g * sum_g / sum_h if sum_h != 0 else 0.0
    for f in range(Z.shape[1]):
        vals = Z[idxs, f]
        order = np.argsort(vals)
        sorted_vals = vals[order]
        sorted_g = g[idxs][order]
        sorted_h = h[idxs][order]
        # prefix sums
        pref_g = np.cumsum(sorted_g)
        pref_h = np.cumsum(sorted_h)
        for i in range(1, n):
            if sorted_vals[i] == sorted_vals[i-1]:
                continue
            left_g = pref_g[i-1]
            left_h = pref_h[i-1]
            right_g = sum_g - left_g
            right_h = sum_h - left_h
            if left_h == 0 or right_h == 0:
                continue
            gain = left_g*left_g/left_h + right_g*right_g/right_h - parent_score
            if gain > best_gain:
                best_gain = gain
                best_feat = f
                best_thresh = (sorted_vals[i-1] + sorted_vals[i]) * 0.5
    if best_gain <= 0:
        return Tree(value = float(sum_g / sum_h) if sum_h != 0 else 0.0)
    left_mask = Z[idxs, best_feat] <= best_thresh
    left_idxs = idxs[left_mask]
    right_idxs = idxs[~left_mask]
    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return Tree(value = float(sum_g / sum_h) if sum_h != 0 else 0.0)
    left = build_tree(Z, g, h, left_idxs, depth+1)
    right = build_tree(Z, g, h, right_idxs, depth+1)
    return Tree(feat=best_feat, thresh=best_thresh, left=left, right=right)

def predict_tree(tree, x):
    while tree.feat != -1:
        if x[tree.feat] <= tree.thresh:
            tree = tree.left
        else:
            tree = tree.right
    return tree.value

def train_gbst(Z, y, val_split):
    n = Z.shape[0]
    val_size = int(n * val_split)
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)
    p = y[train_idx].mean()
    base_score = math.log(p / (1 - p)) if p not in (0,1) else 0.0
    train_pred = np.full(len(train_idx), base_score, dtype=np.float32)
    val_pred = np.full(len(val_idx), base_score, dtype=np.float32)
    trees = []
    best_val_loss = float('inf')
    best_iter = 0
    for it in range(MAX_TREES):
        # train gradients
        p_train = 1.0 / (1.0 + np.exp(-train_pred))
        g = p_train - y[train_idx]
        h = p_train * (1.0 - p_train)
        idxs = np.arange(len(train_idx))
        tree = build_tree(Z[train_idx], g, h, idxs, 0)
        trees.append(tree)
        # update predictions
        for i, idx in enumerate(train_idx):
            train_pred[i] += LR * predict_tree(tree, Z[idx])
        for i, idx in enumerate(val_idx):
            val_pred[i] += LR * predict_tree(tree, Z[idx])
        # validation loss
        val_p = 1.0 / (1.0 + np.exp(-val_pred))
        val_loss = -np.mean(y[val_idx] * np.log(val_p + 1e-15) + (1 - y[val_idx]) * np.log(1 - val_p + 1e-15))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_iter = it
        elif it - best_iter >= 20:
            break
    # retrain on all data with best_iter+1 trees
    p = y.mean()
    base_score = math.log(p / (1 - p)) if p not in (0,1) else 0.0
    pred = np.full(n, base_score, dtype=np.float32)
    final_trees = []
    for it in range(best_iter + 1):
        p_all = 1.0 / (1.0 + np.exp(-pred))
        g = p_all - y
        h = p_all * (1.0 - p_all)
        idxs = np.arange(n)
        tree = build_tree(Z, g, h, idxs, 0)
        final_trees.append(tree)
        for i in range(n):
            pred[i] += LR * predict_tree(tree, Z[i])
    return base_score, final_trees

def flatten_tree(tree, nodes):
    # pre-order traversal, assign indices
    idx = len(nodes)
    nodes.append(tree)
    if tree.feat != -1:
        left_start = len(nodes)
        flatten_tree(tree.left, nodes)
        right_start = len(nodes)
        flatten_tree(tree.right, nodes)
        nodes[idx].left = left_start
        nodes[idx].right = right_start
    else:
        nodes[idx].left = -1
        nodes[idx].right = -1

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    # encoding
    enc_info = []
    Z_parts = []
    for j, t in enumerate(types):
        col = X_bytes[:, j]
        if t == "ordinal":
            mu = col.astype(np.float64).mean()
            sd = col.astype(np.float64).std()
            if sd == 0: sd = 1.0
            enc_info.append(("ordinal", float(mu), float(sd)))
            Z_parts.append(((col - mu) / sd).astype(np.float32).reshape(-1, 1))
        else:
            vals = np.unique(col)
            vals = vals[vals != 255]  # 255 is unseen, not in training
            mapping = np.full(256, -1, dtype=np.int32)
            for idx, v in enumerate(vals):
                mapping[v] = idx
            enc_info.append(("categorical", len(vals), mapping))
            onehot = np.zeros((len(col), len(vals)), dtype=np.float32)
            for idx, v in enumerate(vals):
                onehot[col == v, idx] = 1.0
            Z_parts.append(onehot)
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    # train
    base_score, trees = train_gbst(Z, y, VAL_FRACTION)
    # flatten trees
    flat_trees = []
    for t in trees:
        nodes = []
        flatten_tree(t, nodes)
        flat_trees.append(nodes)
    # write weights
    magic = 0x47425354  # 'GBST'
    feat_dim = Z.shape[1]
    num_trees = len(flat_trees)
    # compute total size for validation
    header_size = 4 + 4 + 4 + 4 + 4  # magic, d, feat_dim, num_trees, base_score
    col_sizes = 0
    for info in enc_info:
        if info[0] == "ordinal":
            col_sizes += 1 + 4 + 4  # type, mu, sd
        else:
            col_sizes += 1 + 4 + 256 * 4  # type, num_cats, mapping[256]
    tree_sizes = 0
    for nodes in flat_trees:
        tree_sizes += 4  # num_nodes
        for nd in nodes:
            tree_sizes += 4 + 4 + 4 + 4 + 4  # feat, thresh, left, right, value
    total_size = header_size + col_sizes + tree_sizes
    # pack
    out = bytearray()
    out += struct.pack("<I", magic)
    out += struct.pack("<i", d)
    out += struct.pack("<i", feat_dim)
    out += struct.pack("<i", num_trees)
    out += struct.pack("<f", base_score)
    for info in enc_info:
        if info[0] == "ordinal":
            out += struct.pack("<B", 0)
            out += struct.pack("<f", info[1])
            out += struct.pack("<f", info[2])
        else:
            out += struct.pack("<B", 1)
            out += struct.pack("<i", info[1])
            out += info[2].astype("<i").tobytes()
    for nodes in flat_trees:
        out += struct.pack("<i", len(nodes))
        for nd in nodes:
            out += struct.pack("<i", nd.feat)
            out += struct.pack("<f", nd.thresh)
            out += struct.pack("<i", nd.left)
            out += struct.pack("<i", nd.right)
            out += struct.pack("<f", nd.value)
    if len(out) != total_size:
        print("size mismatch", len(out), total_size, file=sys.stderr)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    int32_t feat;
    float thresh;
    int32_t left;
    int32_t right;
    float value;
} Node;

typedef struct {
    int32_t num_nodes;
    Node *nodes;
} Tree;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    const unsigned char *ptr = wb;
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = ptr[0] | ptr[1]<<8 | ptr[2]<<16 | ptr[3]<<24; ptr += 4;
    if (magic != 0x47425354) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    int32_t d = *(int32_t*)ptr; ptr += 4;
    int32_t feat_dim = *(int32_t*)ptr; ptr += 4;
    int32_t num_trees = *(int32_t*)ptr; ptr += 4;
    float base_score = *(float*)ptr; ptr += 4;

    // column encodings
    float *ord_mu = malloc(d * sizeof(float));
    float *ord_sd = malloc(d * sizeof(float));
    int32_t *cat_num = malloc(d * sizeof(int32_t));
    int32_t **cat_map = malloc(d * sizeof(int32_t*));
    int is_ordinal[14];
    for (int j = 0; j < d; j++) {
        uint8_t type = *ptr++; ptr++;
        if (type == 0) { // ordinal
            is_ordinal[j] = 1;
            ord_mu[j] = *(float*)ptr; ptr += 4;
            ord_sd[j] = *(float*)ptr; ptr += 4;
            cat_num[j] = 0;
            cat_map[j] = NULL;
        } else { // categorical
            is_ordinal[j] = 0;
            cat_num[j] = *(int32_t*)ptr; ptr += 4;
            cat_map[j] = (int32_t*)ptr; ptr += 256 * 4;
            ord_mu[j] = 0; ord_sd[j] = 1;
        }
    }

    // trees
    Tree *trees = malloc(num_trees * sizeof(Tree));
    for (int t = 0; t < num_trees; t++) {
        int32_t nn = *(int32_t*)ptr; ptr += 4;
        trees[t].num_nodes = nn;
        trees[t].nodes = (Node*)ptr;
        ptr += nn * sizeof(Node);
    }
    if (ptr - wb != nw) { fprintf(stderr, "bad weights: size mismatch\n"); return 3; }

    // inference
    float *feat = malloc(feat_dim * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // build feature vector
        int fidx = 0;
        for (int j = 0; j < d; j++) {
            if (is_ordinal[j]) {
                feat[fidx++] = (xi[j] - ord_mu[j]) / ord_sd[j];
            } else {
                int32_t map_idx = cat_map[j][xi[j]];
                if (map_idx >= 0) {
                    feat[fidx + map_idx] = 1.0f;
                }
                fidx += cat_num[j];
            }
        }
        // predict
        float score = base_score;
        for (int t = 0; t < num_trees; t++) {
            Node *nodes = trees[t].nodes;
            int32_t node_idx = 0;
            while (1) {
                Node *nd = &nodes[node_idx];
                if (nd->feat == -1) {
                    score += 0.1f * nd->value; // LR = 0.1
                    break;
                }
                if (feat[nd->feat] <= nd->thresh)
                    node_idx = nd->left;
                else
                    node_idx = nd->right;
            }
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```