```hypothesis
Family: label-dtree (single deterministic decision tree).
Why: Optdigits pixels have strong local spatial correlations (strokes) that a linear model misses. A single tree of depth 10-15 captures conjunctive pixel patterns (e.g., "pixel (2,3)>5 AND pixel (3,4)>8") with zero multiplications at inference—just 64-byte array lookups and integer compares. This should exceed label-linear's 0.9613 while staying far below label-gbdt's 41k Ir/item.
Failure signal: Dev accuracy ≤ 0.9613 or Ir/item ≥ 10k (≈2× label-linear).
```

```python
"""label-dtree: single deterministic decision tree for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Builds a CART tree (Gini, binary splits on ordinal thresholds) with max_depth tuned
on the last 20% of train.in (file order), then retrained on all data.
Weights format: magic 0x44545245, n_nodes, n_features, n_classes,
then per node: u8 is_leaf, u8 feat, u8 thresh, u8 class, i32 left, i32 right."""
import json, struct, sys
import numpy as np

MIN_SAMPLES_LEAF = 5
DEPTH_CANDIDATES = [5, 7, 10, 12, 15, 20]
MAGIC = 0x44545245  # "DTRE"

def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
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

def build_tree(X, y, idx, depth, max_depth, nodes, feat_vals):
    """Returns node index. Nodes appended to list as dicts."""
    n = len(idx)
    if n < MIN_SAMPLES_LEAF or depth == max_depth:
        cls = int(np.bincount(y[idx]).argmax())
        nodes.append({"leaf": True, "class": cls})
        return len(nodes) - 1
    if np.all(y[idx] == y[idx[0]]):
        cls = int(y[idx[0]])
        nodes.append({"leaf": True, "class": cls})
        return len(nodes) - 1

    best_gain = -1.0
    best_f = best_t = -1
    parent_gini = gini(y[idx])
    for f in range(X.shape[1]):
        vals = X[idx, f]
        for t in range(16):  # thresholds 0..15 => split <=t vs >t
            left_mask = vals <= t
            if not np.any(left_mask) or np.all(left_mask):
                continue
            n_left = np.count_nonzero(left_mask)
            n_right = n - n_left
            gain = parent_gini - (n_left/n)*gini(y[idx[left_mask]]) - (n_right/n)*gini(y[idx[~left_mask]])
            if gain > best_gain or (gain == best_gain and (f < best_f or (f == best_f and t < best_t))):
                best_gain, best_f, best_t = gain, f, t
    if best_gain <= 1e-12:
        cls = int(np.bincount(y[idx]).argmax())
        nodes.append({"leaf": True, "class": cls})
        return len(nodes) - 1

    left_idx = idx[X[idx, best_f] <= best_t]
    right_idx = idx[X[idx, best_f] > best_t]
    # placeholder; will fill children after recursion
    node_idx = len(nodes)
    nodes.append({"leaf": False, "feat": best_f, "thresh": best_t, "left": -1, "right": -1})
    left_node = build_tree(X, y, left_idx, depth+1, max_depth, nodes, feat_vals)
    right_node = build_tree(X, y, right_idx, depth+1, max_depth, nodes, feat_vals)
    nodes[node_idx]["left"] = left_node
    nodes[node_idx]["right"] = right_node
    return node_idx

def predict_tree(X_row, nodes):
    node = 0
    while True:
        nd = nodes[node]
        if nd["leaf"]:
            return nd["class"]
        if X_row[nd["feat"]] <= nd["thresh"]:
            node = nd["left"]
        else:
            node = nd["right"]

def eval_tree(X, y, nodes):
    correct = 0
    for i in range(len(X)):
        if predict_tree(X[i], nodes) == y[i]:
            correct += 1
    return correct / len(y)

def write_weights(nodes, n_feat, n_class, path):
    n_nodes = len(nodes)
    # header: magic, n_nodes, n_feat, n_class (all i32)
    buf = struct.pack("<4i", MAGIC, n_nodes, n_feat, n_class)
    for nd in nodes:
        if nd["leaf"]:
            buf += struct.pack("<BBBBii", 1, 0, 0, nd["class"], 0, 0)
        else:
            buf += struct.pack("<BBBBii", 0, nd["feat"], nd["thresh"], 0, nd["left"], nd["right"])
    open(path, "wb").write(buf)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    split = int(0.8 * n)
    idx_tr, idx_val = np.arange(split), np.arange(split, n)

    best_depth, best_acc = DEPTH_CANDIDATES[0], -1.0
    for d in DEPTH_CANDIDATES:
        nodes = []
        build_tree(X, y, idx_tr, 0, d, nodes, None)
        acc = eval_tree(X[idx_val], y[idx_val], nodes)
        if acc > best_acc:
            best_acc, best_depth = acc, d

    # retrain on all data
    nodes = []
    build_tree(X, y, np.arange(n), 0, best_depth, nodes, None)
    write_weights(nodes, X.shape[1], card["classes"], sys.argv[4])

if __name__ == "__main__":
    main()
```

```c
/* label-dtree inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 pixels). Output: 1 byte class.
 * Weights: magic 0x44545245, n_nodes, n_feat, n_class (i32 each),
 * then per node: u8 is_leaf, u8 feat, u8 thresh, u8 class, i32 left, i32 right. */
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

typedef struct { uint8_t leaf, feat, thresh, cls; int32_t left, right; } Node;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x44545245) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t n_nodes = hdr[1], n_feat = hdr[2], n_class = hdr[3];
    if (n_feat != 64 || n_class != 10 || n_nodes <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + (long)n_nodes * 12;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    Node *nodes = (Node *)(wb + 16);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        int node = 0;
        while (1) {
            Node nd = nodes[node];
            if (nd.leaf) { put_rec(stdout, &nd.cls, 1); break; }
            if (xi[nd.feat] <= nd.thresh) node = nd.left;
            else node = nd.right;
        }
    }
    return 0;
}
```