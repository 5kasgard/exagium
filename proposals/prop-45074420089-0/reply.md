```hypothesis
Family: label-tree. A single deterministic decision tree splitting on raw byte features (0-255) with Gini impurity, tuned on the last 20% of train.in for max_depth and min_samples_leaf. The Adult dataset's mix of ordinal and categorical features is well-suited to axis-aligned splits; a shallow tree should match or beat label-linear (0.8524) at lower inference cost (few comparisons per item). Failure if dev accuracy < 0.852 or inference cost > 1000 Ir/item.
```

```python
import json, struct, sys
import numpy as np

SEED = 1
np.random.seed(SEED)

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def gini(p):
    return 2.0 * p * (1.0 - p)

def best_split(X, y, min_samples_leaf):
    n, d = X.shape
    total1 = np.count_nonzero(y)
    total0 = n - total1
    parent_gini = gini(total1 / n) if n else 0.0
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1
    for feat in range(d):
        vals = X[:, feat]
        mask1 = (y == 1)
        mask0 = ~mask1
        counts1 = np.bincount(vals[mask1], minlength=256)
        counts0 = np.bincount(vals[mask0], minlength=256)
        cum1 = np.cumsum(counts1)
        cum0 = np.cumsum(counts0)
        for t in range(255):
            left1 = cum1[t]
            left0 = cum0[t]
            left_n = left1 + left0
            right1 = total1 - left1
            right0 = total0 - left0
            right_n = right1 + right0
            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue
            gain = parent_gini - (left_n/n)*gini(left1/left_n) - (right_n/n)*gini(right1/right_n)
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = t
    return best_gain, best_feat, best_thresh

def build_tree(X, y, depth, max_depth, min_samples_leaf):
    if depth >= max_depth or len(y) < 2*min_samples_leaf or np.all(y == y[0]):
        return {'leaf': True, 'class': int(np.round(np.mean(y)))}
    gain, feat, thresh = best_split(X, y, min_samples_leaf)
    if gain <= 0:
        return {'leaf': True, 'class': int(np.round(np.mean(y)))}
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    left_node = build_tree(X[left_mask], y[left_mask], depth+1, max_depth, min_samples_leaf)
    right_node = build_tree(X[right_mask], y[right_mask], depth+1, max_depth, min_samples_leaf)
    return {'leaf': False, 'feat': feat, 'thresh': thresh, 'left': left_node, 'right': right_node}

def tree_to_arrays(node):
    nodes = []
    def rec(n):
        idx = len(nodes)
        if n['leaf']:
            nodes.append((1, n['class'], 0, 0, 0, 0))
        else:
            nodes.append((0, 0, n['feat'], n['thresh'], -1, -1))
            left_idx = rec(n['left'])
            right_idx = rec(n['right'])
            nodes[idx] = (0, 0, n['feat'], n['thresh'], left_idx, right_idx)
        return idx
    rec(node)
    return nodes

def evaluate_tree(nodes, X):
    n_samples = X.shape[0]
    preds = np.zeros(n_samples, dtype=np.uint8)
    for i in range(n_samples):
        idx = 0
        while True:
            is_leaf, cls, feat, thresh, left, right = nodes[idx]
            if is_leaf:
                preds[i] = cls
                break
            if X[i, feat] <= thresh:
                idx = left
            else:
                idx = right
    return preds

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = len(X)
    split_idx = int(n * 0.8)
    train_X, train_y = X[:split_idx], y[:split_idx]
    tune_X, tune_y = X[split_idx:], y[split_idx:]
    
    best_acc = -1.0
    best_params = None
    best_nodes = None
    
    for max_depth in [3,4,5,6,7,8,9,10]:
        for min_samples_leaf in [10,20,50,100]:
            tree = build_tree(train_X, train_y, 0, max_depth, min_samples_leaf)
            nodes = tree_to_arrays(tree)
            preds = evaluate_tree(nodes, tune_X)
            acc = np.mean(preds == tune_y)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)
                best_nodes = nodes
    print(f"Best params: {best_params}, tune acc: {best_acc:.4f}", file=sys.stderr)
    
    tree = build_tree(X, y, 0, best_params[0], best_params[1])
    nodes = tree_to_arrays(tree)
    
    magic = 0x54524545
    num_nodes = len(nodes)
    out = struct.pack("<II", magic, num_nodes)
    for node in nodes:
        is_leaf, cls, feat, thresh, left, right = node
        out += struct.pack("<BBBBii", is_leaf, cls, feat, thresh, left, right)
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
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wbuf[0] | wbuf[1] << 8 | wbuf[2] << 16 | (uint32_t)wbuf[3] << 24;
    if (magic != 0x54524545) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t num_nodes = wbuf[4] | wbuf[5] << 8 | wbuf[6] << 16 | (uint32_t)wbuf[7] << 24;
    if (nw != 8 + (long)num_nodes * 12) { fprintf(stderr, "bad weights size\n"); return 3; }
    
    uint8_t *is_leaf = malloc(num_nodes);
    uint8_t *cls = malloc(num_nodes);
    uint8_t *feat = malloc(num_nodes);
    uint8_t *thresh = malloc(num_nodes);
    int32_t *left = malloc(num_nodes * sizeof(int32_t));
    int32_t *right = malloc(num_nodes * sizeof(int32_t));
    
    unsigned char *p = wbuf + 8;
    for (uint32_t i = 0; i < num_nodes; i++) {
        is_leaf[i] = p[0];
        cls[i] = p[1];
        feat[i] = p[2];
        thresh[i] = p[3];
        left[i] = (int32_t)(p[4] | p[5] << 8 | p[6] << 16 | (uint32_t)p[7] << 24);
        right[i] = (int32_t)(p[8] | p[9] << 8 | p[10] << 16 | (uint32