```hypothesis
Family: label-tree (single CART decision tree with binary splits on binned features).
Why: Adult data has mixed categorical/ordinal features already quantized to 256 bins; a shallow tree captures interactions that linear models miss, while inference is a few branch predictions (extremely low CPU cost). The tuning split selects max_depth and min_samples_leaf to avoid overfitting.
Failure signal: If dev accuracy <= 0.8524 (label-linear) or inference cost exceeds label-linear, the family does not improve on the baseline.
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

def gini(count0, count1):
    total = count0 + count1
    if total == 0:
        return 0.0
    p0 = count0 / total
    p1 = count1 / total
    return 1.0 - p0*p0 - p1*p1

class Node:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'cls')
    def __init__(self, feature, threshold, left, right, cls):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.cls = cls

def build_tree(X, y, depth, max_depth, min_samples_leaf):
    n = len(y)
    count1 = np.sum(y)
    count0 = n - count1
    if count0 == 0 or count1 == 0 or n < min_samples_leaf or depth == max_depth:
        cls = 0 if count0 >= count1 else 1
        return Node(-1, 0, None, None, cls)
    
    best_gini = gini(count0, count1)
    best_feat = -1
    best_thresh = -1
    
    for j in range(X.shape[1]):
        vals = X[:, j]
        hist0 = np.bincount(vals[y==0], minlength=256)
        hist1 = np.bincount(vals[y==1], minlength=256)
        cum0 = np.cumsum(hist0)
        cum1 = np.cumsum(hist1)
        total0 = cum0[-1]
        total1 = cum1[-1]
        for t in range(255):
            left0 = cum0[t]
            left1 = cum1[t]
            right0 = total0 - left0
            right1 = total1 - left1
            if left0+left1 == 0 or right0+right1 == 0:
                continue
            gini_left = gini(left0, left1)
            gini_right = gini(right0, right1)
            weighted_gini = (left0+left1)*gini_left + (right0+right1)*gini_right
            if weighted_gini < best_gini - 1e-12:
                best_gini = weighted_gini
                best_feat = j
                best_thresh = t
            elif abs(weighted_gini - best_gini) < 1e-12:
                if j < best_feat or (j == best_feat and t < best_thresh):
                    best_feat = j
                    best_thresh = t
    
    if best_feat == -1:
        cls = 0 if count0 >= count1 else 1
        return Node(-1, 0, None, None, cls)
    
    left_mask = X[:, best_feat] <= best_thresh
    right_mask = ~left_mask
    left = build_tree(X[left_mask], y[left_mask], depth+1, max_depth, min_samples_leaf)
    right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth, min_samples_leaf)
    return Node(best_feat, best_thresh, left, right, 0)

def flatten(node, flat_nodes):
    idx = len(flat_nodes)
    flat_nodes.append(node)
    if node.left is not None:
        left_idx = flatten(node.left, flat_nodes)
        node.left = left_idx
    else:
        node.left = -1
    if node.right is not None:
        right_idx = flatten(node.right, flat_nodes)
        node.right = right_idx
    else:
        node.right = -1
    return idx

def train_tree(X, y, max_depth, min_samples_leaf):
    root = build_tree(X, y, 0, max_depth, min_samples_leaf)
    flat_nodes = []
    flatten(root, flat_nodes)
    return flat_nodes

def evaluate(flat_nodes, X, y):
    n = len(y)
    correct = 0
    for i in range(n):
        xi = X[i]
        node_idx = 0
        while True:
            node = flat_nodes[node_idx]
            if node.feature == -1:
                pred = node.cls
                break
            val = xi[node.feature]
            if val <= node.threshold:
                node_idx = node.left
            else:
                node_idx = node.right
        if pred == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = len(y)
    split_idx = int(0.8 * n)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    best_acc = -1.0
    best_params = None
    for max_depth in [3,4,5,6,7,8,9,10]:
        for min_samples_leaf in [1,2,5,10,20,50]:
            flat_nodes = train_tree(X_train, y_train, max_depth, min_samples_leaf)
            acc = evaluate(flat_nodes, X_val, y_val)
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_params = (max_depth, min_samples_leaf)
            elif abs(acc - best_acc) < 1e-12:
                if max_depth < best_params[0] or (max_depth == best_params[0] and min_samples_leaf > best_params[1]):
                    best_params = (max_depth, min_samples_leaf)
    
    print(f"Best params: max_depth={best_params[0]}, min_samples_leaf={best_params[1]}, val_acc={best_acc:.4f}", file=sys.stderr)
    
    flat_nodes = train_tree(X, y, best_params[0], best_params[1])
    
    magic = 0x45455254
    n_nodes = len(flat_nodes)
    n_features = X.shape[1]
    n_classes = 2
    out = struct.pack('<Iiii', magic, n_nodes, n_features, n_classes)
    for node in flat_nodes:
        out += struct.pack('<bBiib', node.feature, node.threshold, node.left, node.right, node.cls)
    
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x45455254) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t n_nodes = (int32_t)(wb[4] | wb[5] << 8 | wb[6] << 16 | (uint32_t)wb[7] << 24);
    int32_t n_features = (int32_t)(wb[8] | wb[9] << 8 | wb[10] << 16 | (uint32_t)wb[11] << 24);
    int32_t n_classes = (int32_t)(wb[12] | wb[13] << 8 | wb[14] << 16 | (uint32_t)wb[15] << 24);
    if (n_features != 14 || n_classes != 2 || n_nodes <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + (long)n_nodes * 11;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    int8_t *feature = malloc(n_nodes * sizeof(int8_t));
    uint8_t *threshold = malloc(n_nodes * sizeof(uint8_t));
    int32_t *left = malloc(n_nodes * sizeof(int32_t));
    int32_t *right = malloc(n_nodes * sizeof(int32_t));
    uint8_t *cls = malloc(n_nodes * sizeof(uint8_t));
    unsigned char *ptr = wb + 16;
    for (int i = 0; i < n_nodes; i++) {
        feature[i] = (int8_t)ptr[0];
        threshold[i] = ptr[1];
        left[i] = (int32_t)(ptr[2] | ptr[3] << 8 | ptr[4] << 16 | (uint32_t)ptr[5] << 24);
        right[i] = (int32_t)(ptr[6] | ptr[7] << 8 | ptr[8] << 16 | (uint32_t)ptr[9] << 24);
        cls[i] = ptr[10];
        ptr += 11;
    }
    free(wb);
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != n_features) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, n_features); return 3; }
        int node_idx = 0;
        while (1) {
            if (feature[node_idx] == -1) {
                unsigned char pred = cls[node_idx];
                put_rec(stdout, &pred, 1);
                break;
            }
            uint8_t val = xi[feature[node_idx]];
            if (val <= threshold[node_idx]) {
                node_idx = left[node_idx];
            } else {
                node_idx = right[node_idx];
            }
        }
    }
    free(x);
    free(feature); free(threshold); free(left); free(right); free(cls);
    return 0;
}
```