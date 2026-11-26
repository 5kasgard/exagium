```hypothesis
Family: label-rf (Random Forest). The 8x8 digit pixels exhibit local spatial structure (strokes, loops) that linear models cannot capture; an ensemble of shallow axis-aligned trees models these interactions while inference remains ~100 node visits/item. Hyperparameters (n_trees∈{5,10,15}, max_depth∈{8,10,12,15}, min_samples_leaf∈{1,5,10}, max_features∈{8,16,32,64}) are tuned on the last 20% of train.in. If dev accuracy ≤ 0.965 the family does not justify its complexity over label-linear.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

class Node:
    __slots__ = ('is_leaf', 'feature', 'threshold', 'left', 'right', 'class_label')
    def __init__(self, is_leaf=False, feature=0, threshold=0, left=None, right=None, class_label=0):
        self.is_leaf = is_leaf
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.class_label = class_label

def entropy(y):
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p + 1e-12))

def info_gain(y, left_mask, right_mask):
    n = len(y)
    if n == 0: return 0.0
    h = entropy(y)
    nL, nR = np.sum(left_mask), np.sum(right_mask)
    if nL == 0 or nR == 0: return 0.0
    return h - (nL/n)*entropy(y[left_mask]) - (nR/n)*entropy(y[right_mask])

def build_tree(X, y, rng, max_depth, min_samples_leaf, max_features, depth=0):
    n, d = X.shape
    if depth >= max_depth or n < 2 * min_samples_leaf or len(np.unique(y)) == 1:
        vals, counts = np.unique(y, return_counts=True)
        return Node(is_leaf=True, class_label=int(vals[np.argmax(counts)]))
    
    features = rng.choice(d, size=min(max_features, d), replace=False)
    best_gain = -1.0
    best_feat = -1
    best_thresh = -1
    best_left_mask = None
    best_right_mask = None
    
    for feat in features:
        vals = X[:, feat]
        uniq = np.unique(vals)
        if len(uniq) <= 1:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        for thresh in thresholds:
            left_mask = vals <= thresh
            right_mask = ~left_mask
            if np.sum(left_mask) < min_samples_leaf or np.sum(right_mask) < min_samples_leaf:
                continue
            gain = info_gain(y, left_mask, right_mask)
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh
                best_left_mask = left_mask
                best_right_mask = right_mask
    
    if best_gain <= 0 or best_feat == -1:
        vals, counts = np.unique(y, return_counts=True)
        return Node(is_leaf=True, class_label=int(vals[np.argmax(counts)]))
    
    left_node = build_tree(X[best_left_mask], y[best_left_mask], rng, max_depth, min_samples_leaf, max_features, depth+1)
    right_node = build_tree(X[best_right_mask], y[best_right_mask], rng, max_depth, min_samples_leaf, max_features, depth+1)
    return Node(is_leaf=False, feature=best_feat, threshold=best_thresh, left=left_node, right=right_node)

def flatten_tree(node, nodes_list):
    idx = len(nodes_list)
    if node.is_leaf:
        nodes_list.append((1, node.class_label, 0, -1, -1))
    else:
        nodes_list.append((0, node.feature, node.threshold, -1, -1))
        left_idx = flatten_tree(node.left, nodes_list)
        right_idx = flatten_tree(node.right, nodes_list)
        nodes_list[idx] = (0, node.feature, node.threshold, left_idx, right_idx)
    return idx

def predict_tree(nodes, x):
    idx = 0
    while True:
        is_leaf, feat, thresh, left, right = nodes[idx]
        if is_leaf:
            return feat
        if x[feat] <= thresh:
            idx = left
        else:
            idx = right

def evaluate_forest(trees, X, y):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        votes = np.zeros(10, dtype=int)
        for tree_nodes in trees:
            c = predict_tree(tree_nodes, X[i])
            votes[c] += 1
        pred = np.argmax(votes)
        if pred == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = X_all.shape[0]
    d = card["input"]["bytes"]
    C = card["classes"]
    
    val_size = n_total // 5
    train_size = n_total - val_size
    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val, y_val = X_all[train_size:], y_all[train_size:]
    
    param_grid = {
        'n_trees': [5, 10, 15],
        'max_depth': [8, 10, 12, 15],
        'min_samples_leaf': [1, 5, 10],
        'max_features': [8, 16, 32, 64]
    }
    
    best_acc = -1.0
    best_params = None
    best_trees = None
    
    for n_trees in param_grid['n_trees']:
        for max_depth in param_grid['max_depth']:
            for min_samples_leaf in param_grid['min_samples_leaf']:
                for max_features in param_grid['max_features']:
                    trees = []
                    for t in range(n_trees):
                        bs_rng = np.random.default_rng(SEED + t * 1000 + max_depth * 100 + min_samples_leaf * 10 + max_features)
                        idx = bs_rng.integers(0, train_size, size=train_size)
                        X_bs = X_train[idx]
                        y_bs = y_train[idx]
                        tree_rng = np.random.default_rng(SEED + t * 10000 + max_depth * 1000 + min_samples_leaf * 100 + max_features)
                        root = build_tree(X_bs, y_bs, tree_rng, max_depth, min_samples_leaf, max_features)
                        nodes = []
                        flatten_tree(root, nodes)
                        trees.append(nodes)
                    
                    acc = evaluate_forest(trees, X_val, y_val)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (n_trees, max_depth, min_samples_leaf, max_features)
                        best_trees = trees
    
    print(f"Best params: {best_params}, val acc: {best_acc:.4f}", file=sys.stderr)
    
    n_trees, max_depth, min_samples_leaf, max_features = best_params
    final_trees = []
    for t in range(n_trees):
        bs_rng = np.random.default_rng(SEED + t * 1000 + max_depth * 100 + min_samples_leaf * 10 + max_features)
        idx = bs_rng.integers(0, n_total, size=n_total)
        X_bs = X_all[idx]
        y_bs = y_all[idx]
        tree_rng = np.random.default_rng(SEED + t * 10000 + max_depth * 1000 + min_samples_leaf * 100 + max_features)
        root = build_tree(X_bs, y_bs, tree_rng, max_depth, min_samples_leaf, max_features)
        nodes = []
        flatten_tree(root, nodes)
        final_trees.append(nodes)
    
    out = bytearray()
    out += struct.pack("<III", 0x52465354, d, C)
    out += struct.pack("<I", n_trees)
    for tree_nodes in final_trees:
        out += struct.pack("<I", len(tree_nodes))
        for node in tree_nodes:
            is_leaf, feat, thresh, left, right = node
            if is_leaf:
                thresh_int = 0
            else:
                thresh_int = int(math.floor(thresh))
            out += struct.pack("<BBBBii", is_leaf, feat, thresh_int, 0, left, right)
    
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
    FILE *