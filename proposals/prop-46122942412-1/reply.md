```hypothesis
Family: label-forest (small deterministic random forest). Optdigits pixels are ordinal 0-16; trees split on single pixels at integer thresholds, needing only integer compares at inference. A forest of 3-7 trees depth 4-7 captures nonlinearities that linear models miss, while inference visits ~25 nodes/item (far fewer ops than label-linear's 640 multiply-adds). Hyperparameters (n_trees, max_depth, max_features) chosen on last 20% of train. Failure: dev accuracy ≤ 0.9613 or Ir/item ≥ 5000.
```
```python
import json, struct, sys
import numpy as np

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

def best_split(X, y, features):
    best_gini = 1.0
    best_feat = -1
    best_thresh = -1
    for f in features:
        vals = X[:, f]
        for t in range(16):
            mask = vals <= t
            nl, nr = mask.sum(), (~mask).sum()
            if nl == 0 or nr == 0:
                continue
            g = (nl * gini(y[mask]) + nr * gini(y[~mask])) / len(y)
            if g < best_gini:
                best_gini = g
                best_feat = f
                best_thresh = t
    return best_feat, best_thresh

def build_tree(X, y, max_depth, max_features, rng, depth=0):
    if depth >= max_depth or len(np.unique(y)) == 1:
        return {'leaf': True, 'class': int(np.bincount(y).argmax())}
    n_feats = X.shape[1]
    n_sub = min(max_features, n_feats)
    feats = rng.choice(n_feats, size=n_sub, replace=False)
    feat, thresh = best_split(X, y, feats)
    if feat == -1:
        return {'leaf': True, 'class': int(np.bincount(y).argmax())}
    mask = X[:, feat] <= thresh
    left = build_tree(X[mask], y[mask], max_depth, max_features, rng, depth+1)
    right = build_tree(X[~mask], y[~mask], max_depth, max_features, rng, depth+1)
    return {'leaf': False, 'feat': int(feat), 'thresh': int(thresh), 'left': left, 'right': right}

def tree_to_flat(tree):
    nodes = []
    def rec(t):
        idx = len(nodes)
        if t['leaf']:
            nodes.append([1, t['class'], 0, -1, -1])
        else:
            nodes.append([0, t['feat'], t['thresh'], -1, -1])
            lidx = rec(t['left'])
            ridx = rec(t['right'])
            nodes[idx][3] = lidx
            nodes[idx][4] = ridx
        return idx
    rec(tree)
    return nodes

def eval_forest(trees, X, y):
    n = X.shape[0]
    votes = np.zeros((n, 10), dtype=np.int32)
    for tnodes in trees:
        for i in range(n):
            idx = 0
            while True:
                nd = tnodes[idx]
                if nd[0]:
                    votes[i, nd[1]] += 1
                    break
                if X[i, nd[1]] <= nd[2]:
                    idx = nd[3]
                else:
                    idx = nd[4]
    return (votes.argmax(1) == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C, d = card["classes"], card["input"]["bytes"]
    n = len(X)
    split = int(n * 0.8)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    best_acc, best_cfg, best_trees = -1, None, None
    for n_trees in (3, 5, 7):
        for max_depth in (4, 5, 6, 7):
            for max_feats in (8, 16, 32):
                rng = np.random.default_rng(SEED)
                trees = []
                for t in range(n_trees):
                    idx = rng.integers(0, len(X_tr), len(X_tr))
                    trng = np.random.default_rng(SEED + t + 1)
                    tree = build_tree(X_tr[idx], y_tr[idx], max_depth, max_feats, trng)
                    trees.append(tree_to_flat(tree))
                acc = eval_forest(trees, X_va, y_va)
                if acc > best_acc:
                    best_acc, best_cfg, best_trees = acc, (n_trees, max_depth, max_feats), trees

    n_trees, max_depth, max_feats = best_cfg
    rng = np.random.default_rng(SEED)
    final = []
    for t in range(n_trees):
        idx = rng.integers(0, n, n)
        trng = np.random.default_rng(SEED + t + 1)
        tree = build_tree(X[idx], y[idx], max_depth, max_feats, trng)
        final.append(tree_to_flat(tree))

    out = struct.pack("<5i", 0x464F5253, n_trees, max_depth, C, d)
    for tnodes in final:
        out += struct.pack("<i", len(tnodes))
        for nd in tnodes:
            out += struct.pack("<5i", *nd)
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx=0, N=atol(argv[3]), pos=0;
    int32_t *hd = (int32_t*)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x464F5253) { fprintf(stderr, "bad weights\n"); return 3; }
    int n_trees=hd[1], max_depth=hd[2], C=hd[3], d=hd[4];
    if (C<1||C>256||n_trees<1||n_trees>100||max_depth<1||max_depth>20||d!=64) {
        fprintf(stderr, "bad weights params\n"); return 3;
    }
    int32_t *p = hd+5;
    long rem = (nw-20)/4;
    int32_t **trees = malloc(n_trees*sizeof(int32_t*));
    int32_t *tsz = malloc(n_trees*sizeof(int32_t));
    for (int t=0;t<n_trees;t++) {
        if (rem<1) { fprintf(stderr,"truncated weights\n"); return 3; }
        int32_t nn = *p++; rem--;
        if (rem < nn*5) { fprintf(stderr,"truncated weights\n"); return 3; }
        tsz[t]=nn; trees[t]=p; p+=nn*5; rem-=nn*5;
    }
    if (rem!=0) { fprintf(stderr,"extra weights data\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    int *votes = calloc(C, sizeof(int));
    for (long i=0;i<N;i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k!=d) { fprintf(stderr,"item %ld is %u bytes, not %d\n",i,k,d); return 3; }
        memset(votes,0,C*sizeof(int));
        for (int t=0;t<n_trees;t++) {
            int32_t *nd = trees[t];
            int idx=0;
            while (1) {
                if (nd[idx*5+0]) { votes[nd[idx*5+1]]++; break; }
                int f=nd[idx*5+1], th=nd[idx*5+2], l=nd[idx*5+3], r=nd[idx*5+4];
                idx = (xi[f] <= th) ? l : r;
            }
        }
        int best=0;
        for (int c=1;c<C;c++) if (votes[c]>votes[best]) best=c;
        put_rec(stdout, (unsigned char*)&best, 1);
    }
    return 0;
}
```