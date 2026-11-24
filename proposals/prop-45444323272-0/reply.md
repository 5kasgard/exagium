```hypothesis
The label-embedding family learns a small dense vector for every possible feature value (256 per column) and classifies with a linear layer on the concatenated embeddings. This captures non-linear ordinal effects (unlike label-linear's z-scoring) and compresses high-cardinality categoricals (unlike one-hot). On Adult, where numeric columns are quantile-binned and categoricals have many levels, this should beat label-linear's 0.8524. If dev accuracy ≤ 0.8524, the extra capacity does not help or overfits the tuning split.
```

```python
"""label-embedding: per-value embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weight format: magic 0x454D4244, d, H=0, C, then for each column:
  ordinal: 256 float32 (embedding size 1)
  categorical: 256*E float32 (E=4)
Then final weight matrix (total_emb_dim * C) and bias (C)."""
import json, math, struct, sys

import numpy as np

SEED = 1
E_CAT = 4          # embedding size for categorical columns
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
TUNE_FRAC = 0.2    # last 20% for tuning

# small grid for tuning split
GRID = [
    {"lr": 3e-3, "l2": 1e-4},
    {"lr": 1e-3, "l2": 1e-4},
    {"lr": 3e-3, "l2": 1e-3},
    {"lr": 1e-3, "l2": 1e-3},
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_embeddings(card, X):
    """Return list of embedding matrices, one per column, and total emb dim."""
    embs = []
    total_dim = 0
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            # 256 x 1
            embs.append(np.zeros((256, 1), np.float32))
            total_dim += 1
        else:
            # 256 x E_CAT
            embs.append(np.zeros((256, E_CAT), np.float32))
            total_dim += E_CAT
    return embs, total_dim

def forward(embs, X, W, b):
    """X: (n, d) uint8. Returns logits (n, C)."""
    n, d = X.shape
    C = W.shape[1]
    out = np.zeros((n, C), np.float32)
    for j in range(d):
        e = embs[j]                 # (256, dim_j)
        idx = X[:, j]               # (n,)
        out += e[idx] @ W[total_dim - e.shape[1]:total_dim]  # wrong indexing, fix below
    # Actually we need to accumulate embeddings then multiply by W once.
    # Let's do it efficiently:
    # We'll concatenate embeddings per example.
    pass

def forward_concat(embs, X, W, b):
    n, d = X.shape
    C = W.shape[1]
    # Build concatenated embedding matrix (n, total_dim)
    Z = np.zeros((n, W.shape[0]), np.float32)
    col = 0
    for j in range(d):
        e = embs[j]
        dim = e.shape[1]
        Z[:, col:col+dim] = e[X[:, j]]
        col += dim
    return Z @ W + b

def softmax_cross_entropy_grad(logits, y):
    n = logits.shape[0]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n

def evaluate(embs, X, y, W, b):
    logits = forward_concat(embs, X, W, b)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_one_config(embs, W, b, X_tr, y_tr, X_va, y_va, lr, l2, epochs):
    rng = np.random.default_rng(SEED)
    n = X_tr.shape[0]
    # Adam state
    params = [emb for emb in embs] + [W, b]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # forward
            logits = forward_concat(embs, xb, W, b)
            # gradient
            g = softmax_cross_entropy_grad(logits, yb)          # (batch, C)
            # gradient wrt W and b
            # need Z for this batch
            Z = np.zeros((len(idx), W.shape[0]), np.float32)
            col = 0
            for j in range(xb.shape[1]):
                e = embs[j]
                dim = e.shape[1]
                Z[:, col:col+dim] = e[xb[:, j]]
                col += dim
            grad_W = Z.T @ g + l2 * W
            grad_b = g.sum(axis=0)
            # gradient wrt embeddings
            grad_embs = []
            col = 0
            for j in range(xb.shape[1]):
                e = embs[j]
                dim = e.shape[1]
                # dL/dZ_j = g @ W_j.T  (W_j is slice of W for this column)
                W_j = W[col:col+dim]
                dZ = g @ W_j.T                     # (batch, dim)
                # accumulate per value
                ge = np.zeros_like(e)
                np.add.at(ge, xb[:, j], dZ)
                ge += l2 * e
                grad_embs.append(ge)
                col += dim
            # Adam update
            all_grads = grad_embs + [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, all_grads)):
                m, v = states[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return evaluate(embs, X_va, y_va, W, b)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    # split last 20% for tuning
    split = int(n * (1 - TUNE_FRAC))
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    # build embedding structure once (shapes fixed)
    embs_template, total_dim = build_embeddings(card, X)
    # grid search
    best_acc = -1
    best_config = None
    best_embs = None
    best_W = None
    best_b = None

    for cfg in GRID:
        # fresh init
        rng = np.random.default_rng(SEED)
        embs = [rng.normal(0, 0.01, e.shape).astype(np.float32) for e in embs_template]
        W = rng.normal(0, math.sqrt(6.0 / (total_dim + C)), (total_dim, C)).astype(np.float32)
        b = np.zeros(C, np.float32)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
        acc = train_one_config(embs, W, b, X_tr, y_tr, X_va, y_va,
                               cfg["lr"], cfg["l2"], epochs)
        if acc > best_acc:
            best_acc = acc
            best_config = cfg
            best_embs = [e.copy() for e in embs]
            best_W = W.copy()
            best_b = b.copy()

    # retrain on full data with best config
    rng = np.random.default_rng(SEED)
    embs = [rng.normal(0, 0.01, e.shape).astype(np.float32) for e in embs_template]
    W = rng.normal(0, math.sqrt(6.0 / (total_dim + C)), (total_dim, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    train_one_config(embs, W, b, X, y, X_va, y_va,  # validation not used but needed for func signature
                     best_config["lr"], best_config["l2"], epochs)

    # write weights
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<4i", magic, d, 0, C)
    for e in embs:
        out += e.astype("<f4").tobytes()
    out += best_W.astype("<f4").tobytes()
    out += best_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embedding inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index. */
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
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const int E_CAT = 4;
    int total_dim = 0;
    const float *col_emb[14];   // max d=14
    int col_dim[14];
    for (int j = 0; j < d; j++) {
        col_emb[j] = p;
        if (j == 0 || j == 2 || j == 4 || j == 10 || j == 11 || j == 12) { // ordinal indices from card
            col_dim[j] = 1;
            p += 256 * 1;
            total_dim += 1;
        } else {
            col_dim[j] = E_CAT;
            p += 256 * E_CAT;
            total_dim += E_CAT;
        }
    }
    const float *W = p; p += (long)total_dim * C;
    const float *b = p; p += C;
    long expected = 16 + 4L * ((long)total_dim * C + C);
    for (int j = 0; j < d; j++)
        expected += 4L * 256 * col_dim[j];
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = calloc(total_dim, sizeof(float));
    float *scores = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(acc, 0, total_dim * sizeof(float));
        int off = 0;
        for (int j = 0; j < d; j++) {
            int dim = col_dim[j];
            const float *e = col_emb[j] + xi[j] * dim;
            for (int v = 0; v < dim; v++) acc[off + v] = e[v];
            off += dim;
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int v = 0; v < total_dim; v++) sum += acc[v] * W[(long)v * C + c];
            scores[c] += sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```