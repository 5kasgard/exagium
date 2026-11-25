```hypothesis
Factorization Machine (FM) family: models all pairwise feature interactions via low-rank embeddings (k=4..8). On Adult's 14 categorical/ordinal features this captures non-linearities that logistic regression misses, while inference remains a few hundred FLOPs/item. Expect to beat label-linear (0.8524) and approach label-mlp16 (0.8591) at lower CPU cost. If dev accuracy ≤0.8524 or instructions/item ≥ label-linear, the family does not meet its goal.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes every feature as a categorical with 256 values (ordinal are quantile bins).
Learns w0, per-feature-value weights w, and per-feature-value embeddings V (dim k).
Trained with Adam on logistic loss, L2 regularization. Hyperparameters chosen on
last 20% of train.in (file order), then retrained on all train."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid (tuned on the tuning split)
K_GRID = [4, 8]
LR_GRID = [0.01, 0.005]
L2_GRID = [1e-4, 1e-5]
EPOCHS_GRID = [10, 20]
BATCH = 64
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fm_forward(X_idx, w0, w, V, d, k):
    """X_idx: (n, d) int32 feature indices. Returns scores (n,)."""
    n = X_idx.shape[0]
    # linear part
    lin = w0 + w[X_idx].sum(axis=1)  # (n,)
    # interaction part via FM trick
    V_batch = V[X_idx]                # (n, d, k)
    sum_V = V_batch.sum(axis=1)       # (n, k)
    sum_V_sq = (V_batch ** 2).sum(axis=1)  # (n, k)
    inter = 0.5 * ((sum_V ** 2 - sum_V_sq).sum(axis=1))  # (n,)
    return lin + inter

def fm_gradients(X_idx, y, scores, w, V, d, k, l2):
    """Compute gradients for w0, w, V. y in {-1, 1}."""
    n = X_idx.shape[0]
    # dL/dscore = -y * sigmoid(-y * score)
    g = -y * sigmoid(-y * scores)  # (n,)
    # gradient for w0
    gw0 = g.sum()
    # gradient for w
    gw = np.zeros_like(w)
    np.add.at(gw, X_idx.ravel(), np.repeat(g, d))
    # gradient for V
    V_batch = V[X_idx]                # (n, d, k)
    sum_V = V_batch.sum(axis=1)       # (n, k)
    # d(inter)/dV_i = sum_V - V_i  for each feature i
    g_expanded = g[:, None, None]     # (n, 1, 1)
    sum_V_expanded = sum_V[:, None, :]  # (n, 1, k)
    gV_batch = g_expanded * (sum_V_expanded - V_batch)  # (n, d, k)
    gV = np.zeros_like(V)
    np.add.at(gV, X_idx.ravel(), gV_batch.reshape(-1, k))
    # L2 regularization
    gw += 2 * l2 * w
    gV += 2 * l2 * V
    return gw0, gw, gV

def adam_step(params, grads, states, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    """params, grads, states are lists of arrays."""
    for p, gr, (m, v) in zip(params, grads, states):
        m *= beta1; m += (1 - beta1) * gr
        v *= beta2; v += (1 - beta2) * (gr * gr)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)

def evaluate(X_idx, y, w0, w, V, d, k):
    scores = fm_forward(X_idx, w0, w, V, d, k)
    pred = (scores > 0).astype(np.int64)
    return (pred == y).mean()

def train_fm(X_tr, y_tr, X_val, y_val, d, k, lr, l2, epochs, rng):
    n = X_tr.shape[0]
    # initialize
    w0 = 0.0
    w = np.zeros(d * 256, dtype=np.float32)
    V = (rng.normal(0, 0.01, size=(d * 256, k))).astype(np.float32)
    # Adam states
    params = [np.array([w0]), w, V]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    y_tr_pm = 2 * y_tr - 1  # convert to {-1, 1}
    t = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X_tr[idx]
            yb = y_tr_pm[idx]
            scores = fm_forward(Xb, w0, w, V, d, k)
            gw0, gw, gV = fm_gradients(Xb, yb, scores, w, V, d, k, l2)
            t += 1
            grads = [np.array([gw0]), gw, gV]
            adam_step(params, grads, states, lr, t)
            w0 = params[0][0]
    val_acc = evaluate(X_val, y_val, w0, w, V, d, k)
    return val_acc, w0, w, V

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    n = X.shape[0]
    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_cfg = None
    best_model = None
    # grid search
    for k in K_GRID:
        for lr in LR_GRID:
            for l2 in L2_GRID:
                for epochs in EPOCHS_GRID:
                    val_acc, w0, w, V = train_fm(X_tr, y_tr, X_val, y_val, d, k, lr, l2, epochs, rng)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_cfg = (k, lr, l2, epochs)
                        best_model = (w0, w, V)
    print(f"Best config: k={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, epochs={best_cfg[3]}, val_acc={best_acc:.4f}", file=sys.stderr)
    # retrain on full data with best config
    k, lr, l2, epochs = best_cfg
    _, w0, w, V = train_fm(X, y, X, y, d, k, lr, l2, epochs, rng)  # validation not used, just to reuse function
    # save weights
    magic = 0x464D4C31  # 'FML1'
    C = card["classes"]
    header = struct.pack("<4i", magic, d, k, C)
    # w0 (float32), w (d*256), V (d*256*k)
    body = np.array([w0], dtype=np.float32).tobytes()
    body += w.astype("<f4").tobytes()
    body += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (feature indices 0..255). Output: one byte class (0/1). */
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k < 1 || k > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float linear = w0;
        float sum_V[16] = {0};
        float sum_V_sq[16] = {0};
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            linear += w[idx];
            const float *Vi = V + (long)idx * k;
            for (int kk = 0; kk < k; kk++) {
                float v = Vi[kk];
                sum_V[kk] += v;
                sum_V_sq[kk] += v * v;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < k; kk++) {
            inter += sum_V[kk] * sum_V[kk] - sum_V_sq[kk];
        }
        inter *= 0.5f;
        float score = linear + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```