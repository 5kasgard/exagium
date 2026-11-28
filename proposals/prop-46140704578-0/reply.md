```hypothesis
Family: label-fm (Factorization Machine for tabular byte features).
Why: Adult has 14 features (mix of ordinal binned into 255 quantiles and categorical indices). A degree-2 FM captures pairwise interactions with O(d*k) parameters (d=14, k=8), adding only ~32K floats vs linear's ~3.5K. This should improve accuracy over label-linear (0.8524) while keeping inference cost low (O(d*k) ~112 multiply-adds per item). If dev accuracy does not exceed label-linear by at least 0.005 or inference cost exceeds 2x label-linear, the family does not justify its complexity.
```

```python
"""Factorization Machine for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Uses last 20% of train for tuning k in {4,8}, then trains on all.
Exported as: int32 magic 0x464D4C31, d, k, C; float32 w0, W[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR_GRID = [0.01, 0.03]
L2 = 1e-4
BATCH = 256
SAMPLES = 600000
MAX_EPOCHS = 20
K_GRID = [4, 8]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(X, w0, W, V, k):
    """X: (n, d) uint8 indices. Returns scores (n,)."""
    n, d = X.shape
    # linear part
    lin = w0 + np.sum(W[np.arange(d), X.T], axis=0)  # (n,)
    # interaction part
    # gather embeddings: (n, d, k)
    emb = V[np.arange(d), X.T].transpose(1, 0, 2)
    sum_v = emb.sum(axis=1)          # (n, k)
    sum_v_sq = (emb ** 2).sum(axis=1) # (n, k)
    inter = 0.5 * ((sum_v ** 2).sum(axis=1) - sum_v_sq.sum(axis=1))
    return lin + inter

def train_fm(X, y, k, lr, rng, tune_idx=None):
    n, d = X.shape
    C = 2
    # init
    w0 = np.float32(0.0)
    W = rng.uniform(-0.01, 0.01, (d, 256)).astype(np.float32)
    V = rng.uniform(-0.01, 0.01, (d, 256, k)).astype(np.float32)
    # Adam state
    params = [w0, W, V]
    shapes = [(), (d, 256), (d, 256, k)]
    m = [np.zeros(s, np.float32) for s in shapes]
    v = [np.zeros(s, np.float32) for s in shapes]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    best_loss = float('inf')
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yi = X[idx], y[idx]
            # forward
            scores = fm_forward(xi, w0, W, V, k)
            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-scores))
            g = (p - yi).astype(np.float32)  # (batch,)
            # gradients
            batch_size = len(yi)
            gw0 = g.mean()
            gW = np.zeros_like(W)
            gV = np.zeros_like(V)
            # linear gradient
            for j in range(d):
                vals = xi[:, j]
                np.add.at(gW[j], vals, g)
            gW /= batch_size
            gW += L2 * W
            # interaction gradient
            # For each example: sum_v = sum_u V[u, xu], then derivative w.r.t V[i, xi] is (sum_v - V[i, xi]) * g
            emb = V[np.arange(d), xi.T].transpose(1, 0, 2)  # (batch, d, k)
            sum_v = emb.sum(axis=1)  # (batch, k)
            # dL/dV[i, xi] = g * (sum_v - V[i, xi])
            for j in range(d):
                v_ij = V[j, xi[:, j]]  # (batch, k)
                grad = (g[:, None] * (sum_v - v_ij)) / batch_size
                np.add.at(gV[j], xi[:, j], grad)
            gV += L2 * V
            # Adam update
            t += 1
            grads = [gw0, gW, gV]
            for i, (p, gr, mm, vv) in enumerate(zip(params, grads, m, v)):
                mm *= beta1; mm += (1 - beta1) * gr
                vv *= beta2; vv += (1 - beta2) * gr * gr
                m_hat = mm / (1 - beta1 ** t)
                v_hat = vv / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # tune evaluation
        if tune_idx is not None:
            Xi, yi = X[tune_idx], y[tune_idx]
            scores = fm_forward(Xi, w0, W, V, k)
            loss = -np.mean(yi * np.log(1/(1+np.exp(-scores))) + (1-yi) * np.log(1-1/(1+np.exp(-scores))))
            if loss < best_loss:
                best_loss = loss
                best_params = (w0.copy(), W.copy(), V.copy())
    if best_params is not None:
        w0, W, V = best_params
    return w0, W, V

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    # split last 20% for tuning
    split = int(n * 0.8)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_idx, tune_idx = idx[:split], idx[split:]
    # grid search
    best_cfg = None
    best_acc = -1
    for k in K_GRID:
        for lr in LR_GRID:
            w0, W, V = train_fm(X[train_idx], y[train_idx], k, lr, rng, tune_idx)
            scores = fm_forward(X[tune_idx], w0, W, V, k)
            preds = (scores > 0).astype(np.int64)
            acc = (preds == y[tune_idx]).mean()
            if acc > best_acc:
                best_acc = acc
                best_cfg = (k, lr, w0, W, V)
    k, lr, w0, W, V = best_cfg
    # retrain on all data
    w0, W, V = train_fm(X, y, k, lr, rng, tune_idx=None)
    # save
    out = struct.pack("<4i", 0x464D4C31, d, k, 2)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class index (0 or 1). */
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
    if (C != 2 || d < 1 || k < 1 || k > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score = w0;
        memset(sum_v, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            int idx = xi[j];
            score += W[(long)j * 256 + idx];
            const float *v = V + ((long)j * 256 + idx) * k;
            for (int u = 0; u < k; u++) sum_v[u] += v[u];
        }
        float inter = 0.0f;
        for (int u = 0; u < k; u++) inter += sum_v[u] * sum_v[u];
        for (int j = 0; j < d; j++) {
            int idx = xi[j];
            const float *v = V + ((long)j * 256 + idx) * k;
            for (int u = 0; u < k; u++) inter -= v[u] * v[u];
        }
        score += 0.5f * inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```