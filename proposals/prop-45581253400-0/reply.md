```hypothesis
Family: label-fm (Factorization Machine). 
Why: FMs model pairwise feature interactions via low-rank embeddings, which are crucial for tabular data with mixed categorical/ordinal features. They add only O(d*k) parameters over linear models, keeping inference fast. 
Failure signal: If dev accuracy <= 0.8524 (label-linear) or inference cost > 2x label-linear, the trade-off is not justified.
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only. Uses last 20% of train.in for hyperparameter tuning,
then trains on all data. Exports: magic 0x464D0001, d, k, C, bias, linear[d*256], embed[d*256*k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

SEED = 1
TUNE_FRAC = 0.2
GRID = {
    'k': [4, 8, 16],
    'lr': [0.03, 0.1],
    'l2': [1e-4, 1e-3],
    'epochs': [30, 50],
    'batch': [256],
}

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict(X, bias, W, V, k):
    """X: (n, d) uint8, returns prob class 1"""
    n, d = X.shape
    # linear term
    lin = bias + np.sum(W[np.arange(d), X.T], axis=0)  # W is (d, 256)
    # FM term: 0.5 * sum_f ( (sum_i v_{i,f})^2 - sum_i v_{i,f}^2 )
    # V is (d, 256, k)
    # For each sample, gather embeddings: (n, d, k)
    emb = V[np.arange(d), X.T]  # (d, n, k) -> transpose to (n, d, k)
    emb = emb.transpose(1, 0, 2)
    sum_emb = np.sum(emb, axis=1)          # (n, k)
    sum_emb_sq = np.sum(emb * emb, axis=1) # (n, k)
    fm = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
    logits = lin + fm
    return sigmoid(logits)

def accuracy(X, y, bias, W, V, k):
    probs = predict(X, bias, W, V, k)
    preds = (probs > 0.5).astype(np.int64)
    return np.mean(preds == y)

def train_fm(X, y, k, lr, l2, epochs, batch, rng):
    n, d = X.shape
    # init
    bias = 0.0
    W = np.zeros((d, 256), np.float32)
    V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = len(idx)
            # forward
            # linear
            lin = bias + np.sum(W[np.arange(d), xb.T], axis=0)
            # FM
            emb = V[np.arange(d), xb.T].transpose(1, 0, 2)  # (b, d, k)
            sum_emb = np.sum(emb, axis=1)
            sum_emb_sq = np.sum(emb * emb, axis=1)
            fm = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
            logits = lin + fm
            probs = sigmoid(logits)
            # gradient of logistic loss: (probs - y) / b
            grad = (probs - yb) / b  # (b,)
            # bias grad
            g_bias = np.sum(grad)
            # linear grad: for each feature j, value v, add grad[i] to W[j, v]
            g_W = np.zeros_like(W)
            for j in range(d):
                np.add.at(g_W[j], xb[:, j], grad)
            # V grad: for each feature j, value v, for each factor f:
            # dFM/dv_{j,v,f} = grad[i] * (sum_emb[i,f] - v_{j,v,f})
            g_V = np.zeros_like(V)
            for j in range(d):
                vals = xb[:, j]
                for f in range(k):
                    # sum_emb[i,f] - V[j, vals[i], f]
                    diff = sum_emb[:, f] - V[j, vals, f]
                    np.add.at(g_V[j, :, f], vals, grad * diff)
            # Adam updates
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * (g_bias + l2 * bias)
            v_bias = beta2 * v_bias + (1 - beta2) * (g_bias ** 2)
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # W
            m_W = beta1 * m_W + (1 - beta1) * (g_W + l2 * W)
            v_W = beta2 * v_W + (1 - beta2) * (g_W ** 2)
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * (g_V + l2 * V)
            v_V = beta2 * v_V + (1 - beta2) * (g_V ** 2)
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
    return bias, W, V

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X_all.shape[1]
    C = card["classes"]
    n = len(y_all)
    split = int(n * (1 - TUNE_FRAC))
    X_tr, X_va = X_all[:split], X_all[split:]
    y_tr, y_va = y_all[:split], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_model = None

    for k in GRID['k']:
        for lr in GRID['lr']:
            for l2 in GRID['l2']:
                for epochs in GRID['epochs']:
                    for batch in GRID['batch']:
                        bias, W, V = train_fm(X_tr, y_tr, k, lr, l2, epochs, batch, rng)
                        acc = accuracy(X_va, y_va, bias, W, V, k)
                        if acc > best_acc:
                            best_acc = acc
                            best_params = (k, lr, l2, epochs, batch)
                            best_model = (bias, W, V)
    print(f"Best val acc: {best_acc:.4f}, params: {best_params}", file=sys.stderr)

    # retrain on full data with best params
    k, lr, l2, epochs, batch = best_params
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    bias, W, V = train_fm(X_all, y_all, k, lr, l2, epochs, batch, rng)

    # write weights
    magic = 0x464D0001
    out = struct.pack("<4I", magic, d, k, C)
    out += struct.pack("<f", bias)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x464D0001) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0 || k > 64) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + 4 + (size_t)d * 256 * 4 + (size_t)d * 256 * k * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *p = (float *)(wb + 16);
    float bias = *p++;
    float *W = p; p += d * 256;
    float *V = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(k * sizeof(float));
    float *sum_emb_sq = malloc(k * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float logit = bias;
        // linear term
        for (int j = 0; j < d; j++) logit += W[j * 256 + xi[j]];
        // FM term
        memset(sum_emb, 0, k * sizeof(float));
        memset(sum_emb_sq, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *v = V + (j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float val = v[f];
                sum_emb[f] += val;
                sum_emb_sq[f] += val * val;
            }
        }
        for (int f = 0; f < k; f++) {
            logit += 0.5f * (sum_emb[f] * sum_emb[f] - sum_emb_sq[f]);
        }
        unsigned char pred = (logit > 0.0f) ? 1 : 0; // sigmoid(logit) > 0.5 <=> logit > 0
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```