```hypothesis
Family: label-fm (Factorization Machine). Captures pairwise feature interactions via low-rank embeddings, well-suited for tabular data with mixed ordinal/categorical features. The 14 byte-wide features (quantile bins + category indices) map naturally to FM's one-hot-per-field encoding. Expected to beat label-linear (0.8524) by modeling interactions (e.g., education×occupation) at modest cost (k=8 → ~115KB weights, ~2k Ir/item). Failure: if dev accuracy ≤0.8524 or instructions >3000 Ir/item.
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Field-aware FM on 14 byte features (0-255). Binary classification via logit>0.
Tunes k,lr on last 20% of train, then trains on all data. Exports: magic 0x464D3130, d, k, C=1; float32 w0, W[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03]
TUNE_EPOCHS = 5
FINAL_EPOCHS = 20
BATCH = 256
L2 = 1e-4
MAGIC = 0x464D3130  # 'FM10'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_loop(rng, X, y, k, lr, epochs, w0, W, V, m_w0, v_w0, m_W, v_W, m_V, v_V, t):
    n, d = X.shape
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bsz = len(yb)
            # Forward
            linear = w0 + np.sum(W[np.arange(d)[:, None], xb.T], axis=0)  # (bsz,)
            sum_v = np.sum(V[np.arange(d)[:, None], xb.T], axis=0).transpose(1, 0)  # (bsz, k)
            sum_v2 = np.sum(V[np.arange(d)[:, None], xb.T] ** 2, axis=0).transpose(1, 0)
            interaction = 0.5 * (np.sum(sum_v ** 2, axis=1) - np.sum(sum_v2, axis=1))
            logit = linear + interaction
            # BCE gradient
            sig = sigmoid(logit)
            dlogit = sig - yb  # (bsz,)
            # Gradients
            gw0 = np.mean(dlogit)
            dW = np.zeros_like(W)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(dW[j], vals, dlogit)
            dW /= bsz
            dV = np.zeros_like(V)
            for j in range(d):
                vals = xb[:, j]
                diff = sum_v - V[j, vals]  # (bsz, k)
                grad = dlogit[:, None] * diff  # (bsz, k)
                for i in range(bsz):
                    dV[j, vals[i]] += grad[i]
            dV /= bsz
            # L2
            dW += L2 * W
            dV += L2 * V
            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # W
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * dW * dW
            W -= lr * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * dV * dV
            V -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)
    return w0, W, V, m_w0, v_w0, m_W, v_W, m_V, v_V, t

def evaluate(X, y, w0, W, V):
    n, d = X.shape
    linear = w0 + np.sum(W[np.arange(d)[:, None], X.T], axis=0)
    sum_v = np.sum(V[np.arange(d)[:, None], X.T], axis=0).transpose(1, 0)
    sum_v2 = np.sum(V[np.arange(d)[:, None], X.T] ** 2, axis=0).transpose(1, 0)
    interaction = 0.5 * (np.sum(sum_v ** 2, axis=1) - np.sum(sum_v2, axis=1))
    logit = linear + interaction
    pred = (logit > 0).astype(np.int64)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    assert d == 14
    # Tuning split: last 20%
    tune_size = n // 5
    train_size = n - tune_size
    X_tr, y_tr = X[:train_size], y[:train_size]
    X_tune, y_tune = X[train_size:], y[train_size:]
    
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_k = 8
    best_lr = 0.01
    
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            # Initialize
            w0 = 0.0
            W = np.zeros((d, 256), np.float32)
            V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
            m_w0, v_w0 = 0.0, 0.0
            m_W, v_W = np.zeros_like(W), np.zeros_like(W)
            m_V, v_V = np.zeros_like(V), np.zeros_like(V)
            t = 0
            # Train on tuning train split
            w0, W, V, m_w0, v_w0, m_W, v_W, m_V, v_V, t = train_loop(
                rng, X_tr, y_tr, k, lr, TUNE_EPOCHS, w0, W, V, m_w0, v_w0, m_W, v_W, m_V, v_V, t)
            acc = evaluate(X_tune, y_tune, w0, W, V)
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_lr = lr
    
    # Retrain on full data with best hyperparameters
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    k = best_k
    lr = best_lr
    w0 = 0.0
    W = np.zeros((d, 256), np.float32)
    V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    m_w0, v_w0 = 0.0, 0.0
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    w0, W, V, *_ = train_loop(rng, X, y, k, lr, FINAL_EPOCHS, w0, W, V, m_w0, v_w0, m_W, v_W, m_V, v_V, t)
    
    # Write weights
    # Header: magic, d, k, C=1
    header = struct.pack("<4i", MAGIC, d, k, 1)
    # Data: w0 (float32), W (d*256 float32), V (d*256*k float32)
    w0_arr = np.array([w0], dtype=np.float32)
    W_flat = W.astype("<f4").ravel()
    V_flat = V.astype("<f4").ravel()
    out = header + w0_arr.tobytes() + W_flat.tobytes() + V_flat.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output record: 1 byte class (0/1). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || k <= 0 || k > 64 || C != 1) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * 4 + (long)d * 256 * k * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if (len != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float linear = w0;
        float sum_v[64] = {0};
        float sum_v2[64] = {0};
        for (int j = 0; j < d; j++) {
            int val = xi[j];
            linear += W[(long)j * 256 + val];
            const float *v = V + ((long)j * 256 + val) * k;
            for (int u = 0; u < k; u++) {
                float vu = v[u];
                sum_v[u] += vu;
                sum_v2[u] += vu * vu;
            }
        }
        float interaction = 0.0f;
        for (int u = 0; u < k; u++) {
            interaction += sum_v[u] * sum_v[u] - sum_v2[u];
        }
        interaction *= 0.5f;
        float logit = linear + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```