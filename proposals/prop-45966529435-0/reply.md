```hypothesis
We propose "label-fm", a factorization machine with a low-rank latent dimension (K=8) that models all pairwise feature interactions via dot products of per-value embeddings. The Adult dataset has many categorical features where interactions (e.g., education × occupation) are known to be predictive; a linear model cannot capture these. FM adds only O(d·K) parameters and O(d·K) inference cost, so it should beat label-linear (0.8524) while staying well below label-mlp16's cost (2325 Ir/item). If dev accuracy does not exceed 0.8524 or inference cost exceeds 2500 Ir/item, the family is not worthwhile.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Trains a binary FM with Adam on the tuning split (last 20% of train.in),
then retrains on all data. Exports weights as:
  magic(0x464D4C31), d, K, C=2, bias, linear_weights[d*256], latent_vectors[d*256*K] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Tuning grid (will be evaluated on last 20% of train.in)
K_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3]
L2_CHOICES = [1e-4, 1e-5]
TUNE_EPOCHS = 10
FINAL_EPOCHS = 50
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

def forward(X, bias, W_lin, V, K):
    """X: (B, d) uint8. Returns logits (B,)"""
    B, d = X.shape
    # linear part
    lin = bias + np.sum(W_lin[np.arange(d), X.T], axis=0)  # (B,)
    # interaction part
    sum_v = np.zeros((B, K), dtype=np.float32)
    sum_v_sq = np.zeros((B, K), dtype=np.float32)
    for j in range(d):
        v = V[j, X[:, j]]          # (B, K)
        sum_v += v
        sum_v_sq += v * v
    inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
    return lin + inter

def train_model(X, y, K, lr, L2, epochs, batch_size, rng):
    n, d = X.shape
    # parameters
    bias = np.float32(0.0)
    W_lin = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_W = np.zeros_like(W_lin); v_W = np.zeros_like(W_lin)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s+batch_size]
            xb = X[idx]; yb = y[idx]
            B = len(idx)
            # forward
            lin = bias + np.sum(W_lin[np.arange(d), xb.T], axis=0)
            sum_v = np.zeros((B, K), dtype=np.float32)
            sum_v_sq = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                v = V[j, xb[:, j]]
                sum_v += v
                sum_v_sq += v * v
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
            logits = lin + inter
            p = sigmoid(logits)
            # gradient of BCE wrt logits
            dz = (p - yb).astype(np.float32)  # (B,)
            # gradients
            db = np.sum(dz)
            # linear weights gradient
            dW = np.zeros_like(W_lin)
            for j in range(d):
                np.add.at(dW[j], xb[:, j], dz)
            # V gradient
            dV = np.zeros_like(V)
            dS = dz[:, None] * sum_v          # (B, K)
            dS_sq = -0.5 * dz[:, None]        # (B, K)
            for j in range(d):
                v = V[j, xb[:, j]]            # (B, K)
                grad_v = dS + 2.0 * v * dS_sq
                np.add.at(dV[j], xb[:, j], grad_v)
            # L2 regularization
            dW += 2.0 * L2 * W_lin
            dV += 2.0 * L2 * V
            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * db
            v_bias = beta2 * v_bias + (1 - beta2) * (db * db)
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # W_lin
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * (dW * dW)
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W_lin -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * (dV * dV)
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
    return bias, W_lin, V

def accuracy(X, y, bias, W_lin, V, K):
    logits = forward(X, bias, W_lin, V, K)
    pred = (logits > 0).astype(np.int64)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20%
    n_tune = n // 5
    X_train, X_val = X[:-n_tune], X[-n_tune:]
    y_train, y_val = y[:-n_tune], y[-n_tune:]

    best_acc = -1.0
    best_cfg = None
    best_params = None
    rng = np.random.default_rng(SEED)

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for L2 in L2_CHOICES:
                # deterministic init per config
                cfg_rng = np.random.default_rng(SEED + hash((K, lr, L2)) % 2**32)
                bias, W_lin, V = train_model(X_train, y_train, K, lr, L2, TUNE_EPOCHS, BATCH, cfg_rng)
                acc = accuracy(X_val, y_val, bias, W_lin, V, K)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (K, lr, L2)
                    best_params = (bias, W_lin, V)
    print(f"Best config: K={best_cfg[0]}, lr={best_cfg[1]}, L2={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    K, lr, L2 = best_cfg
    final_rng = np.random.default_rng(SEED)
    bias, W_lin, V = train_model(X, y, K, lr, L2, FINAL_EPOCHS, BATCH, final_rng)

    # Write weights.bin
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=14); output is one byte class index (0 or 1). */
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 16) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *W_lin = p + 1;
    const float *V = W_lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float lin = bias;
        float sum_v[16] = {0};
        float sum_v_sq[16] = {0};
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            lin += W_lin[(long)j * 256 + v];
            const float *Vj = V + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float vu = Vj[u];
                sum_v[u] += vu;
                sum_v_sq[u] += vu * vu;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) {
            inter += sum_v[u] * sum_v[u] - sum_v_sq[u];
        }
        inter *= 0.5f;
        float logit = lin + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```