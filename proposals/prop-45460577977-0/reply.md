```hypothesis
Family: label-fm (Factorization Machine with per-field embeddings). Treats each of the 14 features as a categorical field with 256 values, learns a linear weight and a k-dimensional embedding per field-value. Pairwise interactions captured via dot products of embeddings, computed in O(k*d) using the sum-square trick. Should beat label-linear by modeling feature interactions; if dev accuracy <= 0.8524 or inference cost exceeds label-linear significantly, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Learns per-field embeddings and linear weights. Hyperparameters chosen on last 20% of train.
Exports: magic 0x464D3100, d, k, C; float32 w0, w[d][256], V[d][256][k] (row-major)."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning
K_CHOICES = [2, 4, 8]
LR_CHOICES = [0.03, 0.1, 0.3]
L2_CHOICES = [1e-4, 1e-3]
EPOCHS = 20
BATCH = 64
SAMPLES_PER_EPOCH = 600000  # same as label-linear


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_predict(X, w0, w, V, k):
    """Compute logits for batch X (n, d)."""
    n, d = X.shape
    linear = w0 + np.sum(w[np.arange(d), X], axis=1)
    # interaction: 0.5 * sum_k ( (sum_j V[j, x_j, k])^2 - sum_j V[j, x_j, k]^2 )
    V_sel = V[np.arange(d), X]  # (n, d, k)
    sum_V = V_sel.sum(axis=1)   # (n, k)
    sum_sq_V = (V_sel ** 2).sum(axis=1)
    inter = 0.5 * np.sum(sum_V ** 2 - sum_sq_V, axis=1)
    return linear + inter


def train_fm(X, y, k, lr, l2, epochs, batch, rng, X_val=None, y_val=None):
    n, d = X.shape
    # Initialize parameters
    w0 = 0.0
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    best_val_acc = -1.0
    best_params = None
    steps_per_epoch = max(1, math.ceil(SAMPLES_PER_EPOCH / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # Forward
            V_sel = V[np.arange(d), xb]  # (b, d, k)
            sum_V = V_sel.sum(axis=1)    # (b, k)
            sum_sq_V = (V_sel ** 2).sum(axis=1)
            linear = w0 + np.sum(w[np.arange(d), xb], axis=1)
            inter = 0.5 * np.sum(sum_V ** 2 - sum_sq_V, axis=1)
            logits = linear + inter
            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yb).astype(np.float32)  # (b,)
            # Gradients
            g_w0 = grad.mean()
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)
            # For each sample in batch, accumulate gradients
            for i in range(len(idx)):
                gi = grad[i]
                xi = xb[i]
                # Linear weights
                g_w[np.arange(d), xi] += gi / batch
                # Embeddings: dV_jk = gi * (sum_V_ik - V_jk)
                # sum_V_i is (k,), V_sel[i] is (d, k)
                g_V[np.arange(d), xi] += (gi * (sum_V[i] - V_sel[i])) / batch
            # L2 regularization
            g_w += l2 * w
            g_V += l2 * V
            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
        # Validation check
        if X_val is not None:
            val_logits = fm_predict(X_val, w0, w, V, k)
            val_pred = (val_logits > 0).astype(np.int64)
            val_acc = (val_pred == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (w0, w.copy(), V.copy())
    if best_params is not None:
        w0, w, V = best_params
    return w0, w, V


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                cfg_rng = np.random.default_rng(SEED)  # reset for each config
                w0, w, V = train_fm(X_train, y_train, k, lr, l2, EPOCHS, BATCH, cfg_rng, X_val, y_val)
                val_logits = fm_predict(X_val, w0, w, V, k)
                val_pred = (val_logits > 0).astype(np.int64)
                acc = (val_pred == y_val).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (k, lr, l2)
                    best_params = (w0, w, V)
    k, lr, l2 = best_cfg
    # Retrain on full data
    full_rng = np.random.default_rng(SEED)
    w0, w, V = train_fm(X, y, k, lr, l2, EPOCHS, BATCH, full_rng)
    # Save weights
    magic = 0x464D3100  # "FM1\0"
    out = struct.pack("<4i", magic, d, k, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (0 or 1).
 * Weights format: int32 magic(0x464D3100), d, k, C; float32 w0, w[d][256], V[d][256][k] */
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
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k <= 0 || d <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %zu\n", nw, expected); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * k);
    float *sum_sq_V = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float linear = w0;
        for (int j = 0; j < k; j++) { sum_V[j] = 0.0f; sum_sq_V[j] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += w[(long)j * 256 + v];
            const float *Vjv = V + ((long)j * 256 + v) * k;
            for (int kk = 0; kk < k; kk++) {
                float val = Vjv[kk];
                sum_V[kk] += val;
                sum_sq_V[kk] += val * val;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < k; kk++) {
            inter += sum_V[kk] * sum_V[kk] - sum_sq_V[kk];
        }
        inter *= 0.5f;
        float logit = linear + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_V); free(sum_sq_V);
    return 0;
}
```