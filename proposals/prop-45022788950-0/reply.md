```hypothesis
Propose a Factorization Machine (FM) with latent dimension k=4. FM models pairwise feature interactions via low-rank embeddings, which is effective for tabular data with mixed categorical/ordinal features. It should surpass label-linear (0.8524) by capturing interactions while staying faster than label-mlp16 (2325 Ir/item). If dev accuracy ≤ 0.8524 or inference cost > 2325 Ir/item, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D3130, d, k, C; float32 w0, W[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter search space (tuned on last 20% of train.in)
K_CHOICES = [4]
LR_CHOICES = [0.01, 0.005, 0.02]
EPOCHS_BASE = 600000
BATCH = 64
MAX_EPOCHS = 100
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_train_val(X, y, val_frac=0.2):
    n = X.shape[0]
    val_size = int(n * val_frac)
    return X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fm_forward(X, w0, W, V, k):
    """X: (n, d) uint8. Returns scores (n,)."""
    n, d = X.shape
    # Linear part
    lin = w0 + np.sum(W[np.arange(d), X.T], axis=0)  # (n,)
    # Interaction part
    inter = np.zeros(n, dtype=np.float32)
    for i in range(k):
        v_sum = np.sum(V[np.arange(d), X.T, i], axis=0)  # (n,)
        v_sq_sum = np.sum(V[np.arange(d), X.T, i] ** 2, axis=0)
        inter += 0.5 * (v_sum * v_sum - v_sq_sum)
    return lin + inter


def fm_backward(X, y, scores, w0, W, V, k, l2=0.0):
    """Compute gradients for a batch. Returns (g_w0, g_W, g_V)."""
    n, d = X.shape
    probs = sigmoid(scores)
    diff = (probs - y).astype(np.float32)  # (n,)
    # Linear gradients
    g_w0 = diff.mean()
    g_W = np.zeros_like(W)
    for f in range(d):
        np.add.at(g_W[f], X[:, f], diff)  # sum over batch
    g_W /= n
    # Interaction gradients
    g_V = np.zeros_like(V)
    for i in range(k):
        v_sum = np.sum(V[np.arange(d), X.T, i], axis=0)  # (n,)
        for f in range(d):
            v_f = V[f, X[:, f], i]  # (n,)
            # derivative of 0.5*(sum^2 - sum_sq) w.r.t v_f = v_sum - v_f
            grad = diff * (v_sum - v_f)
            np.add.at(g_V[f, :, i], X[:, f], grad)
    g_V /= n
    # L2 regularization
    if l2:
        g_W += l2 * W
        g_V += l2 * V
    return g_w0, g_W, g_V


def train_fm(X, y, k, lr, epochs, batch_size, seed, l2=1e-4):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    # Initialize
    w0 = 0.0
    W = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, k))).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            Xb, yb = X[idx], y[idx]
            scores = fm_forward(Xb, w0, W, V, k)
            g_w0, g_W, g_V = fm_backward(Xb, yb, scores, w0, W, V, k, l2)
            t += 1
            # Adam update
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            m_W = beta1 * m_W + (1 - beta1) * g_W
            v_W = beta2 * v_W + (1 - beta2) * g_W * g_W
            W -= lr * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + eps)
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)
    return w0, W, V


def eval_accuracy(X, y, w0, W, V, k):
    scores = fm_forward(X, w0, W, V, k)
    preds = (scores > 0).astype(np.int64)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    # Split for tuning
    X_tr, y_tr, X_val, y_val = split_train_val(X, y, 0.2)
    best_acc = -1.0
    best_params = None
    best_hp = None
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            epochs = min(MAX_EPOCHS, math.ceil(EPOCHS_BASE / X_tr.shape[0]))
            w0, W, V = train_fm(X_tr, y_tr, k, lr, epochs, BATCH, SEED)
            acc = eval_accuracy(X_val, y_val, w0, W, V, k)
            if acc > best_acc:
                best_acc = acc
                best_params = (w0, W, V)
                best_hp = (k, lr, epochs)
    # Retrain on full data with best hyperparameters
    k, lr, epochs = best_hp
    epochs_full = min(MAX_EPOCHS, math.ceil(EPOCHS_BASE / X.shape[0]))
    w0, W, V = train_fm(X, y, k, lr, epochs_full, BATCH, SEED)
    # Write weights.bin
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, k, C)
    out += struct.pack("<f", w0)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1). */
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
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3130) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k <= 0 || k > 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4 + (long)d * 256 * 4 + (long)d * 256 * k * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float score = w0;
        // Linear part
        for (int f = 0; f < d; f++) score += W[(long)f * 256 + xi[f]];
        // Interaction part
        for (int i_lat = 0; i_lat < k; i_lat++) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int f = 0; f < d; f++) {
                float v = V[((long)f * 256 + xi[f]) * k + i_lat];
                sum += v;
                sum_sq += v * v;
            }
            score += 0.5f * (sum * sum - sum_sq);
        }
        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```