```hypothesis
Family: label-fm (Factorization Machine). FM models pairwise feature interactions via low-rank latent vectors, which are crucial for tabular data with mixed categorical/ordinal features. It adds only O(d * 256 * k) parameters (k=8) over linear, keeping inference fast (O(d*k) per item). If it fails to beat label-linear on dev accuracy, it suggests interactions are not beneficial or the model overfits the small training set.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each feature value as a linear weight + a k-dim latent vector.
Trains with Adam on logistic loss, L2 regularization.
Tunes k and lr on last 20% of train.in (file order), then retrains on all.
Exports: magic 0x464D3130, d, k, C, w0, w[d][256], v[d][256][k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
L2 = 1e-4
GRID_K = [4, 8, 16]
GRID_LR = [1e-3, 3e-3, 1e-2]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fm_forward(X, w0, w, v, k):
    """X: (n, d) uint8. Returns scores (n,)."""
    n, d = X.shape
    linear = w0 + np.sum(w[np.arange(d), X.T], axis=0)  # (n,)
    sum_v = np.zeros((n, k), dtype=np.float32)
    sum_v2 = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        v_j = v[j, X[:, j]]  # (n, k)
        sum_v += v_j
        sum_v2 += v_j * v_j
    interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
    return linear + interaction

def train_fm(X, y, k, lr, epochs, rng):
    n, d = X.shape
    C = 2
    w0 = 0.0
    w = np.zeros((d, 256), dtype=np.float32)
    v = np.zeros((d, 256, k), dtype=np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_v = np.zeros_like(v); v_v = np.zeros_like(v)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    y_pm1 = 2 * y - 1  # convert 0/1 to -1/1
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y_pm1[idx]
            # forward
            linear = w0 + np.sum(w[np.arange(d), Xb.T], axis=0)
            sum_v = np.zeros((len(idx), k), dtype=np.float32)
            sum_v2 = np.zeros((len(idx), k), dtype=np.float32)
            for j in range(d):
                v_j = v[j, Xb[:, j]]
                sum_v += v_j
                sum_v2 += v_j * v_j
            interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
            scores = linear + interaction
            p = sigmoid(scores)
            grad = (p - y[idx]).astype(np.float32)  # (batch,)
            # gradients
            g_w0 = grad.mean()
            g_w = np.zeros_like(w)
            g_v = np.zeros_like(v)
            for j in range(d):
                # linear gradient
                for val in range(256):
                    mask = (Xb[:, j] == val)
                    if mask.any():
                        g_w[j, val] = grad[mask].mean()
                # interaction gradient
                v_j = v[j, Xb[:, j]]  # (batch, k)
                # sum_v - v_j for each example
                diff = sum_v - v_j  # (batch, k)
                for val in range(256):
                    mask = (Xb[:, j] == val)
                    if mask.any():
                        g_v[j, val] = (grad[mask, None] * diff[mask]).mean(axis=0)
            # L2
            g_w += L2 * w
            g_v += L2 * v
            # Adam update
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= lr * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v
            v -= lr * (m_v / (1 - beta1 ** t)) / (np.sqrt(v_v / (1 - beta2 ** t)) + eps)
    return w0, w, v

def accuracy(X, y, w0, w, v, k):
    scores = fm_forward(X, w0, w, v, k)
    preds = (scores > 0).astype(np.int64)
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_all.shape[0]
    d = card["input"]["bytes"]
    C = card["classes"]
    # split: last 20% for tuning
    split = int(0.8 * n)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_k = GRID_K[0]
    best_lr = GRID_LR[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / split))
    for k in GRID_K:
        for lr in GRID_LR:
            w0, w, v = train_fm(X_train, y_train, k, lr, epochs, np.random.default_rng(SEED))
            acc = accuracy(X_val, y_val, w0, w, v, k)
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_lr = lr
    # retrain on all data
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    w0, w, v = train_fm(X_all, y_all, best_k, best_lr, epochs_full, np.random.default_rng(SEED))
    # save weights
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, best_k, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k <= 0 || d <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 16 + 4 + (size_t)d * 256 * 4 + (size_t)d * 256 * k * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(wb + 16);
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + (size_t)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v2 = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float linear = w0;
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            linear += w[(size_t)j * 256 + val];
            const float *v_j = v + ((size_t)j * 256 + val) * k;
            for (int f = 0; f < k; f++) {
                float vv = v_j[f];
                sum_v[f] += vv;
                sum_v2[f] += vv * vv;
            }
        }
        float interaction = 0.0f;
        for (int f = 0; f < k; f++) interaction += sum_v[f] * sum_v[f] - sum_v2[f];
        interaction *= 0.5f;
        float score = linear + interaction;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```