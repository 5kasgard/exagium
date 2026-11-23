```hypothesis
Family: label-fm (Factorization Machine). Adult income classification benefits from feature interactions (e.g., education×occupation). FM captures pairwise interactions with O(d·k) inference, adding only a small constant over linear models. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds 2× label-linear, the approach fails.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Treats all 14 features as categorical with 256 values (0-255).
Learns linear weights per feature value and k-dim embeddings for pairwise interactions.
Trains with Adam on binary cross-entropy. Chooses k and lr on last 20% tuning split,
then retrains on all data. Exports: magic, d, k, C, w0, W_lin[d*256], V[d*256*k] (float32)."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
EPOCHS_GRID = 15
EPOCHS_FINAL = 20
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def predict_logits(X, w0, W_lin, V, d, k):
    n = X.shape[0]
    linear = w0 + np.sum(W_lin[np.arange(d), X.T], axis=0).astype(np.float32)
    sum_v = np.zeros((n, k), dtype=np.float32)
    sum_v2 = np.zeros(n, dtype=np.float32)
    for j in range(d):
        v = V[j, X[:, j], :]
        sum_v += v
        sum_v2 += np.sum(v * v, axis=1)
    interaction = 0.5 * (np.sum(sum_v * sum_v, axis=1) - sum_v2)
    return linear + interaction

def train_fm(X, y, d, k, lr, epochs, seed):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    w0 = np.float32(0.0)
    W_lin = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, k)).astype(np.float32)
    m_w0, v_w0 = 0.0, 0.0
    m_W = np.zeros_like(W_lin); v_W = np.zeros_like(W_lin)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            b = len(idx)
            # forward
            linear = w0 + np.sum(W_lin[np.arange(d), xb.T], axis=0)
            sum_v = np.zeros((b, k), dtype=np.float32)
            sum_v2 = np.zeros(b, dtype=np.float32)
            for j in range(d):
                v = V[j, xb[:, j], :]
                sum_v += v
                sum_v2 += np.sum(v * v, axis=1)
            interaction = 0.5 * (np.sum(sum_v * sum_v, axis=1) - sum_v2)
            logit = linear + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = p - yb
            # gradients
            grad_w0 = np.mean(grad_logit)
            grad_W = np.zeros_like(W_lin)
            grad_V = np.zeros_like(V)
            for j in range(d):
                np.add.at(grad_W[j], xb[:, j], grad_logit / b)
                diff = grad_logit[:, np.newaxis] * (sum_v - V[j, xb[:, j], :])
                np.add.at(grad_V[j], xb[:, j], diff)
            grad_V /= b
            # Adam update
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * grad_w0 * grad_w0
            w0 -= lr * (m_w0 / (1 - beta1**t)) / (np.sqrt(v_w0 / (1 - beta2**t)) + eps)
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            W_lin -= lr * (m_W / (1 - beta1**t)) / (np.sqrt(v_W / (1 - beta2**t)) + eps)
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            V -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)
    return w0, W_lin, V

def accuracy(X, y, w0, W_lin, V, d, k):
    logits = predict_logits(X, w0, W_lin, V, d, k)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    best_acc = -1.0
    best_k, best_lr = 8, 0.01
    for k in (4, 8, 16):
        for lr in (0.005, 0.01, 0.02):
            w0, W_lin, V = train_fm(X_tr, y_tr, d, k, lr, EPOCHS_GRID, SEED + k*10 + int(lr*1000))
            acc = accuracy(X_val, y_val, w0, W_lin, V, d, k)
            if acc > best_acc:
                best_acc, best_k, best_lr = acc, k, lr

    # final training on all data
    w0, W_lin, V = train_fm(X, y, d, best_k, best_lr, EPOCHS_FINAL, SEED)

    # write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, best_k, C)
    w0_bytes = struct.pack("<f", w0)
    W_lin_bytes = W_lin.astype("<f4").tobytes()
    V_bytes = V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + w0_bytes + W_lin_bytes + V_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0 or 1). */
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
    if (C != 2 || k < 1 || k > 16 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    long expected_floats = 1L + (long)d * 256 * (1L + k);
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *W_lin = p + 1;
    const float *V = W_lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[16];
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float logit = w0;
        for (int j = 0; j < d; j++) logit += W_lin[j * 256 + xi[j]];
        for (int u = 0; u < k; u++) sum_v[u] = 0.0f;
        float sum_v2 = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * k;
            for (int u = 0; u < k; u++) {
                float val = v[u];
                sum_v[u] += val;
                sum_v2 += val * val;
            }
        }
        float sum_v_sq = 0.0f;
        for (int u = 0; u < k; u++) sum_v_sq += sum_v[u] * sum_v[u];
        logit += 0.5f * (sum_v_sq - sum_v2);
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```