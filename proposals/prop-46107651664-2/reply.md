```hypothesis
We propose a two-hidden-layer MLP with position-specific embeddings (family "label-emb-mlp2").
The linear model (label-linear) ignores interactions between bytes; a single hidden layer (label-mlp16) was too small to capture them.
A deeper network with tuned hidden sizes (e.g., 64 and 32) and ReLU activations can model higher-order interactions while keeping inference fast via fused embedding lookup.
We expect >0.35 accuracy on dev. If accuracy ≤0.3375 (label-linear) or inference cost exceeds label-linear by >2×, the hypothesis fails.
```

```python
"""Two-hidden-layer MLP with position-specific embeddings for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 12345), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np


def read_records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def forward(X, T, b1, W2, b2, W3, b3):
    """X: (n, d) uint8 -> logits (n, C) float32"""
    n, d = X.shape
    H1 = b1.shape[0]
    acc1 = np.zeros((n, H1), dtype=np.float32) + b1
    for j in range(d):
        acc1 += T[j, X[:, j], :]
    h1 = np.maximum(acc1, 0)
    acc2 = h1 @ W2 + b2
    h2 = np.maximum(acc2, 0)
    logits = h2 @ W3 + b3
    return logits


def train_model(X, y, d, C, cfg, rng):
    H1, H2 = cfg["H1"], cfg["H2"]
    lr, L2, batch, epochs = cfg["lr"], cfg["L2"], cfg["batch"], cfg["epochs"]
    # He init for ReLU layers
    T = rng.normal(0, 0.01, size=(d, 256, H1)).astype(np.float32)
    b1 = np.zeros(H1, dtype=np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H1), size=(H1, H2)).astype(np.float32)
    b2 = np.zeros(H2, dtype=np.float32)
    W3 = rng.normal(0, math.sqrt(2.0 / H2), size=(H2, C)).astype(np.float32)
    b3 = np.zeros(C, dtype=np.float32)

    params = [T, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    n = X.shape[0]

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # forward
            acc1 = np.zeros((len(yb), H1), dtype=np.float32) + b1
            for j in range(d):
                acc1 += T[j, xb[:, j], :]
            h1 = np.maximum(acc1, 0)
            acc2 = h1 @ W2 + b2
            h2 = np.maximum(acc2, 0)
            logits = h2 @ W3 + b3
            # softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)
            # backward
            grad_W3 = h2.T @ grad_logits + L2 * W3
            grad_b3 = grad_logits.sum(axis=0)
            grad_h2 = grad_logits @ W3.T
            grad_h2[acc2 <= 0] = 0
            grad_W2 = h1.T @ grad_h2 + L2 * W2
            grad_b2 = grad_h2.sum(axis=0)
            grad_h1 = grad_h2 @ W2.T
            grad_h1[acc1 <= 0] = 0
            grad_b1 = grad_h1.sum(axis=0)
            grad_T = np.zeros_like(T)
            for j in range(d):
                np.add.at(grad_T[j], xb[:, j], grad_h1)
            grads = [grad_T, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return T, b1, W2, b2, W3, b3


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    d, C = card["input"]["bytes"], card["classes"]
    n = X.shape[0]
    n_train = int(0.8 * n)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    configs = [
        {"H1": 64, "H2": 32, "lr": 3e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
        {"H1": 64, "H2": 32, "lr": 1e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
        {"H1": 128, "H2": 64, "lr": 3e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
        {"H1": 128, "H2": 64, "lr": 1e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
        {"H1": 32, "H2": 16, "lr": 3e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
        {"H1": 32, "H2": 16, "lr": 1e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
    ]

    best_acc, best_cfg, best_params = -1.0, None, None
    for cfg in configs:
        rng = np.random.default_rng(12345)
        T, b1, W2, b2, W3, b3 = train_model(X_tr, y_tr, d, C, cfg, rng)
        logits = forward(X_val, T, b1, W2, b2, W3, b3)
        acc = (logits.argmax(axis=1) == y_val).mean()
        if acc > best_acc:
            best_acc, best_cfg = acc, cfg
            best_params = (T.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())

    # Retrain on full data with best config
    rng = np.random.default_rng(12345)
    T, b1, W2, b2, W3, b3 = train_model(X, y, d, C, best_cfg, rng)

    # Save weights: magic LBE2 (0x4C424532), d, H1, H2, C; then float32 arrays
    magic = 0x4C424532
    H1, H2 = best_cfg["H1"], best_cfg["H2"]
    header = struct.pack("<5i", magic, d, H1, H2, C)
    out = header
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Two-hidden-layer MLP with position-specific embeddings for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
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
    if (nw < 20 || hd[0] != 0x4C424532) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || H1 < 1 || H2 < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T   = (const float *)(hd + 5);
    const float *b1  = T + (long)d * 256 * H1;
    const float *W2  = b1 + H1;
    const float *b2  = W2 + (long)H1 * H2;
    const float *W3  = b2 + H2;
    const float *b3  = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *h1   = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *h2   = malloc(sizeof(float) * H2);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc1, b1, H1 * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) acc1[u] += t[u];
        }
        for (int u = 0; u < H1; u++) h1[u] = acc1[u] > 0.0f ? acc1[u] : 0.0f;

        for (int v = 0; v < H2; v++) {
            float sum = b2[v];
            for (int u = 0; u < H1; u++) sum += h1[u] * W2[(long)u * H2 + v];
            acc2[v] = sum;
        }
        for (int v = 0; v < H2; v++) h2[v] = acc2[v] > 0.0f ? acc2[v] : 0.0f;

        for (int c = 0; c < C; c++) {
            float sum = b3[c];
            for (int v = 0; v < H2; v++) sum += h2[v] * W3[(long)v * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```