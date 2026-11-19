```hypothesis
Propose a tuned MLP family (label-mlp-tuned) that selects hidden layer size H on the tuning split (last 20% of train.in). The linear model ignores interactions between the 16 context positions; a hidden ReLU layer can capture such interactions. The existing label-mlp16 uses H=16 which may be too small. We will search H in {32,64,128} with fixed LR=3e-3, L2=1e-4, batch=64, epochs=20 on the tuning split, then retrain on all data. If validation accuracy does not exceed label-linear (0.3375), the family does not help.
```

```python
"""Tuned MLP family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Chooses hidden size H on the tuning split (last 20% of train.in), then trains on all data.
Exports weights as: magic 0x4D4C5031 ('MLP1'), d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, os, struct, sys

import numpy as np

# Force single-threaded BLAS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS_TUNE = 20
EPOCHS_FULL = 30
H_CANDIDATES = [32, 64, 128]  # hidden sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def one_hot_encode(X, d):
    """X: (n, d) uint8 -> Z: (n, d*256) float32"""
    n = X.shape[0]
    Z = np.zeros((n, d * 256), dtype=np.float32)
    for j in range(d):
        Z[:, j * 256:(j + 1) * 256][np.arange(n), X[:, j]] = 1.0
    return Z


def init_params(k, H, C, rng):
    # He initialization for ReLU
    W1 = (rng.standard_normal((k, H)) * math.sqrt(2.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.standard_normal((H, C)) * math.sqrt(2.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return W1, b1, W2, b2


def train_epoch(Z, y, W1, b1, W2, b2, batch_size, lr, l2, rng, state, t):
    n = Z.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        z, yy = Z[idx], y[idx]
        # forward
        a = z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        # softmax gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        g = probs / len(yy)  # (batch, C)
        # backward
        gh = g @ W2.T * (a > 0)  # (batch, H)
        grad_W1 = z.T @ gh + l2 * W1
        grad_b1 = gh.sum(axis=0)
        grad_W2 = h.T @ g + l2 * W2
        grad_b2 = g.sum(axis=0)
        # Adam update
        t += 1
        for p, gr, (m, v) in zip([W1, b1, W2, b2],
                                 [grad_W1, grad_b1, grad_W2, grad_b2],
                                 state):
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return t


def evaluate(Z, y, W1, b1, W2, b2):
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def train_model(Z_train, y_train, Z_val, y_val, H, C, seed, epochs):
    rng = np.random.default_rng(seed)
    k = Z_train.shape[1]
    W1, b1, W2, b2 = init_params(k, H, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in [W1, b1, W2, b2]]
    t = 0
    for _ in range(epochs):
        t = train_epoch(Z_train, y_train, W1, b1, W2, b2, BATCH, LR, L2, rng, state, t)
    val_acc = evaluate(Z_val, y_val, W1, b1, W2, b2)
    return W1, b1, W2, b2, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # split: last 20% for tuning
    val_size = n // 5
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    # encode
    Z_train = one_hot_encode(X_train, d)
    Z_val = one_hot_encode(X_val, d)
    # grid search over H
    best_H = H_CANDIDATES[0]
    best_acc = -1.0
    for H in H_CANDIDATES:
        _, _, _, _, acc = train_model(Z_train, y_train, Z_val, y_val, H, C, SEED, EPOCHS_TUNE)
        if acc > best_acc:
            best_acc = acc
            best_H = H
    # retrain on full data with best_H
    Z_full = one_hot_encode(X, d)
    W1, b1, W2, b2, _ = train_model(Z_full, y, Z_val, y_val, best_H, C, SEED, EPOCHS_FULL)
    # prepare tables for export: T[d][256][H]
    T = W1.reshape(d, 256, best_H).astype(np.float32)
    # pack weights
    magic = 0x4D4C5031  # 'MLP1'
    out = struct.pack("<4i", magic, d, best_H, C)
    out += T.tobytes()
    out += b1.astype(np.float32).tobytes()
    out += W2.astype(np.float32).tobytes()
    out += b2.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Tuned MLP family (label-mlp-tuned) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: magic 0x4D4C5031, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H < 1) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int c = 0; c < C; c++) s[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = acc[h] > 0.0f ? acc[h] : 0.0f;
            for (int c = 0; c < C; c++) s[c] += hv * W2[(long)h * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```