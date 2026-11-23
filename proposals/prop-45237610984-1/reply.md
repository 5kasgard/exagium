```hypothesis
We propose a degree-2 polynomial logistic regression (label-poly2) that adds all pairwise pixel products to the original 64 features. On optdigits, digit shapes are defined by local pixel correlations, so quadratic interactions should capture stroke junctions and curves better than a linear model. If dev accuracy does not exceed label-linear's 96.1%, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-3, 1e-2]
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def poly_features(X):
    """X: (n, 64) uint8 -> (n, 2080) float32: 64 linear + 2016 pairwise (i<j)"""
    n, d = X.shape
    D = d + d * (d - 1) // 2
    Z = np.empty((n, D), dtype=np.float32)
    Z[:, :d] = X.astype(np.float32)
    idx = d
    for i in range(d):
        xi = X[:, i].astype(np.float32)
        for j in range(i + 1, d):
            Z[:, idx] = xi * X[:, j].astype(np.float32)
            idx += 1
    return Z


def standardize(Z, mean=None, std=None):
    if mean is None:
        mean = Z.mean(axis=0)
    if std is None:
        std = Z.std(axis=0)
        std[std == 0] = 1.0
    Z = (Z - mean) / std
    return Z, mean, std


def softmax_grad(logits, y):
    # logits: (batch, C), y: (batch,)
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def train_lr(Z, y, C, lr, l2, rng):
    n, k = Z.shape
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)  # (batch, C)
            grad_W = z.T @ g + l2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b


def evaluate(Z, y, W, b):
    logits = Z @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    D = d + d * (d - 1) // 2

    # Build polynomial features
    Z = poly_features(X)
    n = Z.shape[0]

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    idx_train = np.arange(split)
    idx_tune = np.arange(split, n)

    Z_train, Z_tune = Z[idx_train], Z[idx_tune]
    y_train, y_tune = y[idx_train], y[idx_tune]

    # Standardize on training portion
    Z_train, mean, std = standardize(Z_train)
    Z_tune = (Z_tune - mean) / std

    best_acc = -1.0
    best_lr = LR_GRID[0]
    best_l2 = L2_GRID[0]
    best_W = None
    best_b = None

    for lr in LR_GRID:
        for l2 in L2_GRID:
            rng = np.random.default_rng(SEED)
            W, b = train_lr(Z_train, y_train, C, lr, l2, rng)
            acc = evaluate(Z_tune, y_tune, W, b)
            if acc > best_acc:
                best_acc = acc
                best_lr, best_l2 = lr, l2
                best_W, best_b = W.copy(), b.copy()

    # Retrain on all data with best hyperparameters
    Z_all, mean_all, std_all = standardize(Z)
    rng = np.random.default_rng(SEED)
    W_final, b_final = train_lr(Z_all, y, C, best_lr, best_l2, rng)

    # Export weights
    # Magic: 0x504F4C32 ('POL2')
    # int32: magic, d, D, C
    # float32: mean[D], std[D], W[D*C], b[C]
    header = struct.pack("<4i", 0x504F4C32, d, D, C)
    body = mean_all.astype("<f4").tobytes() + std_all.astype("<f4").tobytes()
    body += W_final.astype("<f4").tobytes() + b_final.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16 + 4L * (2L * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + D;
    const float *W = std + D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * D);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* linear features */
        for (int j = 0; j < d; j++) feat[j] = (float)xi[j];
        /* pairwise products (i < j) */
        int idx = d;
        for (int ii = 0; ii < d; ii++) {
            float xvi = (float)xi[ii];
            for (int jj = ii + 1; jj < d; jj++) {
                feat[idx++] = xvi * (float)xi[jj];
            }
        }

        /* standardize */
        for (int j = 0; j < D; j++) feat[j] = (feat[j] - mean[j]) / std[j];

        /* linear classification */
        for (int c = 0; c < C; c++) score[c] = b[c];
        for (int j = 0; j < D; j++) {
            float fj = feat[j];
            const float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) score[c] += fj * wj[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```