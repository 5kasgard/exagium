```hypothesis
Family: label-poly2. A linear model on degree-2 polynomial features (all pairwise products) of the normalized 64 pixel counts. The 8x8 digit images exhibit spatial correlations; pairwise interactions can capture simple shape features (edges, corners) that a purely linear model misses, potentially closing the gap to label-gbdt (97.9%) while remaining faster than tree ensembles. The model adds only O(d^2) multiplications per item (d=64), keeping inference cost low. If dev accuracy does not exceed 96.5% or inference cost exceeds ~20k Ir/item, the approach does not justify its complexity over label-linear.
```

```python
import json
import math
import struct
import sys

import numpy as np

SEED = 1
BATCH = 64
MAX_EPOCHS = 200
SAMPLES = 600000

# Hyperparameter grid for tuning split (last 20% of train.in)
LR_GRID = [0.01, 0.03, 0.1]
L2_GRID = [1e-4, 1e-3, 1e-2]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_poly_features(X_norm):
    """X_norm: (n, 64) normalized floats. Returns (n, 2144) with 64 linear + 2080 pairwise."""
    n, d = X_norm.shape
    # Linear features
    lin = X_norm
    # Pairwise products (including squares) in fixed order: j=0..63, k=j..63
    num_pair = d * (d + 1) // 2
    poly = np.empty((n, d + num_pair), dtype=np.float32)
    poly[:, :d] = lin
    idx = d
    for j in range(d):
        for k in range(j, d):
            poly[:, idx] = lin[:, j] * lin[:, k]
            idx += 1
    return poly

def softmax_cross_entropy_grad(scores, y):
    """scores: (n, C), y: (n,). Returns gradient of loss wrt scores: (n, C)"""
    n = scores.shape[0]
    scores_max = scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n

def train_linear(X, y, C, lr, l2, epochs, batch, rng, verbose=False):
    n, k = X.shape
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            scores = xb @ W + b
            grad_scores = softmax_cross_entropy_grad(scores, yb)
            grad_W = xb.T @ grad_scores + l2 * W
            grad_b = grad_scores.sum(axis=0)
            t += 1
            # Adam updates
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
    return W, b

def eval_acc(X, y, W, b):
    scores = X @ W + b
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_bytes.shape[1]
    assert d == 64 and all(t == "ordinal" for t in card["input"]["types"])

    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning
    n = len(X_bytes)
    val_start = int(0.8 * n)
    train_X_bytes, train_y = X_bytes[:val_start], y[:val_start]
    val_X_bytes, val_y = X_bytes[val_start:], y[val_start:]

    # Compute mean/std on train split only
    train_X = train_X_bytes.astype(np.float64)
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0

    def normalize(X_bytes):
        return ((X_bytes.astype(np.float64) - mean) / std).astype(np.float32)

    train_X_norm = normalize(train_X_bytes)
    val_X_norm = normalize(val_X_bytes)

    # Build polynomial features
    train_X_poly = build_poly_features(train_X_norm)
    val_X_poly = build_poly_features(val_X_norm)
    num_features = train_X_poly.shape[1]  # 2144

    # Grid search on validation
    best_acc = -1.0
    best_params = None
    best_W, best_b = None, None
    for lr in LR_GRID:
        for l2 in L2_GRID:
            W, b = train_linear(train_X_poly, train_y, C, lr, l2, MAX_EPOCHS, BATCH, rng)
            acc = eval_acc(val_X_poly, val_y, W, b)
            if acc > best_acc:
                best_acc = acc
                best_params = (lr, l2)
                best_W, best_b = W.copy(), b.copy()

    # Retrain on full data with best params
    full_X_norm = normalize(X_bytes)
    full_X_poly = build_poly_features(full_X_norm)
    lr, l2 = best_params
    W, b = train_linear(full_X_poly, y, C, lr, l2, MAX_EPOCHS, BATCH, rng)

    # Prepare weights file
    # Magic: 0x504F4C32 ("POL2")
    # int32: d, num_features, C
    # float32: mean[d], std[d]
    # float32: W[num_features * C] (row-major: feature major)
    # float32: b[C]
    magic = 0x504F4C32
    header = struct.pack("<4i", magic, d, num_features, C)
    mean_bytes = mean.astype("<f4").tobytes()
    std_bytes = std.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    out = header + mean_bytes + std_bytes + W_bytes + b_bytes
    open(sys.argv[4], "wb").write(out)

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
    unsigned char *wbuf = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wbuf;
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], num_features = hd[2], C = hd[3];
    if (d != 64 || C < 1 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    // Expected size: header(16) + mean[d]*4 + std[d]*4 + W[num_features*C]*4 + b[C]*4
    long expected = 16 + 4L * (2L * d + (long)num_features * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *W = std + d;
    const float *b = W + (long)num_features * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(sizeof(float) * d);
    float *features = malloc(sizeof(float) * num_features);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize
        for (int j = 0; j < d; j++) {
            xnorm[j] = (xi[j] - mean[j]) / std[j];
        }

        // Build polynomial features: first d linear, then pairwise (j<=k)
        memcpy(features, xnorm, d * sizeof(float));
        int idx = d;
        for (int j = 0; j < d; j++) {
            for (int k = j; k < d; k++) {
                features[idx++] = xnorm[j] * xnorm[k];
            }
        }
        if (idx != num_features) { fprintf(stderr, "feature count mismatch\n"); return 3; }

        // Compute scores
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < num_features; f++) {
            float xf = features[f];
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += xf * wf[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```