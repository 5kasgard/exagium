```hypothesis
Family: label-poly2 (polynomial logistic regression degree 2).
Why: Optdigits pixels have strong local correlations (strokes); pairwise interactions capture shape primitives better than a linear model. The feature space (1 + 64 + 2080 = 2145) is small enough for fast inference and the training set (2867 samples) is sufficient with L2 regularization.
Failure criterion: If dev accuracy does not exceed label-linear (0.9613) by at least 0.5 percentage points (i.e. < 0.9663), the hypothesis is not supported.
```
```python
"""Polynomial degree-2 logistic regression for fixed-size byte items (ordinal only).
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
L2_CANDIDATES = [1e-4, 1e-3, 1e-2, 1e-1]  # tuned on last 20% of train.in
MAGIC = 0x504F4C32  # "POL2"


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_features(X, T):
    """X: (n, d) uint8, T: (d, 256) float32 normalized lookup.
    Returns Z: (n, 1 + d + d*(d+1)//2) float32 with bias, linear, quadratic."""
    n, d = X.shape
    num_quad = d * (d + 1) // 2
    num_feat = 1 + d + num_quad
    Z = np.empty((n, num_feat), dtype=np.float32)
    Z[:, 0] = 1.0  # bias
    # linear
    for j in range(d):
        Z[:, 1 + j] = T[j, X[:, j]]
    # quadratic
    idx = 1 + d
    for i in range(d):
        Ti = T[i, X[:, i]]
        for j in range(i, d):
            Z[:, idx] = Ti * T[j, X[:, j]]
            idx += 1
    return Z


def train_logreg(Z, y, C, L2, epochs, batch, lr, seed):
    n, k = Z.shape
    rng = np.random.default_rng(seed)
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            G = (P / len(yy)).astype(np.float32)
            grad_W = z.T @ G + L2 * W
            grad_b = G.sum(axis=0)
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
    S = Z @ W + b
    pred = S.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = len(X)
    val_size = n // 5
    train_idx = slice(0, n - val_size)
    val_idx = slice(n - val_size, n)

    best_L2 = L2_CANDIDATES[0]
    best_acc = -1.0
    # Hyperparameter search on tuning split
    for L2 in L2_CANDIDATES:
        # Normalization on train split only
        X_tr = X[train_idx]
        mu = X_tr.mean(axis=0).astype(np.float64)
        sd = X_tr.std(axis=0).astype(np.float64)
        sd[sd == 0] = 1.0
        T = ((np.arange(256, dtype=np.float64) - mu[:, None]) / sd[:, None]).astype(np.float32)
        Z_tr = build_features(X_tr, T)
        Z_val = build_features(X[val_idx], T)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
        W, b = train_logreg(Z_tr, y[train_idx], C, L2, epochs, BATCH, LR, SEED)
        acc = evaluate(Z_val, y[val_idx], W, b)
        if acc > best_acc:
            best_acc = acc
            best_L2 = L2

    # Retrain on full data with best L2
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    T = ((np.arange(256, dtype=np.float64) - mu[:, None]) / sd[:, None]).astype(np.float32)
    Z_full = build_features(X, T)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W, b = train_logreg(Z_full, y, C, best_L2, epochs, BATCH, LR, SEED)

    # Combine weights: bias is first row of W? Actually we have separate b.
    # We'll store as single matrix: feature 0 = bias, features 1.. = W rows.
    num_feat = 1 + d + d * (d + 1) // 2
    weights = np.empty((num_feat, C), dtype=np.float32)
    weights[0] = b
    weights[1:] = W

    # Write weights.bin
    out = struct.pack("<4i", MAGIC, d, C, num_feat)
    out += T.astype("<f4").tobytes()
    out += weights.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Polynomial degree-2 logistic regression inference.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class 0..9). */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], num_feat = hd[3];
    int expected_feat = 1 + d + d * (d + 1) / 2;
    if (C < 1 || C > 256 || num_feat != expected_feat) {
        fprintf(stderr, "bad header: d=%d C=%d num_feat=%d expected=%d\n", d, C, num_feat, expected_feat);
        return 3;
    }
    long expected_size = 16L + 4L * d * 256 + 4L * num_feat * C;
    if (nw != expected_size) {
        fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected_size);
        return 3;
    }
    const float *T = (const float *)(hd + 4);
    const float *weights = T + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *scores = malloc(sizeof(float) * C);
    float *norm = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Normalize
        for (int j = 0; j < d; j++) {
            norm[j] = T[(long)j * 256 + xi[j]];
        }
        // Initialize with bias
        for (int c = 0; c < C; c++) scores[c] = weights[c];
        // Linear terms
        for (int j = 0; j < d; j++) {
            const float *w = weights + (long)(1 + j) * C;
            float v = norm[j];
            for (int c = 0; c < C; c++) scores[c] += v * w[c];
        }
        // Quadratic terms
        long feat_idx = 1 + d;
        for (int i1 = 0; i1 < d; i1++) {
            for (int i2 = i1; i2 < d; i2++) {
                const float *w = weights + feat_idx * C;
                float v = norm[i1] * norm[i2];
                for (int c = 0; c < C; c++) scores[c] += v * w[c];
                feat_idx++;
            }
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```