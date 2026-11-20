```hypothesis
Family: label-fm (factorization machine with low-rank pairwise interactions). 
Why: Optdigits pixels have strong local correlations (strokes); a degree-2 FM captures pairwise interactions with O(d*K) parameters (K=8) and O(d*K*C) inference ops, far cheaper than an MLP hidden layer. 
Failure signal: dev accuracy ≤ label-linear (0.9613) or inference cost > label-linear (5058 Ir/item).
```

```python
"""Factorization Machine (degree 2) for ordinal byte features.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Searches K, lr, L2 on last 20% tuning split,
then retrains on full train. Exports: magic, d, C, K, mean[d], std[d], bias[C], linear[d*C], V[C*d*K] (float32 LE)."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split
K_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3]
L2_CHOICES = [1e-4, 1e-3]
SEED = 1
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


def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,)
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)


def fm_forward(X, bias, linear, V, K):
    # X: (B, d), bias: (C,), linear: (d, C), V: (C, d, K)
    B, d = X.shape
    C = bias.shape[0]
    # linear part
    scores = X @ linear + bias  # (B, C)
    # interaction part
    for c in range(C):
        Vc = V[c]  # (d, K)
        # S = X @ Vc  (B, K)
        S = X @ Vc
        # S2 = (X^2) @ (Vc^2)
        S2 = (X * X) @ (Vc * Vc)
        inter = 0.5 * ((S * S).sum(axis=1) - S2.sum(axis=1))
        scores[:, c] += inter
    return scores


def fm_backward(X, grad_scores, linear, V, K, L2):
    # X: (B, d), grad_scores: (B, C)
    B, d = X.shape
    C = grad_scores.shape[1]
    grad_linear = np.zeros_like(linear)
    grad_bias = np.zeros(C)
    grad_V = np.zeros_like(V)

    grad_bias[:] = grad_scores.sum(axis=0)
    grad_linear[:] = X.T @ grad_scores + L2 * linear

    for c in range(C):
        gc = grad_scores[:, c]  # (B,)
        Vc = V[c]  # (d, K)
        S = X @ Vc  # (B, K)
        # d_inter/d_S = S
        # d_inter/d_Vc = X.T @ (S * gc[:, None]) - (X * X).T @ (Vc * gc[:, None])
        # Actually: inter = 0.5 * sum_k (S_k^2 - S2_k)
        # d inter / d Vc_{i,k} = sum_b gc_b * ( S_{b,k} * X_{b,i} - (X_{b,i}^2) * Vc_{i,k} )
        # So grad_Vc = X.T @ (gc[:, None] * S) - (X*X).T @ (gc[:, None] * Vc)
        grad_Vc = X.T @ (gc[:, None] * S) - (X * X).T @ (gc[:, None] * Vc)
        grad_Vc += L2 * Vc
        grad_V[c] = grad_Vc
    return grad_bias, grad_linear, grad_V


def train_fm(X, y, K, lr, L2, epochs, batch, rng):
    n, d = X.shape
    C = 10
    # init
    bias = np.zeros(C, np.float32)
    linear = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    V = (rng.uniform(-1, 1, (C, d, K)) * math.sqrt(6.0 / (d * K))).astype(np.float32)

    params = [bias, linear, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]
            yb = y[idx]
            scores = fm_forward(xb, bias, linear, V, K)
            g = softmax_cross_entropy_grad(scores, yb)  # (batch, C)
            g_bias, g_linear, g_V = fm_backward(xb, g, linear, V, K, L2)
            grads = [g_bias, g_linear, g_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return bias, linear, V


def eval_acc(X, y, bias, linear, V, K):
    scores = fm_forward(X, bias, linear, V, K)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert C == 10

    n = len(y)
    tune_n = n // 5
    train_n = n - tune_n
    X_train, y_train = X[:train_n], y[:train_n]
    X_tune, y_tune = X[train_n:], y[train_n:]

    mu, sd = zscore_fit(X_train)
    X_train_z = zscore_apply(X_train, mu, sd).astype(np.float32)
    X_tune_z = zscore_apply(X_tune, mu, sd).astype(np.float32)

    best_acc = -1.0
    best_cfg = None
    best_params = None

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for L2 in L2_CHOICES:
                rng = np.random.default_rng(SEED)
                epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / train_n))
                bias, linear, V = train_fm(X_train_z, y_train, K, lr, L2, epochs, BATCH, rng)
                acc = eval_acc(X_tune_z, y_tune, bias, linear, V, K)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (K, lr, L2)
                    best_params = (bias.copy(), linear.copy(), V.copy())

    # Retrain on full data with best config
    K, lr, L2 = best_cfg
    mu_full, sd_full = zscore_fit(X)
    X_full_z = zscore_apply(X, mu_full, sd_full).astype(np.float32)
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    bias, linear, V = train_fm(X_full_z, y, K, lr, L2, epochs, BATCH, rng)

    # Export weights
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, C, K)
    out += mu_full.astype("<f4").tobytes()
    out += sd_full.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record: d bytes (pixels). Output: one byte class (argmax). */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }

    float *ptr = (float *)(hd + 4);
    float *mean = ptr; ptr += d;
    float *std  = ptr; ptr += d;
    float *bias = ptr; ptr += C;
    float *linear = ptr; ptr += (long)d * C;
    float *V = ptr;  /* size C * d * K */

    long expected = 16 + 4L * (2L * d + C + (long)d * C + (long)C * d * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(sizeof(float) * d);
    float *S = malloc(sizeof(float) * K);
    float *S2 = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++)
            xnorm[j] = (xi[j] - mean[j]) / std[j];

        for (int c = 0; c < C; c++) {
            float sc = bias[c];
            for (int j = 0; j < d; j++)
                sc += xnorm[j] * linear[(long)j * C + c];
            scores[c] = sc;
        }

        for (int c = 0; c < C; c++) {
            const float *Vc = V + (long)c * d * K;
            memset(S, 0, sizeof(float) * K);
            memset(S2, 0, sizeof(float) * K);
            for (int j = 0; j < d; j++) {
                float xj = xnorm[j];
                const float *v = Vc + (long)j * K;
                for (int kk = 0; kk < K; kk++) {
                    float val = xj * v[kk];
                    S[kk] += val;
                    S2[kk] += val * val;
                }
            }
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++)
                inter += S[kk] * S[kk] - S2[kk];
            scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```