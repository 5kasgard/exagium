```hypothesis
Family: label-poly2 (quadratic polynomial on z-scored pixels).
Why: The 8x8 optdigits pixels are ordinal counts 0..16; a linear model (label-linear) ignores pixel interactions. Adding all pairwise products (i≤j) gives 64+2080=2144 features, capturing simple stroke curvature/shape at modest cost. A multinomial logistic regression on this expansion with L2 regularization should beat linear (0.9613) and approach MLP/GBDT accuracy while keeping inference fast (≈23k FLOPs/item, no branching).
Failure: If dev accuracy ≤0.9613 or inference cost exceeds label-mlp16 (≈8k Ir/item), the family does not improve the trade-off.
```

```python
"""label-poly2: quadratic polynomial (all pairwise products) on z-scored ordinal features.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Hyper-parameters chosen on last 20% of train.in (tuning split), then retrained on all train.in.
Weights format: magic=0x504F4C32, d, Q, C; float32 mean[d], std[d], W_lin[d*C], W_quad[Q*C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Tuning grid (small, fast)
L2_GRID = [1e-4, 1e-3, 1e-2]
LR_GRID = [1e-3, 3e-3, 1e-2]
BATCH = 64
TUNE_EPOCHS = 20
FULL_EPOCHS = 100
SAMPLES = 600000

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def expand(X, mean, std):
    """X: (n, d) z-scored -> (n, d + d*(d+1)//2) with linear then quadratic (i<=j)."""
    n, d = X.shape
    Z = (X - mean) / std
    Q = d * (d + 1) // 2
    Zq = np.empty((n, d + Q), dtype=np.float32)
    Zq[:, :d] = Z
    col = d
    for i in range(d):
        for j in range(i, d):
            Zq[:, col] = Z[:, i] * Z[:, j]
            col += 1
    return Zq

def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits: (P - one_hot) / n."""
    P = np.exp(logits - logits.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return P / len(y)

def train_one(Z, y, C, lr, l2, epochs, rng):
    n, k = Z.shape
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)          # (batch, C)
            gradW = z.T @ g + l2 * W               # (k, C)
            gradb = g.sum(0)                       # (C,)
            t += 1
            # Adam
            for param, grad, m, v in [(W, gradW, mW, vW), (b, gradb, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b

def accuracy(Z, y, W, b):
    logits = Z @ W + b
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    Q = d * (d + 1) // 2
    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    # Preprocess on training split
    mean = X_tr.mean(axis=0).astype(np.float32)
    std = X_tr.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0

    Z_tr = expand(X_tr, mean, std)
    Z_va = expand(X_va, mean, std)

    # Grid search
    best_acc, best_lr, best_l2 = -1, LR_GRID[0], L2_GRID[0]
    for l2 in L2_GRID:
        for lr in LR_GRID:
            W, b = train_one(Z_tr, y_tr, C, lr, l2, TUNE_EPOCHS, np.random.default_rng(SEED))
            acc = accuracy(Z_va, y_va, W, b)
            if acc > best_acc:
                best_acc, best_lr, best_l2 = acc, lr, l2

    # Retrain on full data
    mean_full = X.mean(axis=0).astype(np.float32)
    std_full = X.std(axis=0).astype(np.float32)
    std_full[std_full == 0] = 1.0
    Z_full = expand(X, mean_full, std_full)
    W, b = train_one(Z_full, y, C, best_lr, best_l2, FULL_EPOCHS, np.random.default_rng(SEED))

    # Pack weights
    magic = 0x504F4C32  # 'POL2'
    header = struct.pack("<4i", magic, d, Q, C)
    body = (mean_full.tobytes() + std_full.tobytes() +
            W.astype("<f4").tobytes() + b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class (argmax).
 * Weights: magic 0x504F4C32, d, Q, C; float32 mean[d], std[d], W_lin[d*C], W_quad[Q*C], bias[C].
 * Quadratic features: for i=0..d-1, for j=i..d-1: z[i]*z[j] (z = (x-mean)/std). */
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
    int d = hd[1], Q = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64 || Q != d*(d+1)/2) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * (2L*d + (long)d*C + (long)Q*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *Wlin = std + d;
    const float *Wquad = Wlin + (long)d * C;
    const float *bias = Wquad + (long)Q * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) z[j] = (xi[j] - mean[j]) / std[j];

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // Linear part
        for (int j = 0; j < d; j++) {
            const float *w = Wlin + (long)j * C;
            float zj = z[j];
            for (int c = 0; c < C; c++) scores[c] += zj * w[c];
        }

        // Quadratic part (i<=j)
        const float *wq = Wquad;
        for (int i1 = 0; i1 < d; i1++) {
            for (int j1 = i1; j1 < d; j1++) {
                float prod = z[i1] * z[j1];
                for (int c = 0; c < C; c++) scores[c] += prod * wq[c];
                wq += C;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```