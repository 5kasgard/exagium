```hypothesis
Family: label-quadratic (quadratic logistic regression on z-scored pixels).
Why: Optdigits pixels have local correlations; quadratic terms capture pairwise interactions without deep layers. It should improve over linear (0.961) and approach MLP (0.977) with lower inference cost than MLP (only 2080 features).
Failure: If dev accuracy is below 0.965 or inference cost exceeds label-linear by >2x.
```

```python
"""Quadratic family for label tasks with fixed-size ordinal bytes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each byte as z-score, expands to all pairwise products (including squares),
trains multinomial logistic regression with L2 via Adam.
Hyperparameters (L2, lr) chosen on last 20% of train.in, then retrained on all.
Weights format: magic 0x51554144, d, C; float32 mean[d], std[d], W[C][K], bias[C] (K=d*(d+1)/2)."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
L2_GRID = [1e-3, 1e-4, 1e-5, 1e-6]
LR_GRID = [3e-3, 1e-3]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def expand(Z):
    """Z: (n, d) -> Phi: (n, K) where K = d*(d+1)//2, all products z_i*z_j for i<=j."""
    n, d = Z.shape
    K = d * (d + 1) // 2
    Phi = np.empty((n, K), dtype=np.float32)
    for idx in range(n):
        z = Z[idx]
        p = 0
        for i in range(d):
            zi = z[i]
            for j in range(i, d):
                Phi[idx, p] = zi * z[j]
                p += 1
    return Phi


def train_logreg(Phi, y, C, L2, LR, epochs, rng):
    n, K = Phi.shape
    W = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Phi[idx], y[idx]
            # softmax gradient
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def accuracy(Phi, y, W, b):
    S = Phi @ W + b
    pred = S.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    K = d * (d + 1) // 2

    # z-score per pixel (all ordinal)
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    # tuning split: last 20%
    n = len(X)
    n_tune = n // 5
    tune_idx = slice(n - n_tune, n)
    train_idx = slice(0, n - n_tune)

    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_tune, y_tune = Z[tune_idx], y[tune_idx]

    Phi_train = expand(Z_train)
    Phi_tune = expand(Z_tune)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_W = best_b = None
    best_hp = None

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(Z_train)))
    for L2 in L2_GRID:
        for LR in LR_GRID:
            W, b = train_logreg(Phi_train, y_train, C, L2, LR, epochs, rng)
            acc = accuracy(Phi_tune, y_tune, W, b)
            if acc > best_acc:
                best_acc = acc
                best_W, best_b = W.copy(), b.copy()
                best_hp = (L2, LR)

    # retrain on full data with best hyperparameters
    Phi_full = expand(Z)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)
    W_final, b_final = train_logreg(Phi_full, y, C, best_hp[0], best_hp[1], epochs_full, rng)

    # save weights
    magic = 0x51554144  # 'QUAD'
    header = struct.pack("<3i", magic, d, C)
    means = mu.astype("<f4")
    stds = sd.astype("<f4")
    weights = best_W.astype("<f4")  # shape (K, C) but we need (C, K) for infer? We'll store (C, K) row-major.
    # train_logreg returns W shape (K, C). Transpose to (C, K).
    weights = weights.T
    biases = best_b.astype("<f4")
    out = header + means.tobytes() + stds.tobytes() + weights.tobytes() + biases.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Quadratic family inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class (argmax).
 * Weights format: magic 0x51554144, int32 d, int32 C; float32 mean[d], std[d], W[C][K], bias[C] (K=d*(d+1)/2). */
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
    if (nw < 12 || hd[0] != 0x51554144) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    int K = d * (d + 1) / 2;
    long expected = 12L + 4L * (2L * d + (long)C * K + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 3);
    const float *std = mean + d;
    const float *W = std + d;           // shape [C][K]
    const float *bias = W + (long)C * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // z-score
        for (int j = 0; j < d; j++) {
            float m = mean[j], s = std[j];
            z[j] = (s != 0.0f) ? ((float)xi[j] - m) / s : 0.0f;
        }
        // pairwise products
        int p = 0;
        for (int i1 = 0; i1 < d; i1++) {
            float zi = z[i1];
            for (int i2 = i1; i2 < d; i2++) {
                phi[p++] = zi * z[i2];
            }
        }
        // scores
        for (int c = 0; c < C; c++) {
            const float *Wc = W + (long)c * K;
            float sum = bias[c];
            for (int kk = 0; kk < K; kk++) sum += phi[kk] * Wc[kk];
            scores[c] = sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```