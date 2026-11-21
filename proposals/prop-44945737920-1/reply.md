```hypothesis
Family: label-poly2 (linear model on degree-2 polynomial features).
Why: 8x8 digit pixels have strong local pairwise correlations; a linear model on all 64 features plus their 2080 pairwise products (2144 total) captures this non-linearity while staying fast at inference (~21k multiply-adds/item). The model is trained with the same Adam/L2 regime as label-linear, with hyper-parameters (LR, L2, epochs) chosen on the last 20% of train.in.
Failure signal: if validation accuracy on the tuning split does not exceed label-linear's 96.1%, the extra cost is not justified.
```

```python
"""label-poly2: linear model on degree-2 polynomial features for ordinal byte inputs.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Chooses LR, L2, epochs on last 20% of train.in, then retrains on all.
Exports: magic 0x504F4C32, d, C, K; float32 mean[d], std[d], W[K][C], b[C] (row-major)."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
GRID = {
    "lr": [1e-3, 3e-3, 1e-2],
    "l2": [1e-4, 1e-3, 1e-2],
    "epochs": [30, 50, 70],
}


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def poly_expand(Z):
    """Z: (n, d) -> (n, K) with K = d + d*(d+1)//2"""
    n, d = Z.shape
    K = d + d * (d + 1) // 2
    Phi = np.empty((n, K), dtype=np.float32)
    Phi[:, :d] = Z
    idx = d
    for j in range(d):
        for k in range(j, d):
            Phi[:, idx] = Z[:, j] * Z[:, k]
            idx += 1
    return Phi


def train_logreg(Phi, y, C, lr, l2, epochs, rng):
    n, K = Phi.shape
    W = np.zeros((K, C), np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Phi[idx], y[idx]
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            G = (P / len(yy)).astype(np.float32)
            gW = z.T @ G + l2 * W
            gb = G.sum(0)
            t += 1
            for param, grad, m, v in [(W, gW, mW, vW), (b, gb, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b


def accuracy(Phi, y, W, b):
    S = Phi @ W + b
    pred = S.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    assert d == 64

    rng = np.random.default_rng(SEED)

    # split: last 20% for tuning
    n = len(X)
    n_val = n // 5
    n_tr = n - n_val
    X_tr, X_val = X[:n_tr], X[n_tr:]
    y_tr, y_val = y[:n_tr], y[n_tr:]

    # normalization on training portion
    mu = X_tr.mean(axis=0).astype(np.float64)
    sd = X_tr.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0

    Z_tr = ((X_tr - mu) / sd).astype(np.float32)
    Z_val = ((X_val - mu) / sd).astype(np.float32)

    Phi_tr = poly_expand(Z_tr)
    Phi_val = poly_expand(Z_val)
    K = Phi_tr.shape[1]

    best_acc = -1.0
    best_cfg = None
    best_W = best_b = None

    for lr in GRID["lr"]:
        for l2 in GRID["l2"]:
            for epochs in GRID["epochs"]:
                W, b = train_logreg(Phi_tr, y_tr, C, lr, l2, epochs, rng)
                acc = accuracy(Phi_val, y_val, W, b)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (lr, l2, epochs)
                    best_W, best_b = W.copy(), b.copy()

    # retrain on full data with best config
    mu_full = X.mean(axis=0).astype(np.float64)
    sd_full = X.std(axis=0).astype(np.float64)
    sd_full[sd_full == 0] = 1.0
    Z_full = ((X - mu_full) / sd_full).astype(np.float32)
    Phi_full = poly_expand(Z_full)
    lr, l2, epochs = best_cfg
    W_final, b_final = train_logreg(Phi_full, y, C, lr, l2, epochs, rng)

    # export
    magic = 0x504F4C32  # "POL2"
    out = struct.pack("<4i", magic, d, C, K)
    out += mu_full.astype("<f4").tobytes()
    out += sd_full.astype("<f4").tobytes()
    out += W_final.astype("<f4").tobytes()
    out += b_final.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=64). Output: 1 byte class index.
 * Weights format: magic 0x504F4C32, int32 d, C, K; float32 mean[d], std[d], W[K*C] (row-major), b[C]. */
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
    int d = hd[1], C = hd[2], K = hd[3];
    if (d != 64 || C < 1 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * K * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *W = std + d;
    const float *b = W + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) z[j] = (xi[j] - mean[j]) / std[j];

        int idx = 0;
        for (int j = 0; j < d; j++) phi[idx++] = z[j];
        for (int j = 0; j < d; j++)
            for (int k2 = j; k2 < d; k2++)
                phi[idx++] = z[j] * z[k2];

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k2 = 0; k2 < K; k2++) {
            float ph = phi[k2];
            const float *wrow = W + (long)k2 * C;
            for (int c = 0; c < C; c++) scores[c] += ph * wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```