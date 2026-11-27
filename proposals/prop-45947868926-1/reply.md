```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression).
Why: Optdigits pixels exhibit local correlations (strokes) that a linear model misses; explicit pairwise interactions capture these without the cost of a hidden layer. The 64 ordinal features yield 2080 expanded features – still small enough for fast inference.
Failure condition: Dev accuracy ≤ 0.9613 (label-linear) or CPU cost ≥ label-mlp16 (8685 Ir/item).
```

```python
"""label-poly2: polynomial logistic regression (degree 2) for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyper-parameters (tuned on the last 20% of train.in)
SEED = 1
LR = 3e-3
L2_GRID = [1e-5, 1e-4, 1e-3, 1e-2]   # L2 regularisation strengths to try
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


def standardize(X, mean, std):
    return (X - mean) / std


def expand_pairwise(Z):
    """Z: (n, d) -> (n, d + d*(d-1)//2) with all i<j products appended."""
    n, d = Z.shape
    feat_cnt = d + d * (d - 1) // 2
    X_exp = np.empty((n, feat_cnt), dtype=np.float32)
    X_exp[:, :d] = Z
    idx = d
    for i in range(d):
        for j in range(i + 1, d):
            X_exp[:, idx] = Z[:, i] * Z[:, j]
            idx += 1
    return X_exp


def train_logreg(X, y, C, rng, lr, l2, batch, epochs):
    """Adam optimisation for multinomial logistic regression.
    Returns (weights [feat_cnt, C], bias [C]) as float32."""
    n, feat_cnt = X.shape
    W = (rng.uniform(-1, 1, (feat_cnt, C)) * math.sqrt(6.0 / feat_cnt)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    def softmax_grad(S, y_idx):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(y_idx)), y_idx] -= 1.0
        return (P / len(y_idx)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            scores = xb @ W + b
            g = softmax_grad(scores, yb)
            grad_W = xb.T @ g + l2 * W
            grad_b = g.sum(axis=0)
            t += 1
            # Adam update
            mW = 0.9 * mW + 0.1 * grad_W
            vW = 0.999 * vW + 0.001 * grad_W * grad_W
            mb = 0.9 * mb + 0.1 * grad_b
            vb = 0.999 * vb + 0.001 * grad_b * grad_b
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return W.astype(np.float32), b.astype(np.float32)


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    # Tuning split: last 20% in file order
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    # Compute mean/std on train split for tuning
    mean_tr = X_bytes[train_idx].mean(axis=0).astype(np.float32)
    std_tr = X_bytes[train_idx].std(axis=0).astype(np.float32)
    std_tr[std_tr == 0] = 1.0

    Z_tr = standardize(X_bytes[train_idx].astype(np.float32), mean_tr, std_tr)
    Z_te = standardize(X_bytes[tune_idx].astype(np.float32), mean_tr, std_tr)

    X_tr_exp = expand_pairwise(Z_tr)
    X_te_exp = expand_pairwise(Z_te)

    y_tr, y_te = y[train_idx], y[tune_idx]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(train_idx)))

    best_l2, best_acc = None, -1.0
    for i, l2 in enumerate(L2_GRID):
        rng = np.random.default_rng(SEED + i)
        W, b = train_logreg(X_tr_exp, y_tr, C, rng, LR, l2, BATCH, epochs)
        scores_te = X_te_exp @ W + b
        pred_te = scores_te.argmax(axis=1)
        acc = (pred_te == y_te).mean()
        if acc > best_acc:
            best_acc, best_l2 = acc, l2

    # Retrain on full data with best_l2
    mean_full = X_bytes.mean(axis=0).astype(np.float32)
    std_full = X_bytes.std(axis=0).astype(np.float32)
    std_full[std_full == 0] = 1.0
    Z_full = standardize(X_bytes.astype(np.float32), mean_full, std_full)
    X_full_exp = expand_pairwise(Z_full)

    rng = np.random.default_rng(SEED)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W_final, b_final = train_logreg(X_full_exp, y, C, rng, LR, best_l2, BATCH, epochs_full)

    feat_cnt = X_full_exp.shape[1]
    magic = 0x324C4F50  # "POL2" little-endian
    header = struct.pack("<4i", magic, d, C, feat_cnt)
    out = header
    out += mean_full.tobytes()
    out += std_full.tobytes()
    out += b_final.tobytes()
    out += W_final.T.tobytes()  # column-major: C x feat_cnt
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: d bytes (ordinal). Output record: 1 byte class index. */
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
    if (nw < 16 || hd[0] != 0x324C4F50) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], feat_cnt = hd[3];
    if (d <= 0 || C <= 0 || C > 256 || feat_cnt != d + d * (d - 1) / 2) {
        fprintf(stderr, "bad weights dimensions\n"); return 3; }
    long expected = 16L + 4L * d * 2 + 4L * C + 4L * (long)feat_cnt * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *bias = std + d;
    const float *W    = bias + C;  // column-major: C rows, feat_cnt cols

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_std = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        /* linear terms */
        for (int j = 0; j < d; j++) {
            float val = (xi[j] - mean[j]) / std[j];
            x_std[j] = val;
            for (int c = 0; c < C; c++)
                scores[c] += W[c * feat_cnt + j] * val;
        }

        /* pairwise interaction terms */
        int idx = d;
        for (int u = 0; u < d - 1; u++) {
            for (int v = u + 1; v < d; v++) {
                float prod = x_std[u] * x_std[v];
                for (int c = 0; c < C; c++)
                    scores[c] += W[c * feat_cnt + idx] * prod;
                idx++;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(x_std); free(scores);
    if (x) free(x);
    return 0;
}
```