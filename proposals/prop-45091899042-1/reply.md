```hypothesis
Family: label-onehot (logistic regression with per-pixel one-hot encoding).
Why: The 64 pixels have only 17 distinct values (0..16). Treating them as ordinal imposes a linear trend per pixel; one-hot encoding lets each pixel value learn an independent weight vector, capturing non-linear intensity effects without adding hidden units. The model stays a linear classifier, so inference remains a single dot product per class via lookup tables (same format as label-linear). If this does not beat label-linear (0.9613), then the ordinal linearity assumption is already near-optimal for this task.
```

```python
"""label-onehot: logistic regression with per-pixel one-hot encoding (0..16).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each of the 64 bytes as 17 one-hot features (values 0..16, others zero).
Grid-searches LR and L2 on the last 20% of train.in (tuning split), then trains on all data.
Exports weights as: magic=0x48454E4F, d, H=0, C; float32 T[d][256][C], b1[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
# Grid search space
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-5, 1e-4, 1e-3]
TUNE_EPOCHS = 50  # epochs for grid search on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def one_hot_encode(X, d, max_val=16):
    """X: (n, d) uint8, returns (n, d*(max_val+1)) float32 one-hot."""
    n = X.shape[0]
    width = max_val + 1
    Z = np.zeros((n, d * width), dtype=np.float32)
    for j in range(d):
        vals = X[:, j]
        # values are 0..16, clip just in case
        vals = np.clip(vals, 0, max_val)
        Z[np.arange(n), j * width + vals] = 1.0
    return Z

def train_logreg(Z, y, C, lr, l2, epochs, batch, rng, verbose=False):
    n, k = Z.shape
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            scores = z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            probs /= len(yy)
            gradW = z.T @ probs + l2 * W
            gradb = probs.sum(axis=0)
            t += 1
            # Adam updates
            mW = 0.9 * mW + 0.1 * gradW
            vW = 0.999 * vW + 0.001 * gradW * gradW
            mb = 0.9 * mb + 0.1 * gradb
            vb = 0.999 * vb + 0.001 * gradb * gradb
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return W, b

def evaluate(Z, y, W, b):
    scores = Z @ W + b
    pred = scores.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = X.shape[0]
    # Split: last 20% for tuning (validation), first 80% for grid search training
    split = int(0.8 * n_total)
    X_train_full, y_train_full = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # One-hot encode once
    Z_train_full = one_hot_encode(X_train_full, d)
    Z_val = one_hot_encode(X_val, d)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_lr, best_l2 = LR_GRID[0], L2_GRID[0]

    # Grid search on tuning split
    for lr in LR_GRID:
        for l2 in L2_GRID:
            W, b = train_logreg(Z_train_full, y_train_full, C, lr, l2, TUNE_EPOCHS, BATCH, rng)
            acc = evaluate(Z_val, y_val, W, b)
            if acc > best_acc:
                best_acc = acc
                best_lr, best_l2 = lr, l2

    # Retrain on all data with best hyperparameters
    Z_all = one_hot_encode(X, d)
    rng_full = np.random.default_rng(SEED)  # fresh seed for final training
    W_final, b_final = train_logreg(Z_all, y, C, best_lr, best_l2, MAX_EPOCHS, BATCH, rng_full)

    # Build per-byte lookup tables T[d][256][C]
    # For each pixel j, value v in 0..16 -> weight vector W_final[j*17 + v]
    # Values 17..255 get zero vector.
    W_flat = W_final  # shape (d*17, C)
    T = np.zeros((d, 256, C), dtype=np.float32)
    for j in range(d):
        for v in range(17):
            T[j, v] = W_flat[j * 17 + v]

    # Write weights.bin
    magic = 0x48454E4F  # "ONEH" little-endian
    H = 0
    W = C  # since H=0
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b_final.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-onehot infer: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels 0..16). Output: one byte class 0..9.
 * Weights format: magic=0x48454E4F, int32 d, H=0, C; float32 T[d][256][C], b1[C]. */
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
    if (nw < 16 || hd[0] != 0x48454E4F) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) sc[c] = b1[c];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) sc[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```