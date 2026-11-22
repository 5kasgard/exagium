```hypothesis
Propose label-elm: an Extreme Learning Machine with a fixed random first layer (ReLU) and ridge-regression output weights. The 8x8 optdigits pixels are ordinal; per-pixel z-scoring feeds a random projection (hidden size tuned on the last 20% of train). Training is a single linear solve (deterministic, seconds), inference is two dense matrix multiplies with precomputed per-pixel lookup tables (like label-linear but with a hidden layer). Expected to beat label-linear (0.9613) by capturing non-linearities at similar CPU cost. If dev accuracy ≤ 0.9613 or instructions/item exceed label-linear by >20%, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED = 1
LAMBDA = 1e-3          # ridge regularization for output weights
MAX_H = 512            # maximum hidden size for random weight matrix
H_CANDIDATES = [32, 64, 128, 256, 512]  # hidden sizes to try on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n_all = X_all.shape[0]

    # Split: last 20% for tuning (by file order)
    tune_size = n_all // 5
    train_size = n_all - tune_size
    X_train, X_tune = X_all[:train_size], X_all[train_size:]
    y_train, y_tune = y_all[:train_size], y_all[train_size:]

    rng = np.random.default_rng(SEED)
    # Pre-generate a large random weight matrix for the first layer (d x MAX_H)
    W1_big = (rng.uniform(-1, 1, (d, MAX_H)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1_big = np.zeros(MAX_H, np.float32)

    best_H = H_CANDIDATES[0]
    best_acc = -1.0

    for H in H_CANDIDATES:
        # Per-pixel z-scoring on training split
        mu = X_train.mean(axis=0).astype(np.float64)
        sd = X_train.std(axis=0).astype(np.float64)
        sd[sd == 0] = 1.0
        Z_train = ((X_train - mu) / sd).astype(np.float32)

        # First layer weights for this H
        W1 = W1_big[:, :H]
        b1 = b1_big[:H]

        # Hidden activations
        H_train = np.maximum(Z_train @ W1 + b1, 0, dtype=np.float32)
        # Add bias column
        H_train_bias = np.hstack([H_train, np.ones((train_size, 1), np.float32)])

        # One-hot labels
        Y_train = np.zeros((train_size, C), np.float32)
        Y_train[np.arange(train_size), y_train] = 1.0

        # Ridge regression: (H_train_bias^T H_train_bias + lambda I) W = H_train_bias^T Y
        A = H_train_bias.T @ H_train_bias
        A[np.diag_indices_from(A)] += LAMBDA
        B = H_train_bias.T @ Y_train
        W2_bias = np.linalg.solve(A, B).astype(np.float32)  # (H+1, C)
        W2 = W2_bias[:H, :]
        b2 = W2_bias[H, :]

        # Evaluate on tuning split
        Z_tune = ((X_tune - mu) / sd).astype(np.float32)
        H_tune = np.maximum(Z_tune @ W1 + b1, 0, dtype=np.float32)
        H_tune_bias = np.hstack([H_tune, np.ones((tune_size, 1), np.float32)])
        scores = H_tune_bias @ W2_bias
        pred = scores.argmax(axis=1)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_H = H

    # Retrain on all data with best_H
    mu = X_all.mean(axis=0).astype(np.float64)
    sd = X_all.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z_all = ((X_all - mu) / sd).astype(np.float32)

    W1 = W1_big[:, :best_H]
    b1 = b1_big[:best_H]

    H_all = np.maximum(Z_all @ W1 + b1, 0, dtype=np.float32)
    H_all_bias = np.hstack([H_all, np.ones((n_all, 1), np.float32)])

    Y_all = np.zeros((n_all, C), np.float32)
    Y_all[np.arange(n_all), y_all] = 1.0

    A = H_all_bias.T @ H_all_bias
    A[np.diag_indices_from(A)] += LAMBDA
    B = H_all_bias.T @ Y_all
    W2_bias = np.linalg.solve(A, B).astype(np.float32)
    W2 = W2_bias[:best_H, :]
    b2 = W2_bias[best_H, :]

    # Precompute per-pixel lookup tables T[d][256][H]
    # T[j, v, u] = W1[j, u] * (v - mu[j]) / sd[j]
    T = np.zeros((d, 256, best_H), dtype=np.float32)
    for j in range(d):
        scale = W1[j, :] / sd[j]          # (H,)
        offset = -mu[j] * scale           # (H,)
        vals = np.arange(256, dtype=np.float32)[:, None]  # (256,1)
        T[j] = vals * scale + offset      # (256, H)

    # Write weights.bin
    # Magic: 0x454C4D01 ('ELM\x01')
    magic = 0x454C4D01
    header = struct.pack("<4i", magic, d, best_H, C)
    out = header
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Extreme Learning Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class index.
 * Weights format: magic(0x454C4D01), d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C].
 * Inference: acc = b1 + sum_j T[j][x[j]]; ReLU; scores = b2 + acc @ W2; argmax.
 */
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
    if (nw < 16 || hd[0] != 0x454C4D01) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc, b1, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int u = 0; u < H; u++) acc[u] += t[u];
        }
        for (int u = 0; u < H; u++) if (acc[u] < 0.0f) acc[u] = 0.0f;

        memcpy(scores, b2, C * sizeof(float));
        for (int u = 0; u < H; u++) {
            float au = acc[u];
            if (au != 0.0f) {
                const float *w = W2 + (long)u * C;
                for (int c = 0; c < C; c++) scores[c] += au * w[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```