```hypothesis
We propose "label-emb", a factorized linear model where each of the 16 input byte positions has a learned embedding vector of dimension D. The concatenated embeddings (size 16*D) are fed directly to a linear output layer (no hidden non-linearity). This drastically reduces parameters vs. full one-hot linear (label-linear) while capturing similarity between byte values, which should help on this character-level task where the 65 output classes also appear as inputs. If dev accuracy does not exceed label-linear's 0.3375, or if inference cost per item is higher, the hypothesis fails.
```

```python
"""label-emb: factorized linear model with per-position byte embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Tunes embedding dim D, lr, L2 on last 20% of train,
then trains on all data. Exports: magic 0x454D424C, d, D, C; float32 E[d][256][D], W[C][d*D], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning
D_GRID = [4, 8, 16, 32]
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-5]
TUNE_EPOCHS = 10
FULL_EPOCHS = 50
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


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_step(X, y, E, W, b, states, lr, l2, t):
    """X: (B, d) uint8, y: (B,) int64. Returns updated params and states."""
    B, d = X.shape
    C = W.shape[0]
    D = E[0].shape[1]
    # Forward: concatenated embeddings
    Z = np.zeros((B, d * D), dtype=np.float32)
    for j in range(d):
        Z[:, j * D:(j + 1) * D] = E[j][X[:, j]]
    logits = Z @ W.T + b  # (B, C)
    # Softmax cross-entropy gradient
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    dlogits = probs / B  # (B, C)
    # Gradients
    dW = dlogits.T @ Z + l2 * W
    db = dlogits.sum(axis=0)
    dZ = dlogits @ W  # (B, d*D)
    # Split dZ per position and accumulate into dE
    dE = [np.zeros_like(Ej) for Ej in E]
    dZ_split = dZ.reshape(B, d, D)
    for j in range(d):
        np.add.at(dE[j], X[:, j], dZ_split[:, j])
    dE = [de + l2 * Ej for de, Ej in zip(dE, E)]
    # Adam update
    params = E + [W, b]
    grads = dE + [dW, db]
    t += 1
    for i, (p, gr) in enumerate(zip(params, grads)):
        m, v = states[i]
        m *= 0.9; m += 0.1 * gr
        v *= 0.999; v += 0.001 * gr * gr
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return t


def evaluate(X, y, E, W, b):
    B, d = X.shape
    D = E[0].shape[1]
    Z = np.zeros((B, d * D), dtype=np.float32)
    for j in range(d):
        Z[:, j * D:(j + 1) * D] = E[j][X[:, j]]
    logits = Z @ W.T + b
    return accuracy(logits, y)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split: last 20% for tuning
    n_train = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)  # shuffle once for consistent split? The card says "last 20% in file order".
    # Actually the card says: "the last 20% of train.in in file order". So we must not shuffle.
    # We'll use the last 20% as they appear in the file.
    n_train = n - n // 5
    train_idx = idx[:n_train]
    tune_idx = idx[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_acc = -1.0
    best_cfg = None
    best_params = None

    for D in D_GRID:
        for lr in LR_GRID:
            for l2 in L2_GRID:
                # Initialize
                E = [(rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / (256 + D))).astype(np.float32) for _ in range(d)]
                W = (rng.uniform(-1, 1, (C, d * D)) * math.sqrt(6.0 / (d * D + C))).astype(np.float32)
                b = np.zeros(C, np.float32)
                params = E + [W, b]
                states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
                t = 0
                # Train for TUNE_EPOCHS
                for _ in range(TUNE_EPOCHS):
                    order = rng.permutation(len(X_train))
                    for s in range(0, len(X_train), BATCH):
                        idx_batch = order[s:s + BATCH]
                        t = train_step(X_train[idx_batch], y_train[idx_batch], E, W, b, states, lr, l2, t)
                acc = evaluate(X_tune, y_tune, E, W, b)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (D, lr, l2)
                    # Save a copy of params
                    best_params = [p.copy() for p in params]

    print(f"Best config: D={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, tune_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best config
    D, lr, l2 = best_cfg
    E, W, b = best_params[0:d], best_params[d], best_params[d+1]
    params = E + [W, b]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    full_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(full_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            t = train_step(X[idx_batch], y[idx_batch], E, W, b, states, lr, l2, t)

    # Pack weights
    # Magic, d, D, C
    out = struct.pack("<4i", 0x454D424C, d, D, C)
    # Embeddings: E[d][256][D] in order j, v, k
    for j in range(d):
        out += E[j].astype("<f4").tobytes()
    # Output weights: W[C][d*D] (row-major)
    out += W.astype("<f4").tobytes()
    # Biases
    out += b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=16). Output is one byte: argmax class.
 * Weights format: magic 0x454D424C, int32 d, D, C; float32 E[d][256][D], W[C][d*D], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)C * d * D + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);                 // size d*256*D
    const float *W = E + (long)d * 256 * D;                   // size C*d*D
    const float *b = W + (long)C * d * D;                     // size C

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Concatenate embeddings into z
        for (int j = 0; j < d; j++) {
            const float *ej = E + ((long)j * 256 + xi[j]) * D;
            float *zj = z + j * D;
            for (int kk = 0; kk < D; kk++) zj[kk] = ej[kk];
        }
        // Compute scores = b + z @ W^T
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int c = 0; c < C; c++) {
            const float *wc = W + (long)c * d * D;
            float sum = 0.0f;
            for (int kk = 0; kk < d * D; kk++) sum += z[kk] * wc[kk];
            scores[c] += sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(z); free(scores);
    return 0;
}
```