```hypothesis
Factorization Machine (label-fm) with 8-d embeddings captures pairwise interactions efficiently.
Adult has mixed categorical/ordinal features; FMs excel on such sparse tabular data.
Should beat label-linear (0.8524) and approach label-gbdt (0.8664) at near label-linear speed.
Failure: dev accuracy ≤ 0.8524 or instructions/item > 2000.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each byte as a categorical feature (256 values per column).
Learns per-category bias and k-dim embedding. Interaction via (sum v)^2 - sum v^2.
Trains with SGD + momentum, logistic loss, L2 regularization.
Tunes lr, L2 on last 20% of train.in (file order), then retrains on all data.
Exports: magic 0x464D4C31, d, k, C, w0, biases[d*256], embeddings[d*256*k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (tuned on validation split)
SEED = 1
K = 8                 # embedding dimension
EPOCHS = 30
BATCH = 256
MOMENTUM = 0.9
LR_GRID = [0.1, 0.03, 0.01]
L2_GRID = [1e-4, 1e-3, 1e-2]
VAL_FRAC = 0.2        # last 20% for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fm_forward(X, w0, biases, embeddings, d, k):
    """X: (n, d) uint8. Returns logits (n,)."""
    n = X.shape[0]
    logits = np.full(n, w0, dtype=np.float32)
    # bias term
    for j in range(d):
        logits += biases[j * 256 + X[:, j]]
    # interaction term
    sum_emb = np.zeros((n, k), dtype=np.float32)
    sum_emb_sq = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        emb = embeddings[(j * 256 + X[:, j]) * k:(j * 256 + X[:, j] + 1) * k]  # wrong shape
    # Better: gather embeddings for all features at once
    # Precompute flat indices
    idx = (np.arange(d) * 256 + X.T).T  # (n, d)
    # embeddings shape: (d*256, k)
    emb_all = embeddings[idx]  # (n, d, k)
    sum_emb = emb_all.sum(axis=1)          # (n, k)
    sum_emb_sq = (emb_all ** 2).sum(axis=1)  # (n, k)
    interaction = 0.5 * ((sum_emb ** 2) - sum_emb_sq).sum(axis=1)
    logits += interaction
    return logits


def fm_backward(X, y, logits, w0, biases, embeddings, d, k, l2):
    """Compute gradients for one batch. Returns grad_w0, grad_biases, grad_embeddings."""
    n = X.shape[0]
    probs = sigmoid(logits)
    grad_logits = probs - y  # (n,)
    # grad_w0
    gw0 = grad_logits.mean()
    # grad_biases: for each feature j, category v, sum grad_logits[i] where X[i,j]==v
    gbiases = np.zeros(d * 256, dtype=np.float32)
    for j in range(d):
        cats = X[:, j]
        np.add.at(gbiases, j * 256 + cats, grad_logits)
    gbiases /= n
    gbiases += l2 * biases
    # grad_embeddings: for each feature j, category v, embedding dim f:
    # sum_i grad_logits[i] * (sum_emb[i,f] - emb[j,v,f])
    gemb = np.zeros(d * 256 * k, dtype=np.float32)
    # compute sum_emb for batch
    idx = (np.arange(d) * 256 + X.T).T  # (n, d)
    emb_all = embeddings[idx]  # (n, d, k)
    sum_emb = emb_all.sum(axis=1)  # (n, k)
    for j in range(d):
        cats = X[:, j]
        base = j * 256 * k
        for f in range(k):
            # grad for embedding dim f: sum_i grad_logits[i] * (sum_emb[i,f] - emb[j, cats[i], f])
            # vectorized:
            diff = sum_emb[:, f] - embeddings[(j * 256 + cats) * k + f]
            grad = grad_logits * diff
            # accumulate per category
            np.add.at(gemb, base + cats * k + f, grad)
    gemb /= n
    gemb += l2 * embeddings
    return gw0, gbiases, gemb


def train_fm(X, y, d, k, lr, l2, epochs, batch, momentum, rng):
    n = X.shape[0]
    # init
    w0 = 0.0
    biases = rng.normal(0, 0.01, d * 256).astype(np.float32)
    embeddings = rng.normal(0, 0.01, d * 256 * k).astype(np.float32)
    # momentum buffers
    v_w0 = 0.0
    v_biases = np.zeros_like(biases)
    v_emb = np.zeros_like(embeddings)
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, yb = X[idx], y[idx]
            logits = fm_forward(Xb, w0, biases, embeddings, d, k)
            gw0, gbiases, gemb = fm_backward(Xb, yb, logits, w0, biases, embeddings, d, k, l2)
            # momentum update
            v_w0 = momentum * v_w0 + gw0
            v_biases = momentum * v_biases + gbiases
            v_emb = momentum * v_emb + gemb
            w0 -= lr * v_w0
            biases -= lr * v_biases
            embeddings -= lr * v_emb
    return w0, biases, embeddings


def eval_acc(X, y, w0, biases, embeddings, d, k):
    logits = fm_forward(X, w0, biases, embeddings, d, k)
    preds = (logits > 0).astype(np.int64)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]
    val_size = int(n * VAL_FRAC)
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_model = None

    for lr in LR_GRID:
        for l2 in L2_GRID:
            # train on train split
            w0, biases, embeddings = train_fm(X_train, y_train, d, K, lr, l2, EPOCHS, BATCH, MOMENTUM, rng)
            acc = eval_acc(X_val, y_val, w0, biases, embeddings, d, K)
            if acc > best_acc:
                best_acc = acc
                best_params = (lr, l2)
                best_model = (w0, biases.copy(), embeddings.copy())

    print(f"Best val acc: {best_acc:.4f} lr={best_params[0]} l2={best_params[1]}", file=sys.stderr)
    # retrain on full data with best params
    lr, l2 = best_params
    rng_full = np.random.default_rng(SEED)
    w0, biases, embeddings = train_fm(X, y, d, K, lr, l2, EPOCHS, BATCH, MOMENTUM, rng_full)

    # write weights
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    w0_bytes = struct.pack("<f", w0)
    biases_bytes = biases.astype("<f4").tobytes()
    embeddings_bytes = embeddings.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + w0_bytes + biases_bytes + embeddings_bytes)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1).
 * Weights format: int32 magic(0x464D4C31), d, k, C; float32 w0; float32 biases[d*256]; float32 embeddings[d*256*k]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || k <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + d * 256 + d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *biases = p + 1;
    const float *embeddings = biases + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * k);
    float *sum_emb_sq = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float logit = w0;
        // bias term
        for (int j = 0; j < d; j++) {
            logit += biases[j * 256 + xi[j]];
        }
        // interaction term
        memset(sum_emb, 0, sizeof(float) * k);
        memset(sum_emb_sq, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            const float *emb = embeddings + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float v = emb[f];
                sum_emb[f] += v;
                sum_emb_sq[f] += v * v;
            }
        }
        for (int f = 0; f < k; f++) {
            logit += 0.5f * (sum_emb[f] * sum_emb[f] - sum_emb_sq[f]);
        }

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_emb);
    free(sum_emb_sq);
    return 0;
}
```