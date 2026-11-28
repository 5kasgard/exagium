```hypothesis
We propose "label-fm", a Factorization Machine with low-rank pairwise interactions (K=8).
Adult income classification benefits from feature interactions (e.g., education × occupation).
FM models all pairwise interactions via factorized embeddings, improving over linear models
while keeping inference O(d·K) ≈ 14·8 = 112 multiply-adds per item.
If dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ 2325 (label-mlp16),
the family does not justify its complexity.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Trains an FM with binary cross-entropy, Adam, L2.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all data.
Weights format: magic 0x464D3132, int32 d, K, C; float32 b, w[d*256], e[d*256*K]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
K_OPTIONS = [4, 8, 16]
LR_OPTIONS = [0.01, 0.003]
L2_OPTIONS = [1e-4, 1e-5]
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


def train_fm(X, y, K, lr, l2, epochs, batch_size, rng):
    """Train FM on given data, return parameters (b, w, e)."""
    n, d = X.shape
    # Initialize parameters
    b = np.float32(0.0)
    w = rng.uniform(-0.01, 0.01, (d, 256)).astype(np.float32)
    e = rng.uniform(-0.01, 0.01, (d, 256, K)).astype(np.float32)
    # Adam state
    mb, mw, me = np.float32(0.0), np.zeros_like(w), np.zeros_like(e)
    vb, vw, ve = np.float32(0.0), np.zeros_like(w), np.zeros_like(e)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Gather parameters for this batch
            # w_batch: (B, d), e_batch: (B, d, K)
            w_batch = w[np.arange(d), xb.T].T
            e_batch = e[np.arange(d), xb.T].transpose(1, 0, 2)

            # Forward
            linear = b + w_batch.sum(axis=1)                     # (B,)
            emb_sum = e_batch.sum(axis=1)                        # (B, K)
            emb_sq_sum = (e_batch ** 2).sum(axis=1)              # (B, K)
            interaction = 0.5 * ((emb_sum ** 2 - emb_sq_sum).sum(axis=1))  # (B,)
            logit = linear + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            # Gradients
            dlogit = (p - yb) / B                                # (B,)
            db = dlogit.sum()
            # Gradient for w
            grad_w = np.zeros_like(w)
            for j in range(d):
                np.add.at(grad_w[j], xb[:, j], dlogit)
            # Gradient for e
            grad_e = np.zeros_like(e)
            for j in range(d):
                term = dlogit[:, None] * (emb_sum - e_batch[:, j, :])  # (B, K)
                np.add.at(grad_e[j], xb[:, j], term)
            # L2 regularization
            grad_w += l2 * w
            grad_e += l2 * e
            # Adam updates
            t += 1
            # b
            mb = beta1 * mb + (1 - beta1) * db
            vb = beta2 * vb + (1 - beta2) * db * db
            b -= lr * (mb / (1 - beta1 ** t)) / (np.sqrt(vb / (1 - beta2 ** t)) + eps)
            # w
            mw = beta1 * mw + (1 - beta1) * grad_w
            vw = beta2 * vw + (1 - beta2) * grad_w * grad_w
            w -= lr * (mw / (1 - beta1 ** t)) / (np.sqrt(vw / (1 - beta2 ** t)) + eps)
            # e
            me = beta1 * me + (1 - beta1) * grad_e
            ve = beta2 * ve + (1 - beta2) * grad_e * grad_e
            e -= lr * (me / (1 - beta1 ** t)) / (np.sqrt(ve / (1 - beta2 ** t)) + eps)
    return b, w, e


def evaluate_fm(X, y, b, w, e):
    """Compute accuracy on given data."""
    n, d = X.shape
    K = e.shape[2]
    # Vectorized evaluation for speed
    w_batch = w[np.arange(d), X.T].T          # (n, d)
    e_batch = e[np.arange(d), X.T].transpose(1, 0, 2)  # (n, d, K)
    linear = b + w_batch.sum(axis=1)
    emb_sum = e_batch.sum(axis=1)
    emb_sq_sum = (e_batch ** 2).sum(axis=1)
    interaction = 0.5 * ((emb_sum ** 2 - emb_sq_sum).sum(axis=1))
    logit = linear + interaction
    pred = (logit > 0).astype(np.int64)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    # Split: last 20% for tuning
    n = len(X)
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Grid search on tuning split
    best_acc = -1.0
    best_params = None
    best_cfg = None
    base_rng = np.random.default_rng(SEED)
    for K in K_OPTIONS:
        for lr in LR_OPTIONS:
            for l2 in L2_OPTIONS:
                # Use a deterministic seed per config
                cfg_seed = hash((K, lr, l2)) & 0xFFFFFFFF
                rng = np.random.default_rng(cfg_seed)
                epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
                b, w, e = train_fm(X_train, y_train, K, lr, l2, epochs, BATCH, rng)
                acc = evaluate_fm(X_val, y_val, b, w, e)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (K, lr, l2)
                    best_params = (b, w, e)

    # Retrain on full data with best config
    K, lr, l2 = best_cfg
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    b, w, e = train_fm(X, y, K, lr, l2, epochs, BATCH, rng)

    # Write weights
    magic = 0x464D3132  # "FM12"
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([b], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += e.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class index (0 or 1).
 * Weights format: magic 0x464D3132, int32 d, K, C; float32 b, w[d*256], e[d*256*K]. */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || d > 256 || K < 1 || K > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (1L + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *b = (const float *)(hd + 4);
    const float *w = b + 1;
    const float *e = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = b[0];
        float emb_sum[32] = {0};
        float emb_sq_sum[32] = {0};

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            linear += w[(long)j * 256 + v];
            const float *ev = e + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = ev[kk];
                emb_sum[kk] += val;
                emb_sq_sum[kk] += val * val;
            }
        }

        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float s = emb_sum[kk];
            interaction += s * s - emb_sq_sum[kk];
        }
        interaction *= 0.5f;

        float logit = linear + interaction;
        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```