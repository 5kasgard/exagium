```hypothesis
We propose "label-fm", a factorization machine with low-rank embeddings for pairwise feature interactions.
Adult has mixed categorical/ordinal features where interactions (e.g., education × occupation) are known to be predictive.
A factorization machine captures all pairwise interactions with O(d·K) parameters and inference cost, avoiding O(d²) blowup.
It should beat label-linear (0.8524) at similar cost (K=8 → ~28k params, ~1k Ir/item). If dev accuracy ≤0.8524 or Ir/item >1000, the approach fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D4C31, d, K, C=2; float32 bias, W[d*256], V[d*256*K]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters for grid search and final training
MASTER_SEED = 1
SAMPLES = 600000
MAX_EPOCHS = 100
BATCH = 64
L2 = 1e-4
GRID_EPOCHS = 10  # epochs for each grid config on tuning split

# Grid of (K, learning_rate) to try on the tuning split
GRID_CONFIGS = [
    (4, 0.03),
    (8, 0.03),
    (8, 0.1),
    (16, 0.03),
]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, W, V, bias, K):
    """Compute FM scores for a batch X (n, d). Returns scores (n,)."""
    n, d = X.shape
    scores = np.full(n, bias, dtype=np.float32)
    # Linear part
    for j in range(d):
        scores += W[j, X[:, j]]
    # Interaction part
    for k in range(K):
        sum_V = np.zeros(n, dtype=np.float32)
        sum_V2 = np.zeros(n, dtype=np.float32)
        for j in range(d):
            v = V[j, X[:, j], k]
            sum_V += v
            sum_V2 += v * v
        scores += 0.5 * (sum_V * sum_V - sum_V2)
    return scores


def fm_train(X, y, K, lr, epochs, rng):
    """Train FM on (X, y) for given epochs. Returns W, V, bias."""
    n, d = X.shape
    # Initialize parameters
    W = rng.uniform(-0.01, 0.01, (d, 256)).astype(np.float32)
    V = rng.uniform(-0.01, 0.01, (d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.float32(0.0), np.float32(0.0)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward pass: compute scores and gradients
            # We need per-sample sum_V and sum_V2 for each k
            sum_V = np.zeros((bsz, K), dtype=np.float32)
            sum_V2 = np.zeros((bsz, K), dtype=np.float32)
            for j in range(d):
                v = V[j, Xb[:, j], :]  # (bsz, K)
                sum_V += v
                sum_V2 += v * v

            linear = bias + np.sum(W[np.arange(d)[:, None], Xb.T], axis=0)  # (bsz,)
            interaction = 0.5 * np.sum(sum_V * sum_V - sum_V2, axis=1)
            scores = linear + interaction
            p = 1.0 / (1.0 + np.exp(-scores))
            g = p - yb  # (bsz,)

            # Gradients
            grad_bias = np.mean(g)
            grad_W = np.zeros_like(W)
            grad_V = np.zeros_like(V)

            # Linear weights gradient
            for j in range(d):
                np.add.at(grad_W[j], Xb[:, j], g)

            # Embedding gradients
            for k in range(K):
                for j in range(d):
                    v = V[j, Xb[:, j], k]
                    # g * (sum_V[:, k] - v)
                    np.add.at(grad_V[j, :, k], Xb[:, j], g * (sum_V[:, k] - v))

            # L2 regularization
            grad_W += L2 * W
            grad_V += L2 * V

            # Adam update
            t += 1
            # bias
            m_b = beta1 * m_b + (1 - beta1) * grad_bias
            v_b = beta2 * v_b + (1 - beta2) * (grad_bias * grad_bias)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            bias -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

            # W
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * (grad_W * grad_W)
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * (grad_V * grad_V)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

    return W, V, bias


def evaluate(X, y, W, V, bias, K):
    scores = fm_forward(X, W, V, bias, K)
    preds = (scores > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2, "Only binary classification supported"

    # Load all training data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Split: last 20% for tuning (card's tuning split)
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    # Grid search on tuning split
    best_acc = -1.0
    best_cfg = None
    best_params = None

    for idx, (K, lr) in enumerate(GRID_CONFIGS):
        rng = np.random.default_rng(MASTER_SEED + idx)
        W, V, bias = fm_train(X_train, y_train, K, lr, GRID_EPOCHS, rng)
        acc = evaluate(X_tune, y_tune, W, V, bias, K)
        if acc > best_acc:
            best_acc = acc
            best_cfg = (K, lr)
            best_params = (W, V, bias)

    # Final training on all data with best config
    K, lr = best_cfg
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(MASTER_SEED + 1000)  # fixed seed for final run
    W, V, bias = fm_train(X, y, K, lr, epochs, rng)

    # Write weights.bin
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    # Flatten arrays in C order (row-major)
    W_flat = W.astype("<f4").ravel()
    V_flat = V.astype("<f4").ravel()
    bias_arr = np.array([bias], dtype="<f4")
    out = header + bias_arr.tobytes() + W_flat.tobytes() + V_flat.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class (0 or 1). */
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * (1L + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *W = bias + 1;
    const float *V = W + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = *bias;
        float sum_V[16];   // K <= 16 in our grid
        float sum_V2[16];
        for (int kk = 0; kk < K; kk++) { sum_V[kk] = 0.0f; sum_V2[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += W[(long)j * 256 + v];
            const float *Vj = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = Vj[kk];
                sum_V[kk] += val;
                sum_V2[kk] += val * val;
            }
        }

        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            interaction += 0.5f * (sum_V[kk] * sum_V[kk] - sum_V2[kk]);
        }
        float score = linear + interaction;
        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```