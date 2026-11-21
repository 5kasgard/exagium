```hypothesis
Factorization Machine (degree 2) with low-rank embeddings for pairwise interactions.
Adult's 14 features are all encoded as 256 discrete bins; FM captures interactions without the parameter explosion of one-hot crosses.
Expect to beat label-linear (0.8524) on dev accuracy with similar inference cost (O(d^2*k) vs O(d)).
If dev accuracy ≤ 0.8524 or instructions/item exceed label-linear by >2x, the hypothesis fails.
```

```python
"""Factorization Machine (degree 2) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each feature as categorical with 256 values (0-255).
Hyperparameters tuned on last 20% of train.in (file order), then retrained on all data.
Weights format: magic 0x464D5F32, d, k, C, w0, linear[d][256], latent[d][256][k] (all float32 little-endian)."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03, 0.1]
L2_CHOICES = [1e-4, 1e-3, 1e-2]
EPOCHS_CHOICES = [10, 20, 30]
BATCH = 64
VAL_FRACTION = 0.2  # last 20% for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(x_idx, w0, W, V):
    """x_idx: (batch, d) int32 indices 0-255. Returns scores (batch,)."""
    batch, d = x_idx.shape
    # linear part
    lin = w0 + np.sum(W[np.arange(d), x_idx.T].T, axis=1)  # (batch,)
    # interaction part
    # sum of latent vectors per sample: (batch, k)
    sum_v = np.sum(V[np.arange(d), x_idx.T].transpose(1, 0, 2), axis=1)
    # sum of squares of latent vectors per sample: (batch, k)
    sum_v_sq = np.sum(V[np.arange(d), x_idx.T].transpose(1, 0, 2) ** 2, axis=1)
    # interaction = 0.5 * ( (sum_v)^2 - sum_v_sq ) summed over k
    inter = 0.5 * np.sum(sum_v ** 2 - sum_v_sq, axis=1)
    return lin + inter


def train_fm(X, y, k, lr, l2, epochs, rng):
    n, d = X.shape
    # parameters
    w0 = np.float32(0.0)
    W = np.zeros((d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, k)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = len(yb)
            # forward
            # linear
            lin = w0 + np.sum(W[np.arange(d), xb.T].T, axis=1)
            # interaction
            V_sel = V[np.arange(d), xb.T].transpose(1, 0, 2)  # (b, d, k)
            sum_v = np.sum(V_sel, axis=1)                     # (b, k)
            sum_v_sq = np.sum(V_sel ** 2, axis=1)             # (b, k)
            inter = 0.5 * np.sum(sum_v ** 2 - sum_v_sq, axis=1)
            score = lin + inter
            p = 1.0 / (1.0 + np.exp(-score))
            g = (p - yb).astype(np.float32)   # (b,)
            # gradients
            # dw0
            dw0 = np.mean(g)
            # dW
            dW = np.zeros_like(W)
            for j in range(d):
                cats = xb[:, j]
                np.add.at(dW[j], cats, g)
            dW /= b
            # dV
            dV = np.zeros_like(V)
            # For each sample, for each feature j: g * (sum_v - V_j)
            # Vectorized over batch and features:
            # sum_v shape (b, k), V_sel shape (b, d, k)
            diff = sum_v[:, None, :] - V_sel   # (b, d, k)
            grad_V = g[:, None, None] * diff   # (b, d, k)
            # accumulate over batch
            for j in range(d):
                cats = xb[:, j]
                np.add.at(dV[j], cats, grad_V[:, j, :])
            dV /= b
            # L2 regularization
            dw0 += l2 * w0
            dW += l2 * W
            dV += l2 * V
            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * dw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * dw0 * dw0
            m_hat = m_w0 / (1 - beta1 ** t)
            v_hat = v_w0 / (1 - beta2 ** t)
            w0 -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # W
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * dW * dW
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * dV * dV
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
    return w0, W, V


def eval_acc(X, y, w0, W, V):
    scores = fm_forward(X, w0, W, V)
    preds = (scores > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # split last 20% for validation
    val_size = int(n * VAL_FRACTION)
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_model = None

    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                for epochs in EPOCHS_CHOICES:
                    # train on train split
                    w0, W, V = train_fm(X_train, y_train, k, lr, l2, epochs, rng)
                    acc = eval_acc(X_val, y_val, w0, W, V)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (k, lr, l2, epochs)
                        best_model = (w0, W, V)
                    print(f"k={k} lr={lr} l2={l2} epochs={epochs} val_acc={acc:.4f}")

    print(f"Best: k={best_params[0]} lr={best_params[1]} l2={best_params[2]} epochs={best_params[3]} val_acc={best_acc:.4f}")
    # retrain on full data with best params
    k, lr, l2, epochs = best_params
    rng_full = np.random.default_rng(SEED)
    w0, W, V = train_fm(X, y, k, lr, l2, epochs, rng_full)

    # write weights.bin
    magic = 0x464D5F32  # "FM_2"
    out = struct.pack("<4i", magic, d, k, C)
    out += struct.pack("<f", w0)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (degree 2) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes (0-255). Output: one byte class (0 or 1). */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D5F32) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    // expected size: 16 (header) + 4 (w0) + 4*d*256 (W) + 4*d*256*k (V)
    long expected = 16 + 4 + 4L * d * 256 * (1 + k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(wb + 16);
    float w0 = *ptr++;
    const float *W = ptr; ptr += (long)d * 256;
    const float *V = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v_sq = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        // linear part
        float score = w0;
        for (int j = 0; j < d; j++) {
            score += W[(long)j * 256 + xi[j]];
        }
        // interaction part
        memset(sum_v, 0, k * sizeof(float));
        memset(sum_v_sq, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float val = v[f];
                sum_v[f] += val;
                sum_v_sq[f] += val * val;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) {
            float sv = sum_v[f];
            inter += 0.5f * (sv * sv - sum_v_sq[f]);
        }
        score += inter;

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_v);
    free(sum_v_sq);
    return 0;
}
```