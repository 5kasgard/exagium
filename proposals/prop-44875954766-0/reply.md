```hypothesis
Family: label-fm (Factorization Machine for tabular byte features).
Why: FMs capture pairwise feature interactions via low-rank embeddings, crucial for tabular data where interactions (e.g., education×occupation) boost accuracy over linear models. With 14 features and 256 values each, a latent dim k=16 yields ~61k parameters, trainable via SGD in seconds. Inference is O(d*k) per item (≈224 ops), faster than GBDT/MLP.
Failure: If dev accuracy ≤ label-linear (0.8524) or Ir/item > label-linear (719), the model lacks sufficient signal or is too slow.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D31, d, k, C; float32 w0, w[d*256], v[d*256*k]."""
import json, math, struct, sys

import numpy as np

# Tuning grid (chosen on last 20% of train.in)
K_CHOICES = [8, 16, 32]
LR_CHOICES = [0.01, 0.03]
L2_CHOICES = [1e-4, 1e-3]
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


def train_fm(X, y, k, lr, l2, epochs, rng):
    n, d = X.shape
    # Parameters: w0 scalar, w[d,256], v[d,256,k]
    w0 = 0.0
    w = np.zeros((d, 256), dtype=np.float32)
    v = rng.normal(0, 0.01, size=(d, 256, k)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = len(idx)

            # Gather parameters for this batch
            # xb is (b, d) uint8 -> we need w[xb] and v[xb]
            # Use advanced indexing: for each feature j, gather w[j, xb[:,j]]
            w_batch = w[np.arange(d), xb.T]          # (d, b)
            v_batch = v[np.arange(d), xb.T]          # (d, b, k)

            sum_w = w_batch.sum(axis=0)              # (b,)
            sum_v = v_batch.sum(axis=0)              # (b, k)
            sum_v2 = (v_batch ** 2).sum(axis=0)      # (b, k)

            logits = w0 + sum_w + 0.5 * ((sum_v ** 2 - sum_v2).sum(axis=1))
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logit = (p - yb) / b                # (b,)

            # Gradients
            gw0 = grad_logit.sum()
            gw_batch = grad_logit                    # (b,)
            # dL/dv_batch = grad_logit[:,None] * (sum_v - v_batch)  # (d, b, k)
            gv_batch = grad_logit[:, None] * (sum_v - v_batch.transpose(1, 0, 2))  # (b, d, k) -> transpose to (d, b, k)
            gv_batch = gv_batch.transpose(1, 0, 2)   # now (d, b, k)

            # Scatter-add to parameter gradients
            gw = np.zeros_like(w)
            gv = np.zeros_like(v)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(gw[j], vals, gw_batch)
                for l in range(k):
                    np.add.at(gv[j, :, l], vals, gv_batch[j, :, l])

            # L2 regularization
            gw += l2 * w
            gv += l2 * v

            # Adam updates
            t += 1
            # w0
            m_w0 = 0.9 * m_w0 + 0.1 * gw0
            v_w0 = 0.999 * v_w0 + 0.001 * gw0 * gw0
            m_w0_hat = m_w0 / (1 - 0.9 ** t)
            v_w0_hat = v_w0 / (1 - 0.999 ** t)
            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + 1e-8)

            # w
            m_w = 0.9 * m_w + 0.1 * gw
            v_w = 0.999 * v_w + 0.001 * gw * gw
            m_w_hat = m_w / (1 - 0.9 ** t)
            v_w_hat = v_w / (1 - 0.999 ** t)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)

            # v
            m_v = 0.9 * m_v + 0.1 * gv
            v_v = 0.999 * v_v + 0.001 * gv * gv
            m_v_hat = m_v / (1 - 0.9 ** t)
            v_v_hat = v_v / (1 - 0.999 ** t)
            v -= lr * m_v_hat / (np.sqrt(v_v_hat) + 1e-8)

    return w0, w, v


def eval_fm(X, y, w0, w, v):
    n, d = X.shape
    k = v.shape[2]
    w_batch = w[np.arange(d), X.T]       # (d, n)
    v_batch = v[np.arange(d), X.T]       # (d, n, k)
    sum_w = w_batch.sum(axis=0)
    sum_v = v_batch.sum(axis=0)
    sum_v2 = (v_batch ** 2).sum(axis=0)
    logits = w0 + sum_w + 0.5 * ((sum_v ** 2 - sum_v2).sum(axis=1))
    preds = (logits > 0).astype(np.int64)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_all.shape[0]

    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, X_valid = X_all[:split], X_all[split:]
    y_train, y_valid = y_all[:split], y_all[split:]

    # Epochs based on sample budget
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / split))

    best_acc = -1.0
    best_params = None
    best_hp = None

    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                rng = np.random.default_rng(SEED)
                w0, w, v = train_fm(X_train, y_train, k, lr, l2, epochs, rng)
                acc = eval_fm(X_valid, y_valid, w0, w, v)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (w0, w, v)
                    best_hp = (k, lr, l2)

    print(f"Best hyperparams: k={best_hp[0]}, lr={best_hp[1]}, l2={best_hp[2]}, val_acc={best_acc:.4f}")

    # Retrain on all data with best hyperparams
    k, lr, l2 = best_hp
    epochs_all = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)
    w0, w, v = train_fm(X_all, y_all, k, lr, l2, epochs_all, rng)

    # Write weights.bin
    # Format: magic, d, k, C (int32), then w0, w (d*256), v (d*256*k) as float32 little-endian
    magic = 0x464D31
    out = struct.pack("<4i", magic, d, k, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class index (0 or 1). */
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
    if (hd[0] != 0x464D31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || k <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v2 = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        float logit = w0;
        for (int l = 0; l < k; l++) { sum_v[l] = 0.0f; sum_v2[l] = 0.0f; }

        for (int j = 0; j < d; j++) {
            int val = xi[j];
            long idx = (long)j * 256 + val;
            logit += w[idx];
            const float *v_ptr = v + (long)idx * k;
            for (int l = 0; l < k; l++) {
                float v_val = v_ptr[l];
                sum_v[l] += v_val;
                sum_v2[l] += v_val * v_val;
            }
        }
        for (int l = 0; l < k; l++) {
            logit += 0.5f * (sum_v[l] * sum_v[l] - sum_v2[l]);
        }
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```