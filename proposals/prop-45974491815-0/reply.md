```hypothesis
We propose a Factorization Machine (FM) family for tabular data with byte-valued features. FM models pairwise feature interactions via low-rank embeddings, which should capture non-linear relationships in the Adult dataset (e.g., education × occupation) while remaining extremely fast at inference (O(d²k) with d=14, k≤8). The model is trained with Adam on logistic loss, using a held-out 20% tuning split to select the embedding dimension k. If dev accuracy does not exceed label-linear's 0.8524, or if inference cost (instructions/item) exceeds label-linear's 719 by more than a small factor, the approach would be deemed ineffective for this task.
```
```python
"""Factorization Machine for tabular byte features (binary classification).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x464D31, d, k, C=2, float32 w0, w[d][256], v[d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 30
K_CHOICES = [2, 4, 8, 16]  # tuning choices for embedding dim


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_fm(X, y, k, rng, epochs=EPOCHS, batch=BATCH, lr=LR, l2=L2):
    n, d = X.shape
    # parameters: w0 (scalar), w[d,256], v[d,256,k]
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = (rng.normal(0, 0.01, (d, 256, k))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_v = np.zeros_like(v)
    v_v = np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]          # (b, d) uint8
            yb = y[idx]          # (b,) int64 0/1
            b = len(yb)

            # forward
            # score = w0 + sum_i w[i, xi] + sum_{i<j} sum_f v[i,xi,f] * v[j,xj,f]
            scores = np.full(b, w0, dtype=np.float32)
            # linear part
            for i in range(d):
                scores += w[i, xb[:, i]]
            # interaction part
            # accumulate sum_i v[i,xi] for each factor
            sum_v = np.zeros((b, k), dtype=np.float32)
            for i in range(d):
                sum_v += v[i, xb[:, i], :]
            # sum_{i<j} <v_i, v_j> = 0.5 * ( (sum_i v_i)^2 - sum_i v_i^2 )
            inter = 0.5 * (np.sum(sum_v * sum_v, axis=1) - np.sum(v[xb] ** 2, axis=(1, 2)))
            scores += inter

            # logistic loss gradient
            # p = sigmoid(scores)
            p = 1.0 / (1.0 + np.exp(-scores))
            grad = (p - yb).astype(np.float32)  # (b,)

            # gradients
            g_w0 = grad.mean()
            g_w = np.zeros_like(w)
            g_v = np.zeros_like(v)

            for i in range(d):
                xi = xb[:, i]
                # linear gradient
                for val in range(256):
                    mask = (xi == val)
                    if np.any(mask):
                        g_w[i, val] = grad[mask].mean()
                # interaction gradient: for each sample, grad * (sum_v - v[i,xi])
                # sum_v shape (b,k), v[i,xi] shape (b,k)
                diff = sum_v - v[i, xi, :]  # (b,k)
                # g_v[i, xi, f] += mean_b( grad[b] * diff[b,f] )
                for val in range(256):
                    mask = (xi == val)
                    if np.any(mask):
                        g_v[i, val, :] = (grad[mask, None] * diff[mask]).mean(axis=0)

            # L2 regularization
            g_w += l2 * w
            g_v += l2 * v

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # v
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)
            v -= lr * m_v_hat / (np.sqrt(v_v_hat) + eps)

    return w0, w, v


def accuracy(X, y, w0, w, v):
    n, d = X.shape
    k = v.shape[2]
    scores = np.full(n, w0, dtype=np.float32)
    for i in range(d):
        scores += w[i, X[:, i]]
    sum_v = np.zeros((n, k), dtype=np.float32)
    for i in range(d):
        sum_v += v[i, X[:, i], :]
    inter = 0.5 * (np.sum(sum_v * sum_v, axis=1) - np.sum(v[X] ** 2, axis=(1, 2)))
    scores += inter
    preds = (scores > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)

    # split tuning: last 20% of train.in in file order
    n = len(y)
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    best_k = K_CHOICES[0]
    best_acc = -1.0
    for k in K_CHOICES:
        w0, w, v = train_fm(X_train, y_train, k, rng)
        acc = accuracy(X_tune, y_tune, w0, w, v)
        if acc > best_acc:
            best_acc = acc
            best_k = k

    # retrain on full data with best_k
    w0, w, v = train_fm(X, y, best_k, rng)

    # write weights.bin
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D31, d, best_k, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for binary classification.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k < 1 || k > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * d * 256 * (1L + k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(wb + 16);  // after 4 int32 = 16 bytes
    float w0 = p[0];
    const float *w = p + 1;                     // [d][256]
    const float *v = w + (long)d * 256;         // [d][256][k]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char pred;

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        float score = w0;
        // linear part
        for (int j = 0; j < d; j++) score += w[(long)j * 256 + xi[j]];
        // interaction part: 0.5 * ( (sum_f v_j)^2 - sum_f v_j^2 )
        // We'll accumulate sum_v[f] and sum_v_sq[f]
        float sum_v[16] = {0};   // k <= 16
        float sum_v_sq[16] = {0};
        for (int j = 0; j < d; j++) {
            const float *vj = v + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                sum_v[f] += vj[f];
                sum_v_sq[f] += vj[f] * vj[f];
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += sum_v[f] * sum_v[f] - sum_v_sq[f];
        score += 0.5f * inter;

        pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```