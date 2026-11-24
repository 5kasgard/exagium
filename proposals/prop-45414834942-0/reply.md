```hypothesis
Family: label-fm (factorization machine with 4-dimensional embeddings per feature value).
Why: Captures pairwise feature interactions in O(d*k) time, which should improve over linear on tabular data with mixed ordinal/categorical features while keeping inference fast.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-mlp16 (≈2300 Ir/item).
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Uses last 20% of train.in for early stopping,
then retrains on all data. Exports: magic 0x464D3130, d, k, C, bias, w[d][256], emb[d][256][k] (float32 LE)."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, MAX_EPOCHS, PATIENCE = 1, 0.01, 1e-4, 256, 50, 5


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_logits(X, bias, w, emb):
    """X: (n, d) uint8. Returns logits (n,)."""
    n, d = X.shape
    logits = np.full(n, bias, dtype=np.float32)
    # linear part
    for j in range(d):
        logits += w[j, X[:, j]]
    # FM interaction part
    for f in range(K):
        sum_emb = np.zeros(n, dtype=np.float32)
        sum_emb_sq = np.zeros(n, dtype=np.float32)
        for j in range(d):
            e = emb[j, X[:, j], f]
            sum_emb += e
            sum_emb_sq += e * e
        logits += 0.5 * (sum_emb * sum_emb - sum_emb_sq)
    return logits


def accuracy(logits, y):
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert C == 2

    # split: last 20% for validation (tuning split)
    n = X.shape[0]
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    # init parameters
    bias = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K)) * np.sqrt(6.0 / (256 * K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    best_val_acc = -1.0
    best_epoch = 0
    best_params = None

    n_tr = X_tr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            bs = len(yb)

            # forward
            logits = np.full(bs, bias, dtype=np.float32)
            for j in range(d):
                logits += w[j, xb[:, j]]
            sum_emb = np.zeros((bs, K), dtype=np.float32)
            sum_emb_sq = np.zeros((bs, K), dtype=np.float32)
            for j in range(d):
                e = emb[j, xb[:, j], :]  # (bs, K)
                sum_emb += e
                sum_emb_sq += e * e
            for f in range(K):
                logits += 0.5 * (sum_emb[:, f] * sum_emb[:, f] - sum_emb_sq[:, f])

            # gradient of logistic loss
            p = sigmoid(logits)
            grad = (p - yb).astype(np.float32)  # (bs,)

            # bias grad
            g_bias = grad.mean()
            # w grad
            g_w = np.zeros_like(w)
            for j in range(d):
                for v in range(256):
                    mask = (xb[:, j] == v)
                    if np.any(mask):
                        g_w[j, v] = grad[mask].mean() + L2 * w[j, v]
            # emb grad
            g_emb = np.zeros_like(emb)
            for j in range(d):
                for v in range(256):
                    mask = (xb[:, j] == v)
                    if np.any(mask):
                        # for each factor f: grad = grad_i * (sum_emb_i_f - emb_j_v_f)
                        for f in range(K):
                            g_emb[j, v, f] = np.mean(grad[mask] * (sum_emb[mask, f] - emb[j, v, f])) + L2 * emb[j, v, f]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

        # validation
        val_logits = predict_logits(X_val, bias, w, emb)
        val_acc = accuracy(val_logits, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = (bias.copy(), w.copy(), emb.copy())
        elif epoch - best_epoch >= PATIENCE:
            break

    # retrain on full data for best_epoch epochs
    bias, w, emb = best_params
    # reset Adam state
    m_bias, v_bias = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0
    n_full = X.shape[0]
    for epoch in range(1, best_epoch + 1):
        order = rng.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            logits = np.full(bs, bias, dtype=np.float32)
            for j in range(d):
                logits += w[j, xb[:, j]]
            sum_emb = np.zeros((bs, K), dtype=np.float32)
            sum_emb_sq = np.zeros((bs, K), dtype=np.float32)
            for j in range(d):
                e = emb[j, xb[:, j], :]
                sum_emb += e
                sum_emb_sq += e * e
            for f in range(K):
                logits += 0.5 * (sum_emb[:, f] * sum_emb[:, f] - sum_emb_sq[:, f])

            p = sigmoid(logits)
            grad = (p - yb).astype(np.float32)

            g_bias = grad.mean()
            g_w = np.zeros_like(w)
            for j in range(d):
                for v in range(256):
                    mask = (xb[:, j] == v)
                    if np.any(mask):
                        g_w[j, v] = grad[mask].mean() + L2 * w[j, v]
            g_emb = np.zeros_like(emb)
            for j in range(d):
                for v in range(256):
                    mask = (xb[:, j] == v)
                    if np.any(mask):
                        for f in range(K):
                            g_emb[j, v, f] = np.mean(grad[mask] * (sum_emb[mask, f] - emb[j, v, f])) + L2 * emb[j, v, f]

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    # write weights.bin
    # header: magic, d, k, C (all int32 LE)
    # then bias (float32), then w[d][256] (float32), then emb[d][256][k] (float32)
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += struct.pack("<f", bias)
    out += w.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 4) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected_floats = 1L + (long)d * 256 * (1 + K);
    long expected_bytes = 16 + expected_floats * 4;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias_ptr = (const float *)(hd + 4);
    const float *w = bias_ptr + 1;
    const float *emb = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias_ptr[0];
        float sum_emb[4] = {0};
        float sum_emb_sq[4] = {0};

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += w[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                sum_emb[f] += e[f];
                sum_emb_sq[f] += e[f] * e[f];
            }
        }
        for (int f = 0; f < K; f++) {
            logit += 0.5f * (sum_emb[f] * sum_emb[f] - sum_emb_sq[f]);
        }

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```