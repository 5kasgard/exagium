```hypothesis
Factorization Machine (FM) with embedding dimension 8 captures pairwise feature interactions crucial for tabular data like Adult. It should surpass the linear baseline (0.8524) and approach GBRT (0.8664) with far lower inference cost than tree ensembles. If dev accuracy does not exceed 0.855, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks: linear + pairwise interactions via embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only. Uses last 20% of train.in for tuning (epoch selection), then retrains on all data.
Exports: magic 0x464D3130, d, k, C, w0, w[d*256], V[d*256*k] (all float32 little-endian)."""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 8                 # embedding dimension
LR = 0.01             # Adam learning rate
L2 = 1e-4             # L2 regularization
BATCH = 64
MAX_EPOCHS = 50
VAL_FRACTION = 0.2    # last 20% for validation


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward(X_idx, w0, w, V, k):
    """X_idx: (n, d) uint8. Returns scores (n,)."""
    n, d = X_idx.shape
    linear = w0 + np.sum(w[X_idx, np.arange(d)], axis=1)  # w shape (d, 256)
    inter = np.zeros(n, dtype=np.float32)
    for f in range(k):
        Vf = V[:, :, f]                     # (d, 256)
        vals = Vf[np.arange(d), X_idx.T]    # (d, n)
        sum_v = np.sum(vals, axis=0)
        sum_sq = np.sum(vals * vals, axis=0)
        inter += 0.5 * (sum_v * sum_v - sum_sq)
    return linear + inter


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    val_start = int(n * (1 - VAL_FRACTION))
    tr_idx = np.arange(val_start)
    val_idx = np.arange(val_start, n)

    # Initialize parameters
    w0 = np.float32(0.0)
    w = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    best_val_acc = -1.0
    best_epoch = 0
    best_params = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # Shuffle training data
        rng.shuffle(tr_idx)
        for s in range(0, len(tr_idx), BATCH):
            batch_idx = tr_idx[s:s + BATCH]
            Xb = X[batch_idx]
            yb = y[batch_idx]
            n_b = len(yb)

            # Forward
            scores = forward(Xb, w0, w, V, K)
            p = sigmoid(scores)
            grad_score = (p - yb).astype(np.float32)  # (n_b,)

            # Gradients
            g_w0 = np.mean(grad_score)
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)

            # Linear weights gradient
            for j in range(d):
                vals = Xb[:, j]
                np.add.at(g_w[j], vals, grad_score)
            g_w /= n_b

            # V gradients
            for f in range(K):
                Vf = V[:, :, f]
                vals = Vf[np.arange(d), Xb.T]          # (d, n_b)
                sum_v = np.sum(vals, axis=0)           # (n_b,)
                for j in range(d):
                    v_j = vals[j]                      # (n_b,)
                    grad_v = grad_score * (sum_v - v_j)
                    np.add.at(g_V[j, :, f], Xb[:, j], grad_v)
            g_V /= n_b

            # L2 regularization
            g_w0 += L2 * w0
            g_w += L2 * w
            g_V += L2 * V

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

        # Validation accuracy
        if len(val_idx) > 0:
            val_scores = forward(X[val_idx], w0, w, V, K)
            val_pred = (val_scores > 0).astype(np.int64)
            val_acc = np.mean(val_pred == y[val_idx])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_params = (w0.copy(), w.copy(), V.copy())

    # Retrain on all data for best_epoch epochs
    if best_epoch > 0:
        w0, w, V = best_params
        # Re-initialize Adam state
        m_w0, v_w0 = 0.0, 0.0
        m_w, v_w = np.zeros_like(w), np.zeros_like(w)
        m_V, v_V = np.zeros_like(V), np.zeros_like(V)
        t = 0
        all_idx = np.arange(n)
        for epoch in range(1, best_epoch + 1):
            rng.shuffle(all_idx)
            for s in range(0, n, BATCH):
                batch_idx = all_idx[s:s + BATCH]
                Xb = X[batch_idx]
                yb = y[batch_idx]
                n_b = len(yb)

                scores = forward(Xb, w0, w, V, K)
                p = sigmoid(scores)
                grad_score = (p - yb).astype(np.float32)

                g_w0 = np.mean(grad_score)
                g_w = np.zeros_like(w)
                g_V = np.zeros_like(V)

                for j in range(d):
                    vals = Xb[:, j]
                    np.add.at(g_w[j], vals, grad_score)
                g_w /= n_b

                for f in range(K):
                    Vf = V[:, :, f]
                    vals = Vf[np.arange(d), Xb.T]
                    sum_v = np.sum(vals, axis=0)
                    for j in range(d):
                        v_j = vals[j]
                        grad_v = grad_score * (sum_v - v_j)
                        np.add.at(g_V[j, :, f], Xb[:, j], grad_v)
                g_V /= n_b

                g_w0 += L2 * w0
                g_w += L2 * w
                g_V += L2 * V

                t += 1
                m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
                v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
                m_w0_hat = m_w0 / (1 - beta1 ** t)
                v_w0_hat = v_w0 / (1 - beta2 ** t)
                w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

                m_w = beta1 * m_w + (1 - beta1) * g_w
                v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
                m_w_hat = m_w / (1 - beta1 ** t)
                v_w_hat = v_w / (1 - beta2 ** t)
                w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

                m_V = beta1 * m_V + (1 - beta1) * g_V
                v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Save weights
    MAGIC = 0x464D3130  # "FM10"
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label-fm: ./infer weights.bin in.rec N
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
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int j = 0; j < d; j++) linear += w[(long)j * 256 + xi[j]];

        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float sum_v = 0.0f, sum_sq = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[((long)j * 256 + xi[j]) * K + f];
                sum_v += v;
                sum_sq += v * v;
            }
            inter += 0.5f * (sum_v * sum_v - sum_sq);
        }
        float score = linear + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```