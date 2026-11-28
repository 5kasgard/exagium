```hypothesis
We propose a Factorization Machine (label-fm) that learns a linear weight and a low-rank embedding vector for each feature value, capturing pairwise interactions via dot products. This model is well-suited for the Adult dataset because it handles mixed categorical/ordinal features uniformly and can model feature interactions that a linear model misses, while remaining far cheaper than gradient-boosted trees. If dev accuracy does not exceed label-linear (0.8524) by at least 0.005, the approach fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: int32 magic 0x464D4C31, d, K, C; float32 w0, w[d*256], V[d*256*K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
BATCH = 64
EPOCHS = 20
K_CANDIDATES = [4, 8, 16]  # tried on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fm_forward(X, w0, w, V, K):
    """X: (n, d) uint8. Returns logits (n,)."""
    n, d = X.shape
    # linear part
    lin = w0 + np.sum(w[X.ravel()].reshape(n, d), axis=1)  # w indexed by (j*256 + val)
    # interaction part
    # V shape (d, 256, K)
    Vx = V[X, np.arange(d)]  # (n, d, K)
    sum_V = Vx.sum(axis=1)  # (n, K)
    sum_V_sq = (Vx ** 2).sum(axis=1)
    inter = 0.5 * ((sum_V ** 2 - sum_V_sq).sum(axis=1))
    return lin + inter


def train_fm(X, y, K, rng, epochs=EPOCHS, batch=BATCH, lr=LR):
    n, d = X.shape
    # parameters
    w0 = np.float32(0.0)
    w = np.zeros(d * 256, dtype=np.float32)
    V = (rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32))
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]  # (b, d)
            yb = y[idx]  # (b,)
            b = len(yb)

            # forward
            # linear
            lin = w0 + np.sum(w[xb.ravel()].reshape(b, d), axis=1)
            # interaction
            Vx = V[xb, np.arange(d)]  # (b, d, K)
            sum_V = Vx.sum(axis=1)  # (b, K)
            sum_V_sq = (Vx ** 2).sum(axis=1)
            inter = 0.5 * ((sum_V ** 2 - sum_V_sq).sum(axis=1))
            logits = lin + inter
            p = sigmoid(logits)
            # loss gradient
            dlogit = (p - yb).astype(np.float32)  # (b,)

            # gradients
            gw0 = dlogit.mean()
            # linear weights
            gw = np.zeros_like(w)
            np.add.at(gw, xb.ravel(), np.repeat(dlogit, d))
            gw /= b
            # embeddings
            # dinter/dV_{j,v,f} = dlogit * (sum_V[f] - V_{j,v,f})
            # sum_V is (b, K), Vx is (b, d, K)
            # For each sample i, feature j, we need dlogit[i] * (sum_V[i] - Vx[i,j])
            gV = np.zeros_like(V)
            # Expand dlogit to (b, 1, K) and sum_V to (b, 1, K)
            dlogit_exp = dlogit[:, None, None]  # (b,1,1)
            sum_V_exp = sum_V[:, None, :]  # (b,1,K)
            diff = sum_V_exp - Vx  # (b,d,K)
            grad_V = dlogit_exp * diff  # (b,d,K)
            # accumulate over batch
            for j in range(d):
                vals = xb[:, j]  # (b,)
                # grad_V[:, j, :] has shape (b, K)
                np.add.at(gV[j], vals, grad_V[:, j, :])
            gV /= b

            # Adam update
            t += 1
            # w0
            m_w0 = BETA1 * m_w0 + (1 - BETA1) * gw0
            v_w0 = BETA2 * v_w0 + (1 - BETA2) * gw0 * gw0
            m_w0_hat = m_w0 / (1 - BETA1 ** t)
            v_w0_hat = v_w0 / (1 - BETA2 ** t)
            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + EPS)

            # w
            m_w = BETA1 * m_w + (1 - BETA1) * gw
            v_w = BETA2 * v_w + (1 - BETA2) * gw * gw
            m_w_hat = m_w / (1 - BETA1 ** t)
            v_w_hat = v_w / (1 - BETA2 ** t)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + EPS)

            # V
            m_V = BETA1 * m_V + (1 - BETA1) * gV
            v_V = BETA2 * v_V + (1 - BETA2) * gV * gV
            m_V_hat = m_V / (1 - BETA1 ** t)
            v_V_hat = v_V / (1 - BETA2 ** t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + EPS)

    return w0, w, V


def accuracy(X, y, w0, w, V, K):
    logits = fm_forward(X, w0, w, V, K)
    pred = (logits > 0).astype(np.int64)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # tuning split: last 20% in file order
    n = len(y)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_K, best_acc = K_CANDIDATES[0], -1.0
    for K in K_CANDIDATES:
        w0, w, V = train_fm(X_train, y_train, K, rng)
        acc = accuracy(X_val, y_val, w0, w, V, K)
        if acc > best_acc:
            best_acc, best_K = acc, K

    # retrain on full data with best_K
    w0, w, V = train_fm(X, y, best_K, rng)

    # write weights.bin
    # magic, d, K, C
    header = struct.pack("<4i", 0x464D4C31, d, best_K, C)
    # w0, w, V (flattened)
    body = struct.pack("<f", w0) + w.astype("<f4").tobytes() + V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0 or 1).
 * Weights format: int32 magic 0x464D4C31, d, K, C; float32 w0, w[d*256], V[d*256*K]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * (1L + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = w0;
        // zero sum_V
        for (int f = 0; f < K; f++) sum_V[f] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            long idx = (long)j * 256 + val;
            logit += w[idx];
            const float *v = V + idx * K;
            for (int f = 0; f < K; f++) sum_V[f] += v[f];
        }

        float inter = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            long idx = (long)j * 256 + val;
            const float *v = V + idx * K;
            for (int f = 0; f < K; f++) {
                float diff = sum_V[f] - v[f];
                inter += diff * v[f];
            }
        }
        logit += 0.5f * inter;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_V);
    return 0;
}
```