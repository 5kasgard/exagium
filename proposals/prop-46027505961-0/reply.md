```hypothesis
Family: label-fm (Factorization Machine). The Adult dataset has strong pairwise interactions (education×occupation, marital-status×sex, capital-gain×hours) that linear models miss. A rank-8 FM captures all 91 feature-pair interactions in O(d·k) time via the FM trick (d=14, k=8 → ~32K params). Training uses Adam on the 80/20 tuning split for early stopping, then retrains on all data. If dev accuracy ≤ 0.853 (≤ label-linear), the interaction hypothesis fails for this quantization.
```

```python
"""Factorization Machine for tabular label tasks (binary).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format (little-endian):
  magic=0x464D3130, d, k, C(=1), w0, w[d*256], V[d*256*k]  (all float32)"""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 8                 # embedding rank
LR = 0.05
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 50
TUNE_FRAC = 0.2       # last 20% for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fm_forward(X, w0, W, V):
    """X: (n, d) uint8, W: (d, 256), V: (d, 256, k) -> scores (n,)"""
    n, d = X.shape
    k = V.shape[2]
    # linear term
    lin = w0 + np.sum(W[np.arange(d), X], axis=1)               # (n,)
    # interaction term: 0.5 * (||sum v||^2 - sum ||v||^2)
    V_x = V[np.arange(d), X]                                    # (n, d, k)
    sum_v = np.sum(V_x, axis=1)                                 # (n, k)
    sum_sq = np.sum(V_x * V_x, axis=(1, 2))                     # (n,)
    inter = 0.5 * (np.sum(sum_v * sum_v, axis=1) - sum_sq)      # (n,)
    return lin + inter


def train():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    n = X.shape[0]
    assert card["classes"] == 2

    # split: last 20% tune
    n_tune = int(n * TUNE_FRAC)
    n_tr = n - n_tune
    idx = np.arange(n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_tune, y_tune = X[n_tr:], y[n_tr:]

    rng = np.random.default_rng(SEED)
    # init
    w0 = np.float32(0.0)
    W = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def logloss(scores, labels):
        p = sigmoid(scores)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    best_epoch, best_loss = 0, float('inf')
    best_W, best_V, best_w0 = W.copy(), V.copy(), w0

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx_b = order[s:s + BATCH]
            xb, yb = X[idx_b], y[idx_b]
            bsz = len(yb)

            # forward
            V_xb = V[np.arange(d), xb.T].transpose(1, 0, 2)     # (bsz, d, k)
            sum_v = np.sum(V_xb, axis=1)                        # (bsz, k)
            sum_sq = np.sum(V_xb * V_xb, axis=(1, 2))           # (bsz,)
            inter = 0.5 * (np.sum(sum_v * sum_v, axis=1) - sum_sq)
            lin = w0 + np.sum(W[np.arange(d), xb.T], axis=1)
            scores = lin + inter

            # gradient of logloss wrt scores
            p = sigmoid(scores)
            grad_s = (p - yb) / bsz                             # (bsz,)

            # gradients
            gw0 = np.sum(grad_s)
            gW = np.zeros_like(W)
            gV = np.zeros_like(V)

            # linear grad
            for j in range(d):
                vals = xb[:, j]
                np.add.at(gW[j], vals, grad_s)

            # interaction grad: dL/dV = grad_s * (sum_v - V_x)
            # sum_v: (bsz, k), V_xb: (bsz, d, k)
            factor = grad_s[:, None] * (sum_v[:, None, :] - V_xb)   # (bsz, d, k)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(gV[j], vals, factor[:, j, :])

            # L2 regularization
            gW += L2 * W
            gV += L2 * V
            gw0 += L2 * w0

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m_hat = m_w0 / (1 - beta1 ** t)
            v_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # W
            m_W = beta1 * m_W + (1 - beta1) * gW
            v_W = beta2 * v_W + (1 - beta2) * gW * gW
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W -= LR * m_hat_W / (np.sqrt(v_hat_W) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            m_hat_V = m_V / (1 - beta1 ** t)
            v_hat_V = v_V / (1 - beta2 ** t)
            V -= LR * m_hat_V / (np.sqrt(v_hat_V) + eps)

        # tune evaluation
        tune_scores = fm_forward(X_tune, w0, W, V)
        loss = logloss(tune_scores, y_tune)
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch + 1
            best_w0, best_W, best_V = w0.copy(), W.copy(), V.copy()

    # retrain on full data for best_epoch epochs
    w0, W, V = best_w0, best_W, best_V
    m_w0, v_w0 = 0.0, 0.0
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    for epoch in range(best_epoch):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_b = order[s:s + BATCH]
            xb, yb = X[idx_b], y[idx_b]
            bsz = len(yb)

            V_xb = V[np.arange(d), xb.T].transpose(1, 0, 2)
            sum_v = np.sum(V_xb, axis=1)
            sum_sq = np.sum(V_xb * V_xb, axis=(1, 2))
            inter = 0.5 * (np.sum(sum_v * sum_v, axis=1) - sum_sq)
            lin = w0 + np.sum(W[np.arange(d), xb.T], axis=1)
            scores = lin + inter

            p = sigmoid(scores)
            grad_s = (p - yb) / bsz

            gw0 = np.sum(grad_s)
            gW = np.zeros_like(W)
            gV = np.zeros_like(V)

            for j in range(d):
                vals = xb[:, j]
                np.add.at(gW[j], vals, grad_s)

            factor = grad_s[:, None] * (sum_v[:, None, :] - V_xb)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(gV[j], vals, factor[:, j, :])

            gW += L2 * W
            gV += L2 * V
            gw0 += L2 * w0

            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_W = beta1 * m_W + (1 - beta1) * gW
            v_W = beta2 * v_W + (1 - beta2) * gW * gW
            W -= LR * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # save weights
    magic = 0x464D3130  # "FM10"
    C = 1
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    train()
```

```c
/* Factorization Machine inference for binary label tasks.
 * ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (0/1). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1], K = (int)hd[2], C = (int)hd[3];
    if (C != 1) { fprintf(stderr, "expected C=1\n"); return 3; }

    float *ptr = (float *)(hd + 4);
    float w0 = *ptr++;
    float *W = ptr; ptr += (long)d * 256;
    float *V = ptr; ptr += (long)d * 256 * K;

    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *V_x = malloc(sizeof(float) * d * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // linear term
        float score = w0;
        for (int j = 0; j < d; j++) score += W[(long)j * 256 + xi[j]];

        // interaction term via FM trick
        memset(sum_v, 0, sizeof(float) * K);
        float sum_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float vf = v[f];
                sum_v[f] += vf;
                sum_sq += vf * vf;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += sum_v[f] * sum_v[f];
        inter = 0.5f * (inter - sum_sq);
        score += inter;

        // sigmoid decision
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```