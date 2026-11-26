```hypothesis
Family: label-fm (Factorization Machine with pairwise interactions).
Why: Adult income has known feature interactions (e.g., education-occupation, marital-status-occupation). FM captures all pairwise interactions via low-rank embeddings, adding only O(d*K) parameters per feature value. With K=8, it's still fast at inference (14*8=112 multiply-adds per item) and should beat linear models.
Failure signal: If dev accuracy <= 0.8524 (label-linear) or inference cost > 2x label-linear, the family does not meet the goal.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Model: logit = w0 + sum_j w_j[x_j] + 0.5 * sum_k ( (sum_j v_{j,k}[x_j])^2 - sum_j v_{j,k}[x_j]^2 )
Embedding dimension K=8. Adam lr=0.03, L2=1e-4, batch=64, epochs=30. Hyperparameters chosen on last 20% of train.in.
Weights format: magic 0x464D3130, d, K, C, then float32 arrays: w0, w[d*256], v[d*256*K]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED = 1
LR_GRID = [0.01, 0.03]
L2_GRID = [1e-4, 1e-3]
BATCH_GRID = [64, 256]
EPOCHS = 30
MAGIC = 0x464D3130  # 'FM10'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict_logit(X, w0, w, v, d, K):
    n = X.shape[0]
    logit = np.full(n, w0, dtype=np.float32)
    # linear term
    for j in range(d):
        logit += w[j * 256 + X[:, j]]
    # pairwise term via (sum v)^2 - sum v^2
    sum_v = np.zeros((n, K), dtype=np.float32)
    sum_v2 = np.zeros((n, K), dtype=np.float32)
    for j in range(d):
        idx = X[:, j]
        vj = v[j * 256 * K:(j + 1) * 256 * K].reshape(256, K)
        sum_v += vj[idx]
        sum_v2 += vj[idx] ** 2
    logit += 0.5 * np.sum(sum_v ** 2 - sum_v2, axis=1)
    return logit

def train_fm(X, y, d, K, lr, l2, batch, epochs, rng, observed_vals):
    n = X.shape[0]
    # parameters
    w0 = 0.0
    w = np.zeros(d * 256, dtype=np.float32)
    v = np.zeros(d * 256 * K, dtype=np.float32)
    # initialize observed values with small random
    for j in range(d):
        base = j * 256 * K
        for val in observed_vals[j]:
            v[base + val * K:base + (val + 1) * K] = rng.normal(0, 0.01, K).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)
            # forward
            logit = np.full(bsz, w0, dtype=np.float32)
            for j in range(d):
                logit += w[j * 256 + xb[:, j]]
            sum_v = np.zeros((bsz, K), dtype=np.float32)
            sum_v2 = np.zeros((bsz, K), dtype=np.float32)
            v_used = []
            for j in range(d):
                base = j * 256 * K
                vj = v[base:base + 256 * K].reshape(256, K)
                v_used.append(vj)
                vals = xb[:, j]
                sum_v += vj[vals]
                sum_v2 += vj[vals] ** 2
            logit += 0.5 * np.sum(sum_v ** 2 - sum_v2, axis=1)
            # loss gradient
            p = sigmoid(logit)
            grad = (p - yb).astype(np.float32)  # dL/dlogit
            # gradients
            gw0 = grad.mean()
            gw = np.zeros_like(w)
            gv = np.zeros_like(v)
            for j in range(d):
                vals = xb[:, j]
                # linear weight gradient
                for c in range(256):
                    mask = (vals == c)
                    if np.any(mask):
                        gw[j * 256 + c] = grad[mask].mean()
                # embedding gradient
                base = j * 256 * K
                vj = v_used[j]
                # dL/dv_{j,k}[x_j] = grad * (sum_{l!=j} v_l[x_l])_k
                # = grad * (sum_v - vj[vals])_k
                diff = sum_v - vj[vals]
                for c in range(256):
                    mask = (vals == c)
                    if np.any(mask):
                        gv[base + c * K:base + (c + 1) * K] = (grad[mask][:, None] * diff[mask]).mean(axis=0)
            # L2 regularization
            gw += l2 * w
            gv += l2 * v
            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (math.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= lr * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            # v
            m_v = beta1 * m_v + (1 - beta1) * gv
            v_v = beta2 * v_v + (1 - beta2) * gv * gv
            v -= lr * (m_v / (1 - beta1 ** t)) / (np.sqrt(v_v / (1 - beta2 ** t)) + eps)
    return w0, w, v

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    # grid search on tuning split
    best_acc = -1.0
    best_params = None
    best_w0 = best_w = best_v = None
    for lr in LR_GRID:
        for l2 in L2_GRID:
            for batch in BATCH_GRID:
                rng = np.random.default_rng(SEED)
                # observed values in train split only
                observed = [set(X[:split, j]) for j in range(d)]
                w0, w, v = train_fm(X[:split], y[:split], d, K, lr, l2, batch, EPOCHS, rng, observed)
                # evaluate on tune split
                logit = predict_logit(X[split:], w0, w, v, d, K)
                pred = (logit > 0).astype(np.int64)
                acc = (pred == y[split:]).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = (lr, l2, batch)
                    best_w0, best_w, best_v = w0, w.copy(), v.copy()
    # final training on all data
    rng = np.random.default_rng(SEED)
    observed_all = [set(X[:, j]) for j in range(d)]
    lr, l2, batch = best_params
    w0, w, v = train_fm(X, y, d, K, lr, l2, batch, EPOCHS, rng, observed_all)
    # write weights
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0 or 1).
 * Weights format: magic 0x464D3130, int32 d, K, C; float32 w0, w[d*256], v[d*256*K]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 8 || nw != 16 + 4L * (1 + d * 256 + d * 256 * K)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(K * sizeof(float));
    float *sum_v2 = malloc(K * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = w0;
        for (int j = 0; j < d; j++) logit += w[j * 256 + xi[j]];
        memset(sum_v, 0, K * sizeof(float));
        memset(sum_v2, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *vj = v + (j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = vj[kk];
                sum_v[kk] += val;
                sum_v2[kk] += val * val;
            }
        }
        for (int kk = 0; kk < K; kk++) logit += 0.5f * (sum_v[kk] * sum_v[kk] - sum_v2[kk]);
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_v); free(sum_v2);
    return 0;
}
```