```hypothesis
Factorization Machine (FM) with efficient inference: captures pairwise feature interactions
via low-rank embeddings (k=8) while keeping inference cost near linear models. Adult has
mixed ordinal/categorical features where interactions (e.g., education×occupation) matter.
FM's O(d·k) inference (d=14, k=8) is ~10× cheaper than label-mlp16. If dev accuracy
does not exceed label-linear (0.8524) the family is not justified.
```

```python
"""Factorization Machine for tabular classification (FM family).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Encodes each of the 14 byte features as a categorical index (0-255).
Model: logit = w0 + sum_j w_j[x_j] + 0.5*(||sum_j v_j[x_j]||^2 - sum_j ||v_j[x_j]||^2)
Binary cross-entropy, Adam lr=0.01, L2=1e-4, batch=64, epochs=10, k=8.
Weights file: magic 0x464D4231, d, k, C=2, w0, w[d][256], v[d][256][k], v_sq[d][256]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 10
K = 8

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    y_bytes = b"".join(records(sys.argv[3]))
    d = card["input"]["bytes"]
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.frombuffer(y_bytes, dtype=np.uint8).astype(np.int64)
    n = X.shape[0]
    assert n == y.shape[0]

    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]

    rng = np.random.default_rng(SEED)
    # parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_v = np.zeros_like(v)
    v_v = np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(indices):
        # indices: (B, d)
        B = indices.shape[0]
        # linear part
        lin = w0 + w[np.arange(d), indices.T].sum(axis=0)  # (B,)
        # interaction part
        v_batch = v[np.arange(d)[:, None], indices.T]      # (d, B, K) -> (B, d, K)
        v_batch = v_batch.transpose(1, 0, 2)                # (B, d, K)
        sum_v = v_batch.sum(axis=1)                         # (B, K)
        v_sq_batch = (v_batch ** 2).sum(axis=2).sum(axis=1) # (B,)
        inter = 0.5 * ((sum_v ** 2).sum(axis=1) - v_sq_batch)
        return lin + inter

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for epoch in range(EPOCHS):
        order = rng.permutation(split)
        for s in range(0, split, BATCH):
            idx = order[s:s + BATCH]
            batch_idx = X_train[idx]          # (B, d)
            batch_y = y_train[idx]            # (B,)
            B = batch_idx.shape[0]

            # forward
            v_batch = v[np.arange(d)[:, None], batch_idx.T].transpose(1, 0, 2)  # (B, d, K)
            sum_v = v_batch.sum(axis=1)                         # (B, K)
            v_sq_batch = (v_batch ** 2).sum(axis=2).sum(axis=1) # (B,)
            lin = w0 + w[np.arange(d), batch_idx.T].sum(axis=0)  # (B,)
            inter = 0.5 * ((sum_v ** 2).sum(axis=1) - v_sq_batch)
            logit = lin + inter
            p = sigmoid(logit)
            # gradient of BCE w.r.t logit
            g = (p - batch_y).astype(np.float32)                # (B,)

            # gradients
            gw0 = g.sum()
            # w gradient: for each feature j, scatter-add g to w[j, idx]
            gw = np.zeros_like(w)
            for j in range(d):
                np.add.at(gw[j], batch_idx[:, j], g)
            # v gradient: for each feature j, each sample i, grad = g[i] * (sum_v[i] - v_batch[i, j])
            gv = np.zeros_like(v)
            for j in range(d):
                diff = sum_v - v_batch[:, j, :]                 # (B, K)
                np.add.at(gv[j], batch_idx[:, j], (g[:, None] * diff).astype(np.float32))

            # L2 regularization
            gw += 2 * L2 * w
            gv += 2 * L2 * v

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (gw0 * gw0)
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * (gw * gw)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # v
            m_v = beta1 * m_v + (1 - beta1) * gv
            v_v = beta2 * v_v + (1 - beta2) * (gv * gv)
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # train on full data
    X_full, y_full = X, y
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_idx = X_full[idx]
            batch_y = y_full[idx]
            B = batch_idx.shape[0]

            v_batch = v[np.arange(d)[:, None], batch_idx.T].transpose(1, 0, 2)
            sum_v = v_batch.sum(axis=1)
            v_sq_batch = (v_batch ** 2).sum(axis=2).sum(axis=1)
            lin = w0 + w[np.arange(d), batch_idx.T].sum(axis=0)
            inter = 0.5 * ((sum_v ** 2).sum(axis=1) - v_sq_batch)
            logit = lin + inter
            p = sigmoid(logit)
            g = (p - batch_y).astype(np.float32)

            gw0 = g.sum()
            gw = np.zeros_like(w)
            for j in range(d):
                np.add.at(gw[j], batch_idx[:, j], g)
            gv = np.zeros_like(v)
            for j in range(d):
                diff = sum_v - v_batch[:, j, :]
                np.add.at(gv[j], batch_idx[:, j], (g[:, None] * diff).astype(np.float32))

            gw += 2 * L2 * w
            gv += 2 * L2 * v

            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (gw0 * gw0)
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * (gw * gw)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            m_v = beta1 * m_v + (1 - beta1) * gv
            v_v = beta2 * v_v + (1 - beta2) * (gv * gv)
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # precompute v_sq for fast inference
    v_sq = (v ** 2).sum(axis=2).astype(np.float32)  # (d, 256)

    # write weights
    MAGIC = 0x464D4231
    out = struct.pack("<4i", MAGIC, d, K, 2)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    out += v_sq.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=14); output one byte: class 0 (<=50K) or 1 (>50K).
 * Weights format: magic 0x464D4231, int32 d, int32 k, int32 C=2; float32 w0;
 * float32 w[d][256]; float32 v[d][256][k]; float32 v_sq[d][256]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || d > 255 || K < 1 || K > 255) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K + 4L * d * 256;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;
    const float *w = p; p += d * 256;
    const float *v = p; p += d * 256 * K;
    const float *v_sq = p; p += d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = calloc(K, sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = w0;
        float v_sq_sum = 0.0f;
        memset(sum_v, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            logit += w[(long)j * 256 + val];
            const float *vj = v + ((long)j * 256 + val) * K;
            for (int u = 0; u < K; u++) sum_v[u] += vj[u];
            v_sq_sum += v_sq[(long)j * 256 + val];
        }
        float sum_v_sq = 0.0f;
        for (int u = 0; u < K; u++) sum_v_sq += sum_v[u] * sum_v[u];
        logit += 0.5f * (sum_v_sq - v_sq_sum);
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_v);
    return 0;
}
```