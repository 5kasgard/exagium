```hypothesis
Family: label-fm (Factorization Machine with pairwise interactions).  
Adult has mixed categorical/ordinal features where pairwise interactions (e.g., education×occupation) are predictive.  
FM captures these with low-rank embeddings, beating linear models at similar parameter count.  
If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each byte value (0-255) per feature with a bias and a K-dim embedding.
Trains with Adam (lr=3e-3, L2=1e-4, batch=64) for min(100, ceil(600000/n)) epochs.
Tunes embedding size K on last 20% of training data, then retrains on all data.
Exports: magic 0x464D4C31, d, K, C; float32 bias, linear[d*256], embed[d*256*K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_K = [4, 8, 16]
TUNE_EPOCHS = 5  # quick epochs for tuning

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_logits(X, b, w, e, d, K):
    """X: (n, d) uint8. Returns logits (n,)."""
    n = X.shape[0]
    linear = np.full(n, b, dtype=np.float32)
    sum_emb = np.zeros((n, K), dtype=np.float32)
    sum_sq_emb = np.zeros((n, K), dtype=np.float32)
    for j in range(d):
        idx = X[:, j]
        linear += w[j, idx]
        emb = e[j, idx]          # (n, K)
        sum_emb += emb
        sum_sq_emb += emb * emb
    interaction = 0.5 * (np.sum(sum_emb * sum_emb, axis=1) - np.sum(sum_sq_emb, axis=1))
    return linear + interaction

def train_fm(X, y, d, K, epochs, rng):
    """Returns (b, w, e) trained on (X, y)."""
    n = X.shape[0]
    # init
    b = np.float32(0.0)
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    e = rng.normal(0, 1.0 / math.sqrt(K), (d, 256, K)).astype(np.float32)
    # Adam state
    mb, vb = np.float32(0.0), np.float32(0.0)
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    me, ve = np.zeros_like(e), np.zeros_like(e)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(idx)
            # forward
            linear = np.full(bs, b, dtype=np.float32)
            sum_emb = np.zeros((bs, K), dtype=np.float32)
            sum_sq_emb = np.zeros((bs, K), dtype=np.float32)
            for j in range(d):
                xj = xb[:, j]
                linear += w[j, xj]
                emb = e[j, xj]
                sum_emb += emb
                sum_sq_emb += emb * emb
            interaction = 0.5 * (np.sum(sum_emb * sum_emb, axis=1) - np.sum(sum_sq_emb, axis=1))
            logits = linear + interaction
            p = sigmoid(logits)
            g = (p - yb).astype(np.float32)  # (bs,)
            # gradients
            db = np.mean(g)
            dw = np.zeros_like(w)
            de = np.zeros_like(e)
            for j in range(d):
                xj = xb[:, j]
                # linear weight gradients
                np.add.at(dw[j], xj, g)
                # embedding gradients: g_i * (sum_emb_i - e_j)
                diff = g[:, None] * (sum_emb - e[j, xj])  # (bs, K)
                np.add.at(de[j], xj, diff)
            # L2 regularization
            dw += L2 * w
            de += L2 * e
            # Adam updates
            t += 1
            # bias
            mb = beta1 * mb + (1 - beta1) * db
            vb = beta2 * vb + (1 - beta2) * db * db
            mhat = mb / (1 - beta1 ** t)
            vhat = vb / (1 - beta2 ** t)
            b -= LR * mhat / (np.sqrt(vhat) + eps)
            # linear weights
            mw = beta1 * mw + (1 - beta1) * dw
            vw = beta2 * vw + (1 - beta2) * dw * dw
            mhat = mw / (1 - beta1 ** t)
            vhat = vw / (1 - beta2 ** t)
            w -= LR * mhat / (np.sqrt(vhat) + eps)
            # embeddings
            me = beta1 * me + (1 - beta1) * de
            ve = beta2 * ve + (1 - beta2) * de * de
            mhat = me / (1 - beta1 ** t)
            vhat = ve / (1 - beta2 ** t)
            e -= LR * mhat / (np.sqrt(vhat) + eps)
    return b, w, e

def evaluate_acc(X, y, b, w, e, d, K):
    logits = compute_logits(X, b, w, e, d, K)
    pred = (logits > 0).astype(np.int64)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]
    # split last 20% for tuning
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]
    rng = np.random.default_rng(SEED)
    # tune K
    full_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    best_K, best_acc = CANDIDATE_K[0], -1.0
    for K in CANDIDATE_K:
        b, w, e = train_fm(train_X, train_y, d, K, TUNE_EPOCHS, rng)
        acc = evaluate_acc(tune_X, tune_y, b, w, e, d, K)
        if acc > best_acc:
            best_acc, best_K = acc, K
    # final training on all data
    b, w, e = train_fm(X, y, d, best_K, full_epochs, rng)
    # save weights
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, best_K, C)
    # float32 little-endian
    bias_bytes = np.array([b], dtype="<f4").tobytes()
    w_bytes = w.astype("<f4").tobytes()
    e_bytes = e.astype("<f4").tobytes()
    with open(sys.argv[4], "wb") as f:
        f.write(header + bias_bytes + w_bytes + e_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0 or 1).
 * Weights format: magic 0x464D4C31, int32 d, K, C; float32 bias, linear[d*256], embed[d*256*K]. */
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
    if (C != 2 || d < 1 || d > 256 || K < 1 || K > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1L + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *embed = linear + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_emb[16], sum_sq_emb[16];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = bias[0];
        for (int j = 0; j < d; j++) {
            logit += linear[(long)j * 256 + xi[j]];
        }
        for (int kk = 0; kk < K; kk++) {
            sum_emb[kk] = 0.0f;
            sum_sq_emb[kk] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = e[kk];
                sum_emb[kk] += v;
                sum_sq_emb[kk] += v * v;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            inter += sum_emb[kk] * sum_emb[kk] - sum_sq_emb[kk];
        }
        logit += 0.5f * inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```