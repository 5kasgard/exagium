```hypothesis
Factorization Machines (FM) model all pairwise feature interactions via low-rank embeddings, which is crucial for tabular data like Adult where combinations (e.g., education × occupation) are highly predictive. With embedding size K=8, the model adds only ~32k parameters (14 features × 256 values × (1+K)), keeping inference fast (O(d·K) per item). If dev accuracy does not exceed label-linear's 0.8524, the interactions are not helping or the model is underfitting.
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Treats all 14 columns as categorical (256 bins each).
Learns per-value linear weight and K-dim embedding. Binary cross-entropy, Adam, L2=1e-4.
Tunes K on last 20% of train.in, then trains on all data. Exports: magic 0x464D3132, d, K, C; float32 w0, w[d*256], v[d*256*K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, EPOCHS_TUNE, EPOCHS_FULL = 0.01, 1e-4, 256, 10, 30
K_CANDIDATES = [4, 8, 16]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fm_forward(X, w0, w, v, K):
    """X: (n, d) uint8. Returns scores (n,)."""
    n, d = X.shape
    linear = w0 + w[X].sum(axis=1)  # w indexed by X, sum over features
    # interaction: 0.5 * ( (sum v)^2 - sum v^2 ) per embedding dim
    sum_v = np.zeros((n, K), dtype=np.float32)
    sum_v2 = np.zeros((n, K), dtype=np.float32)
    for j in range(d):
        vj = v[j]  # (256, K)
        xj = X[:, j]
        sum_v += vj[xj]
        sum_v2 += vj[xj] ** 2
    inter = 0.5 * ((sum_v ** 2 - sum_v2).sum(axis=1))
    return linear + inter

def train_fm(X_tr, y_tr, X_val, y_val, K, rng, epochs):
    n, d = X_tr.shape
    w0 = 0.0
    w = np.zeros((d, 256), dtype=np.float32)
    v = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # forward
            linear = w0 + w[xb].sum(axis=(1,2))  # w indexed by xb (BATCH, d) -> (BATCH, d) -> sum over d
            # Actually w[xb] gives (BATCH, d) because xb is (BATCH, d) and w is (d, 256) -> need advanced indexing
            # Let's do it feature by feature for clarity and speed
            linear = np.full(len(yb), w0, dtype=np.float32)
            sum_v = np.zeros((len(yb), K), dtype=np.float32)
            sum_v2 = np.zeros((len(yb), K), dtype=np.float32)
            for j in range(d):
                wj = w[j]  # (256,)
                vj = v[j]  # (256, K)
                xbj = xb[:, j]
                linear += wj[xbj]
                sum_v += vj[xbj]
                sum_v2 += vj[xbj] ** 2
            inter = 0.5 * ((sum_v ** 2 - sum_v2).sum(axis=1))
            scores = linear + inter
            probs = sigmoid(scores)
            # gradient of binary cross-entropy: probs - yb
            grad = (probs - yb).astype(np.float32)  # (BATCH,)
            # gradients
            g_w0 = grad.mean()
            g_w = np.zeros_like(w)
            g_v = np.zeros_like(v)
            for j in range(d):
                xbj = xb[:, j]
                # g_w[j][val] += mean(grad * indicator)
                for val in range(256):
                    mask = (xbj == val)
                    if mask.any():
                        g_w[j, val] = grad[mask].mean()
                # g_v[j][val] += mean(grad * (sum_v_other * indicator))
                # sum_v_other = sum_v - vj[xbj]
                sum_v_other = sum_v - vj[xbj]
                for val in range(256):
                    mask = (xbj == val)
                    if mask.any():
                        g_v[j, val] = (grad[mask, None] * sum_v_other[mask]).mean(axis=0)
            # L2 regularization
            g_w += L2 * w
            g_v += L2 * v
            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            # v
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v
            v -= LR * (m_v / (1 - beta1 ** t)) / (np.sqrt(v_v / (1 - beta2 ** t)) + eps)
    # validation accuracy
    scores_val = fm_forward(X_val, w0, w, v, K)
    preds = (sigmoid(scores_val) > 0.5).astype(np.int64)
    acc = (preds == y_val).mean()
    return w0, w, v, acc

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = len(X)
    split = int(0.8 * n)
    idx = np.arange(n)
    tr_idx, val_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    rng = np.random.default_rng(SEED)
    best_K, best_acc = K_CANDIDATES[0], -1.0
    best_params = None
    for K in K_CANDIDATES:
        rng_k = np.random.default_rng(SEED + K)  # deterministic per K
        w0, w, v, acc = train_fm(X_tr, y_tr, X_val, y_val, K, rng_k, EPOCHS_TUNE)
        if acc > best_acc:
            best_acc, best_K = acc, K
            best_params = (w0, w, v)
    # retrain on full data with best_K
    rng_full = np.random.default_rng(SEED + best_K + 100)
    w0, w, v, _ = train_fm(X, y, X_val, y_val, best_K, rng_full, EPOCHS_FULL)

    # pack weights: magic, d, K, C; float32 w0, w[d*256], v[d*256*K]
    magic = 0x464D3132  # 'FM12'
    out = struct.pack("<4i", magic, d, best_K, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0/1). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x464D3132) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = (int)hd[1], K = (int)hd[2], C = (int)hd[3];
    if (C != 2 || d != 14 || K <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *pf = (float *)(wb + 16);
    float w0 = pf[0];
    float *w = pf + 1;                     // [d * 256]
    float *v = pf + 1 + d * 256;           // [d * 256 * K]
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(K * sizeof(float));
    float *sum_v2 = malloc(K * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear = w0;
        for (int kk = 0; kk < K; kk++) { sum_v[kk] = 0.0f; sum_v2[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            linear += w[j * 256 + val];
            const float *vj = v + (j * 256 + val) * K;
            for (int kk = 0; kk < K; kk++) {
                float vv = vj[kk];
                sum_v[kk] += vv;
                sum_v2[kk] += vv * vv;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += sum_v[kk] * sum_v[kk] - sum_v2[kk];
        inter *= 0.5f;
        float score = linear + inter;
        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char pred = prob > 0.5f ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```