```hypothesis
Factorization Machines (FM) capture pairwise feature interactions via low-rank embeddings, which are crucial for tabular data with categorical features. The adult dataset has known interactions (e.g., education × occupation). An FM with K=8 latent factors should beat logistic regression (label-linear) while keeping inference cost O(d·K) per example. If dev accuracy does not exceed 0.8524, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then retrained on all data.
Weights format: int32 magic 0x464D0001, d, K, C; float32 w0, w[d][256], V[d][256][K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
GRID = {"K": [4, 8, 16], "lr": [0.01, 0.03, 0.1], "l2": [1e-4, 1e-5]}
BATCH, GRID_EPOCHS, FINAL_EPOCHS = 64, 10, 20


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, w0, w, V, K):
    """X: (B, d) uint8. Returns logits (B,)"""
    B, d = X.shape
    lin = np.full(B, w0, dtype=np.float32)
    for j in range(d):
        lin += w[j, X[:, j]]
    inter = np.zeros(B, dtype=np.float32)
    for k in range(K):
        sum_v = np.zeros(B, dtype=np.float32)
        sum_v2 = np.zeros(B, dtype=np.float32)
        for j in range(d):
            v = V[j, X[:, j], k]
            sum_v += v
            sum_v2 += v * v
        inter += 0.5 * (sum_v * sum_v - sum_v2)
    return lin + inter


def train_fm(X_tr, y_tr, X_val, y_val, K, lr, l2, epochs, rng):
    n, d = X_tr.shape
    w0 = 0.0
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            B = len(yb)
            # Forward
            logits = fm_forward(Xb, w0, w, V, K)
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logit = (p - yb).astype(np.float32)  # (B,)
            # Gradients
            g_w0 = grad_logit.mean()
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)
            for j in range(d):
                xj = Xb[:, j]
                g_w[j, xj] += grad_logit
                for k in range(K):
                    v_j = V[j, xj, k]
                    sum_v = np.zeros(B, dtype=np.float32)
                    for jj in range(d):
                        sum_v += V[jj, Xb[:, jj], k]
                    g_V[j, xj, k] += grad_logit * (sum_v - v_j)
            g_w /= B
            g_V /= B
            # L2
            g_w += l2 * w
            g_V += l2 * V
            # Adam updates
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= lr * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)
    # Validation accuracy
    logits_val = fm_forward(X_val, w0, w, V, K)
    pred = (logits_val > 0).astype(np.int64)
    acc = (pred == y_val).mean()
    return acc, (w0, w, V)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    n = X.shape[0]
    n_val = n // 5
    n_tr = n - n_val
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr:], y[n_tr:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None
    for K in GRID["K"]:
        for lr in GRID["lr"]:
            for l2 in GRID["l2"]:
                cfg_rng = np.random.default_rng(SEED + hash((K, lr, l2)) % 2**32)
                acc, params = train_fm(X_tr, y_tr, X_val, y_val, K, lr, l2, GRID_EPOCHS, cfg_rng)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (K, lr, l2)
                    best_params = params
    print(f"Best config: K={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data
    K, lr, l2 = best_cfg
    final_rng = np.random.default_rng(SEED)
    _, final_params = train_fm(X, y, X[:1], y[:1], K, lr, l2, FINAL_EPOCHS, final_rng)
    w0, w, V = final_params
    # Save weights
    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array(w0, dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class (0 or 1). */
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
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D0001) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float lin = w0;
        for (int j = 0; j < d; j++) lin += w[(long)j * 256 + xi[j]];
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float sum_v = 0.0f, sum_v2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[((long)j * 256 + xi[j]) * K + kk];
                sum_v += v;
                sum_v2 += v * v;
            }
            inter += 0.5f * (sum_v * sum_v - sum_v2);
        }
        float logit = lin + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```