```hypothesis
Family: label-fm (factorization machine with pairwise interactions).
Why: Adult has strong feature interactions (e.g., education × occupation); FM captures these with low-rank embeddings, beating linear models while staying fast (O(d*K) per item).
Failure: If dev accuracy ≤ 0.8524 (linear) or CPU cost > 2× linear, the model does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each byte value with a linear weight and a K-dim embedding.
Interaction: 0.5 * sum_k ( (sum_i v_{i,x_i,k})^2 - sum_i v_{i,x_i,k}^2 ).
Trains with Adam on cross-entropy; selects K, lr, epochs on last 20% tuning split."""
import json, math, struct, sys

import numpy as np

SEED = 1
GRID_K = [4, 8, 16]
GRID_LR = [0.01, 0.003]
GRID_EPOCHS = [10, 20]
BATCH = 64
L2 = 1e-5

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
    # linear term
    lin = w0 + np.sum(w[np.arange(d), X.T].T, axis=1)  # (n,)
    # interaction term
    sum_v = np.zeros((n, K), dtype=np.float32)
    sum_v_sq = np.zeros((n, K), dtype=np.float32)
    for j in range(d):
        v = V[j, X[:, j]]  # (n, K)
        sum_v += v
        sum_v_sq += v * v
    inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
    return lin + inter

def fm_gradients(X, y, logits, w0, w, V, K, L2):
    """Compute gradients for a batch. Returns (g_w0, g_w, g_V)."""
    n, d = X.shape
    p = sigmoid(logits)
    dz = p - y  # (n,)
    # w0
    g_w0 = np.mean(dz)
    # w
    g_w = np.zeros_like(w)
    for j in range(d):
        for val in range(256):
            mask = (X[:, j] == val)
            if np.any(mask):
                g_w[j, val] = np.mean(dz[mask])
    # V
    g_V = np.zeros_like(V)
    # precompute s_k = sum_i v_{i,x_i,k} for each example
    s = np.zeros((n, K), dtype=np.float32)
    for j in range(d):
        s += V[j, X[:, j]]
    for j in range(d):
        v_j = V[j, X[:, j]]  # (n, K)
        # grad for each k: dz * (s_k - v_{j,k})
        grad = dz[:, None] * (s - v_j)  # (n, K)
        for val in range(256):
            mask = (X[:, j] == val)
            if np.any(mask):
                g_V[j, val] = np.mean(grad[mask], axis=0)
    # L2 regularization
    g_w += L2 * w
    g_V += L2 * V
    return g_w0, g_w, g_V

def train_fm(X, y, K, lr, epochs, rng):
    n, d = X.shape
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
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            logits = fm_forward(Xb, w0, w, V, K)
            g_w0, g_w, g_V = fm_gradients(Xb, yb, logits, w0, w, V, K, L2)
            t += 1
            # Adam updates
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
    return w0, w, V

def evaluate(X, y, w0, w, V, K):
    logits = fm_forward(X, w0, w, V, K)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20%
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_model = None
    for K in GRID_K:
        for lr in GRID_LR:
            for epochs in GRID_EPOCHS:
                rng_i = np.random.default_rng(SEED + K * 1000 + int(lr * 10000) + epochs)
                w0, w, V = train_fm(X_tr, y_tr, K, lr, epochs, rng_i)
                acc = evaluate(X_va, y_va, w0, w, V, K)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K, lr, epochs)
                    best_model = (w0, w, V)
    print(f"Best tuning acc: {best_acc:.4f} params: {best_params}", file=sys.stderr)
    # retrain on full data with best params
    K, lr, epochs = best_params
    rng_full = np.random.default_rng(SEED + 999)
    w0, w, V = train_fm(X, y, K, lr, epochs, rng_full)
    # write weights.bin
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0 or 1).
 * Weights format: int32 magic(0x464D4C31), d, K, C; float32 w0; float32 w[d][256]; float32 V[d][256][K]. */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v_sq = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = w0;
        for (int kk = 0; kk < K; kk++) { sum_v[kk] = 0.0f; sum_v_sq[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += w[(long)j * 256 + v];
            const float *vj = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = vj[kk];
                sum_v[kk] += val;
                sum_v_sq[kk] += val * val;
            }
        }
        for (int kk = 0; kk < K; kk++) {
            logit += 0.5f * (sum_v[kk] * sum_v[kk] - sum_v_sq[kk]);
        }
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```