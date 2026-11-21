```hypothesis
Family: label-fm (Field-aware Factorization Machine). It models pairwise feature interactions via low-rank embeddings per field-value, which is ideal for tabular data with mixed ordinal/categorical columns. It should beat label-linear by capturing interactions (e.g., education × occupation) while staying fast: inference is O(d²·K) with d=14, K=8 (~900 multiply-adds). If dev accuracy ≤ label-linear (0.8524) or instructions/item exceed label-linear by >2×, the approach fails.
```

```python
"""label-fm: Field-aware Factorization Machine for tabular classification.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Encodes each of the 14 input bytes as a field with 256 values. Learns bias, linear weights per field-value,
and an embedding vector (size K) per field-value. Binary logistic loss, Adam, L2 regularization.
Hyperparameters (K, lr, L2, epochs) chosen on last 20% of train.in (tuning split), then retrain on all.
Exports: magic 0x464D464D, d=14, K, C=2; float32 w0, W[d][256], V[d][256][K]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03, 0.1]
L2_CHOICES = [1e-4, 1e-3]
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_fm(X, y, K, lr, l2, rng, n_epochs):
    n, d = X.shape
    # Parameters
    w0 = np.float32(0.0)
    W = np.zeros((d, 256), dtype=np.float32)
    V = np.zeros((d, 256, K), dtype=np.float32)
    # Adam state
    m_w0 = v_w0 = np.float32(0.0)
    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)
            # Forward
            logit = w0 + W[np.arange(d), xb.T].sum(axis=0)  # (B,)
            # Pairwise interactions
            for i in range(d):
                vi = V[i, xb[:, i]]      # (B, K)
                for j in range(i+1, d):
                    vj = V[j, xb[:, j]]
                    logit += (vi * vj).sum(axis=1)
            p = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = (p - yb).astype(np.float32)  # (B,)
            # Gradients
            g_w0 = grad_logit.mean()
            g_W = np.zeros_like(W)
            g_V = np.zeros_like(V)
            # Total embedding per sample for efficient V gradient
            total_emb = np.zeros((B, K), dtype=np.float32)
            for i in range(d):
                total_emb += V[i, xb[:, i]]
            for i in range(d):
                xi = xb[:, i]
                np.add.at(g_W[i], xi, grad_logit)
                vi = V[i, xi]
                # Interaction gradient: grad_logit * (total_emb - vi)
                g_V[i, xi] += grad_logit[:, None] * (total_emb - vi)
            g_W /= B
            g_V /= B
            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * (g_w0 + l2 * w0)
            v_w0 = beta2 * v_w0 + (1 - beta2) * (g_w0 + l2 * w0) ** 2
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # W
            m_W = beta1 * m_W + (1 - beta1) * (g_W + l2 * W)
            v_W = beta2 * v_W + (1 - beta2) * (g_W + l2 * W) ** 2
            W -= lr * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * (g_V + l2 * V)
            v_V = beta2 * v_V + (1 - beta2) * (g_V + l2 * V) ** 2
            V -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)
    return w0, W, V

def evaluate(X, y, w0, W, V):
    n, d = X.shape
    logit = w0 + W[np.arange(d), X.T].sum(axis=0)
    for i in range(d):
        vi = V[i, X[:, i]]
        for j in range(i+1, d):
            vj = V[j, X[:, j]]
            logit += (vi * vj).sum(axis=1)
    pred = (logit > 0).astype(np.int64)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_model = None
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                w0, W, V = train_fm(X_tr, y_tr, K, lr, l2, rng, n_epochs)
                acc = evaluate(X_val, y_val, w0, W, V)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K, lr, l2)
                    best_model = (w0, W, V)
    # Retrain on full data with best params
    K, lr, l2 = best_params
    n_epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    w0, W, V = train_fm(X, y, K, lr, l2, rng, n_epochs_full)
    # Export
    d = X.shape[1]
    C = card["classes"]
    magic = 0x464D464D  # 'FMFM'
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output: 1 byte class (0/1).
 * Weights format: magic(0x464D464D), d, K, C; float32 w0, W[d][256], V[d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D464D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || C != 2 || K <= 0 || K > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = w0;
        // Linear part
        for (int f = 0; f < d; f++) logit += W[f * 256 + xi[f]];
        // Pairwise interactions
        for (int f1 = 0; f1 < d; f1++) {
            const float *v1 = V + ((long)f1 * 256 + xi[f1]) * K;
            for (int f2 = f1 + 1; f2 < d; f2++) {
                const float *v2 = V + ((long)f2 * 256 + xi[f2]) * K;
                float sum = 0.0f;
                for (int kk = 0; kk < K; kk++) sum += v1[kk] * v2[kk];
                logit += sum;
            }
        }
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```