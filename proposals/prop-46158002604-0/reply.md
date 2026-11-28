```hypothesis
Family: label-fm (factorization machine). 
Why: Adult has strong pairwise feature interactions (e.g., education × occupation) that linear models miss. FMs model all pairwise interactions via low-rank embeddings, adding only O(d·K) parameters (d=14, K=8 → ~3k params) and O(d·K) inference cost. 
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear by >2×, the family does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each column as a field with 256 values; learns bias, linear weights, and K-dim embeddings per field-value.
Trains with Adam on binary cross-entropy, L2 on weights/embeddings. Hyperparameters chosen on last 20% of train.in."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000

# Hyperparameter grid for tuning split
K_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3]
L2_CHOICES = [1e-4, 1e-3]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(X, w0, w, V, K):
    """X: (n, d) uint8 indices. Returns scores (n,)."""
    n, d = X.shape
    # linear part
    lin = w0 + np.sum(w[np.arange(d), X.T], axis=0)  # (n,)
    # interaction part: 0.5 * sum_k ( (sum_i V_i[k])^2 - sum_i V_i[k]^2 )
    V_x = V[np.arange(d)[:, None], X.T]  # (d, n, K)
    S = V_x.sum(axis=0)                  # (n, K)
    sum_sq = (V_x ** 2).sum(axis=0)      # (n, K)
    inter = 0.5 * np.sum(S * S - sum_sq, axis=1)  # (n,)
    return lin + inter

def fm_backward(X, y, w0, w, V, K, scores, l2):
    """Returns gradients for w0, w, V."""
    n, d = X.shape
    p = 1.0 / (1.0 + np.exp(-scores))
    grad_score = (p - y) / n  # (n,)

    # w0 gradient
    gw0 = grad_score.sum()

    # w gradient: (d, 256)
    gw = np.zeros_like(w)
    for i in range(d):
        np.add.at(gw[i], X[:, i], grad_score)

    # V gradient: (d, 256, K)
    V_x = V[np.arange(d)[:, None], X.T]  # (d, n, K)
    S = V_x.sum(axis=0)                  # (n, K)
    # dL/dV_i[k] = (S_k - V_i[k]) * grad_score
    gV = np.zeros_like(V)
    for i in range(d):
        # (S - V_x[i]) shape (n, K); multiply by grad_score (n,) -> (n, K)
        diff = (S - V_x[i]) * grad_score[:, None]
        np.add.at(gV[i], X[:, i], diff)

    # L2 regularization gradients
    gw += l2 * w
    gV += l2 * V
    return gw0, gw, gV

def train_fm(X_tr, y_tr, X_val, y_val, K, lr, l2, max_epochs):
    """Trains FM on training set, returns best (w0, w, V) by validation accuracy."""
    n, d = X_tr.shape
    rng = np.random.default_rng(SEED)
    # Initialize
    w0 = 0.0
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K)) * np.sqrt(2.0 / K)).astype(np.float32)

    params = [w0, w, V]
    # Adam state
    m = [0.0, np.zeros_like(w), np.zeros_like(V)]
    v = [0.0, np.zeros_like(w), np.zeros_like(V)]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_acc = -1.0
    best_params = None

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            scores = fm_forward(Xb, w0, w, V, K)
            gw0, gw, gV = fm_backward(Xb, yb, w0, w, V, K, scores, l2)

            t += 1
            # Update w0
            m[0] = beta1 * m[0] + (1 - beta1) * gw0
            v[0] = beta2 * v[0] + (1 - beta2) * (gw0 * gw0)
            m0_hat = m[0] / (1 - beta1 ** t)
            v0_hat = v[0] / (1 - beta2 ** t)
            w0 -= lr * m0_hat / (np.sqrt(v0_hat) + eps)

            # Update w
            m[1] = beta1 * m[1] + (1 - beta1) * gw
            v[1] = beta2 * v[1] + (1 - beta2) * (gw * gw)
            m1_hat = m[1] / (1 - beta1 ** t)
            v1_hat = v[1] / (1 - beta2 ** t)
            w -= lr * m1_hat / (np.sqrt(v1_hat) + eps)

            # Update V
            m[2] = beta1 * m[2] + (1 - beta1) * gV
            v[2] = beta2 * v[2] + (1 - beta2) * (gV * gV)
            m2_hat = m[2] / (1 - beta1 ** t)
            v2_hat = v[2] / (1 - beta2 ** t)
            V -= lr * m2_hat / (np.sqrt(v2_hat) + eps)

        # Validation accuracy
        val_scores = fm_forward(X_val, w0, w, V, K)
        val_pred = (val_scores > 0).astype(np.int64)
        acc = np.mean(val_pred == y_val)
        if acc > best_acc:
            best_acc = acc
            best_params = (w0, w.copy(), V.copy())

    return best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    best_acc = -1.0
    best_hp = None
    best_params = None

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                params = train_fm(X_tr, y_tr, X_val, y_val, K, lr, l2, max_epochs=50)
                w0, w, V = params
                val_scores = fm_forward(X_val, w0, w, V, K)
                val_pred = (val_scores > 0).astype(np.int64)
                acc = np.mean(val_pred == y_val)
                if acc > best_acc:
                    best_acc = acc
                    best_hp = (K, lr, l2)
                    best_params = params

    print(f"Best hyperparams: K={best_hp[0]}, lr={best_hp[1]}, l2={best_hp[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters
    K, lr, l2 = best_hp
    w0, w, V = train_fm(X, y, X_val, y_val, K, lr, l2, MAX_EPOCHS)  # reuse val for early stopping? We'll just train full.
    # Actually train_fm returns best on val; we want final model on all data. Let's do one more pass without early stopping.
    # Simpler: retrain on full data for fixed epochs (no early stopping) using best hyperparams.
    n_full = X.shape[0]
    rng = np.random.default_rng(SEED)
    w0 = 0.0
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K)) * np.sqrt(2.0 / K)).astype(np.float32)

    m = [0.0, np.zeros_like(w), np.zeros_like(V)]
    v = [0.0, np.zeros_like(w), np.zeros_like(V)]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            scores = fm_forward(Xb, w0, w, V, K)
            gw0, gw, gV = fm_backward(Xb, yb, w0, w, V, K, scores, l2)

            t += 1
            m[0] = beta1 * m[0] + (1 - beta1) * gw0
            v[0] = beta2 * v[0] + (1 - beta2) * (gw0 * gw0)
            m0_hat = m[0] / (1 - beta1 ** t)
            v0_hat = v[0] / (1 - beta2 ** t)
            w0 -= lr * m0_hat / (np.sqrt(v0_hat) + eps)

            m[1] = beta1 * m[1] + (1 - beta1) * gw
            v[1] = beta2 * v[1] + (1 - beta2) * (gw * gw)
            m1_hat = m[1] / (1 - beta1 ** t)
            v1_hat = v[1] / (1 - beta2 ** t)
            w -= lr * m1_hat / (np.sqrt(v1_hat) + eps)

            m[2] = beta1 * m[2] + (1 - beta1) * gV
            v[2] = beta2 * v[2] + (1 - beta2) * (gV * gV)
            m2_hat = m[2] / (1 - beta1 ** t)
            v2_hat = v[2] / (1 - beta2 ** t)
            V -= lr * m2_hat / (np.sqrt(v2_hat) + eps)

    # Write weights.bin
    # Magic: 0x464D31 ("FM1"), d, K, C (2 classes, but we output binary 0/1)
    magic = 0x464D31
    C = 2
    header = struct.pack("<4i", magic, d, K, C)
    # w0 (float32), w (d*256 float32), V (d*256*K float32)
    body = struct.pack("<f", w0) + w.astype("<f4").tobytes() + V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (field indices 0..255). Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || K > 256 || d < 1 || d > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int kk = 0; kk < K; kk++) { S[kk] = 0.0f; sum_sq[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            linear += w[(long)j * 256 + v];
            const float *Vj = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = Vj[kk];
                S[kk] += val;
                sum_sq[kk] += val * val;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += S[kk] * S[kk] - sum_sq[kk];
        inter *= 0.5f;

        float score = linear + inter;
        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char cls = prob > 0.5f ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```