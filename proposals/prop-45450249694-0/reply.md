```hypothesis
Propose a Factorization Machine with linear term (family label-fm). It captures pairwise feature interactions via low-rank embeddings (rank k), which is highly effective for tabular data with mixed categorical/numeric features. The model adds only O(d*k) inference cost per item (d=14, k=8) and should beat label-linear (0.8524) on dev accuracy. If dev accuracy does not exceed 0.855, the added complexity is not justified.
```

```python
"""Factorization Machine (FM) with linear term for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: each of the 14 input bytes (0-255) gets a linear weight per class and a k-dim embedding per class.
Training: Adam, softmax cross-entropy, L2 on all params. Hyperparameter k chosen on last 20% of train.in.
Weights format: magic 0x464D4C31, int32 d, k, C; float32 W_lin[d][256][C], V_fm[d][256][C][k]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K_CHOICES = [4, 8, 16]  # tuning grid


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, k, rng, epochs, batch, lr, l2, verbose=False):
    """Train FM on given data, return (W_lin, V_fm, val_acc)."""
    n, d = X.shape
    C = 2  # binary
    # Parameters
    W_lin = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    V_fm = rng.normal(0, 0.01, (d, 256, C, k)).astype(np.float32)
    # Adam state
    m_W = np.zeros_like(W_lin); v_W = np.zeros_like(W_lin)
    m_V = np.zeros_like(V_fm); v_V = np.zeros_like(V_fm)
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)
            # Forward
            lin_scores = np.zeros((B, C), np.float32)
            sum_v = np.zeros((B, C, k), np.float32)
            sum_v2 = np.zeros((B, C, k), np.float32)
            for j in range(d):
                v = xb[:, j]                 # (B,)
                lin_scores += W_lin[j, v, :] # (B, C)
                emb = V_fm[j, v, :, :]       # (B, C, k)
                sum_v += emb
                sum_v2 += emb * emb
            interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=2)  # (B, C)
            scores = lin_scores + interaction
            # Softmax cross-entropy
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            # Gradient on scores
            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B  # (B, C)
            # Backward
            grad_W = np.zeros_like(W_lin)
            grad_V = np.zeros_like(V_fm)
            for j in range(d):
                v = xb[:, j]
                # Linear grad
                np.add.at(grad_W[j], v, grad_scores)
                # FM grad: d(interaction)/d(emb) = sum_v - emb
                emb = V_fm[j, v, :, :]          # (B, C, k)
                g_emb = grad_scores[:, :, None] * (sum_v - emb)  # (B, C, k)
                np.add.at(grad_V[j], v, g_emb)
            # L2 regularization
            grad_W += l2 * W_lin
            grad_V += l2 * V_fm
            # Adam update
            t += 1
            for param, grad, m, v in [(W_lin, grad_W, m_W, v_W), (V_fm, grad_V, m_V, v_V)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W_lin, V_fm


def evaluate(X, y, W_lin, V_fm):
    n, d = X.shape
    C = 2
    k = V_fm.shape[3]
    correct = 0
    for i in range(n):
        xi = X[i]
        scores = np.zeros(C, np.float32)
        sum_v = np.zeros((C, k), np.float32)
        sum_v2 = np.zeros((C, k), np.float32)
        for j in range(d):
            v = xi[j]
            scores += W_lin[j, v, :]
            emb = V_fm[j, v, :, :]
            sum_v += emb
            sum_v2 += emb * emb
        interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
        scores += interaction
        pred = np.argmax(scores)
        if pred == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    # Hyperparameter search
    best_k = K_CHOICES[0]
    best_acc = -1.0
    for k in K_CHOICES:
        rng = np.random.default_rng(SEED)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        W_lin, V_fm = train_model(X_train, y_train, k, rng, epochs, BATCH, LR, L2)
        acc = evaluate(X_tune, y_tune, W_lin, V_fm)
        if acc > best_acc:
            best_acc = acc
            best_k = k
    # Retrain on full data with best_k
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W_lin, V_fm = train_model(X, y, best_k, rng, epochs, BATCH, LR, L2)
    # Write weights.bin
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D4C31, d, best_k, C)
    out += W_lin.astype("<f4").tobytes()
    out += V_fm.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (linear + pairwise FM) for label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (0 or 1).
 * Weights format: magic 0x464D4C31, int32 d, k, C; float32 W_lin[d][256][C], V_fm[d][256][C][k]. */
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
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k <= 0 || d <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * C + (long)d * 256 * C * k;
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W_lin = (const float *)(hd + 4);
    const float *V_fm = W_lin + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    // Temporary buffers for FM sums (C=2, k small)
    float sum_v[2][16];   // max k=16
    float sum_v2[2][16];
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float scores[2] = {0.0f, 0.0f};
        for (int c = 0; c < C; c++) for (int f = 0; f < k; f++) { sum_v[c][f] = 0.0f; sum_v2[c][f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *w = W_lin + ((long)j * 256 + v) * C;
            scores[0] += w[0];
            scores[1] += w[1];
            const float *emb = V_fm + (((long)j * 256 + v) * C + 0) * k;
            for (int f = 0; f < k; f++) {
                float val0 = emb[f];
                sum_v[0][f] += val0;
                sum_v2[0][f] += val0 * val0;
            }
            emb += C * k;
            for (int f = 0; f < k; f++) {
                float val1 = emb[f];
                sum_v[1][f] += val1;
                sum_v2[1][f] += val1 * val1;
            }
        }
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < k; f++) {
                float sv = sum_v[c][f];
                inter += sv * sv - sum_v2[c][f];
            }
            scores[c] += 0.5f * inter;
        }
        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```