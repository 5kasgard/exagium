```hypothesis
Family: label-fm (factorization machine with low-rank embeddings).
Adult data has many categorical features; pairwise interactions matter.
FM captures interactions in O(d*k) time, beating linear accuracy while staying faster than MLP.
If dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 2325 (label-mlp16), the model fails its goal.
```

```python
"""Factorization Machine for tabular label tasks (binary).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each byte feature as a categorical index (0-255). Uses shared embeddings for pairwise interactions.
Tunes embedding size k and learning rate on last 20% of train.in, then trains on all data.
Weights format: magic 0x464D4C31, int32 d, int32 k, float32 bias, float32 linear[d*256], float32 emb[d*256*k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning
K_CHOICES = [2, 4, 8, 16]
LR_CHOICES = [1e-3, 3e-3]
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_fm(X, y, k, lr, l2, epochs, batch, seed):
    """Train binary FM with Adam. Returns (bias, linear, embeddings)."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    bias = np.float32(0.0)
    linear = rng.uniform(-0.01, 0.01, size=(d, 256)).astype(np.float32)
    embeddings = rng.uniform(-0.01, 0.01, size=(d, 256, k)).astype(np.float32)

    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(embeddings)
    v_emb = np.zeros_like(embeddings)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            lin_scores = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (B,)
            emb = embeddings[np.arange(d), xb.T].transpose(1, 0, 2)         # (B, d, k)
            sum_emb = np.sum(emb, axis=1)                                   # (B, k)
            sum_sq_emb = np.sum(emb**2, axis=1)                             # (B, k)
            inter = 0.5 * np.sum(sum_emb**2 - sum_sq_emb, axis=1)           # (B,)
            scores = lin_scores + inter

            # Logistic loss gradient
            sig = 1.0 / (1.0 + np.exp(-scores))
            grad_score = (sig - yb).astype(np.float32)                      # (B,)

            # Gradients
            grad_bias = grad_score.mean() + l2 * bias

            grad_linear = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                grad_linear[j] = np.bincount(vals, weights=grad_score, minlength=256) / B + l2 * linear[j]

            grad_emb = np.zeros_like(embeddings)
            for j in range(d):
                vals = xb[:, j]
                contrib = grad_score[:, None] * (sum_emb - emb[:, j, :])    # (B, k)
                for f in range(k):
                    grad_emb[j, :, f] = np.bincount(vals, weights=contrib[:, f], minlength=256) / B + l2 * embeddings[j, :, f]

            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1 - beta2) * grad_bias * grad_bias
            bias -= lr * (m_bias / (1 - beta1**t)) / (np.sqrt(v_bias / (1 - beta2**t)) + eps)

            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * grad_linear
            v_lin = beta2 * v_lin + (1 - beta2) * grad_linear * grad_linear
            linear -= lr * (m_lin / (1 - beta1**t)) / (np.sqrt(v_lin / (1 - beta2**t)) + eps)

            # embeddings
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * grad_emb * grad_emb
            embeddings -= lr * (m_emb / (1 - beta1**t)) / (np.sqrt(v_emb / (1 - beta2**t)) + eps)

    return bias, linear, embeddings


def predict_fm(X, bias, linear, embeddings):
    """Return predicted class (0/1) for each row."""
    n, d = X.shape
    k = embeddings.shape[2]
    lin_scores = bias + np.sum(linear[np.arange(d), X.T], axis=0)
    emb = embeddings[np.arange(d), X.T].transpose(1, 0, 2)
    sum_emb = np.sum(emb, axis=1)
    sum_sq_emb = np.sum(emb**2, axis=1)
    inter = 0.5 * np.sum(sum_emb**2 - sum_sq_emb, axis=1)
    scores = lin_scores + inter
    return (scores > 0).astype(np.int64)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    # Hyperparameter search
    best_acc = -1.0
    best_params = None
    combo_idx = 0
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            seed = SEED + combo_idx
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, len(X_train))))
            bias, linear, emb = train_fm(X_train, y_train, k, lr, L2, epochs, BATCH, seed)
            pred = predict_fm(X_tune, bias, linear, emb)
            acc = (pred == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (k, lr, seed)
            combo_idx += 1

    # Retrain on full data with best params
    k, lr, seed = best_params
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    bias, linear, emb = train_fm(X, y, k, lr, L2, epochs, BATCH, seed)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<3i", magic, d, k)
    w = bytearray()
    w += header
    w += struct.pack("<f", bias)
    w += linear.astype("<f4").tobytes()
    w += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(w)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for binary label tasks.
 * ./infer weights.bin in.rec N
 * Weights format: magic 0x464D4C31, int32 d, int32 k, float bias, float linear[d*256], float emb[d*256*k].
 * Each input record is d bytes. Output is one byte: class index (0 or 1). */
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
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2];
    if (d <= 0 || k <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 12 + 4 * (1 + (size_t)d * 256 + (size_t)d * 256 * k);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(wb + 12);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (size_t)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = calloc(k, sizeof(float));
    float *sum_sq_emb = calloc(k, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float score = bias;
        memset(sum_emb, 0, k * sizeof(float));
        memset(sum_sq_emb, 0, k * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += linear[(size_t)j * 256 + v];
            const float *e = emb + ((size_t)j * 256 + v) * k;
            for (int f = 0; f < k; f++) {
                float val = e[f];
                sum_emb[f] += val;
                sum_sq_emb[f] += val * val;
            }
        }
        for (int f = 0; f < k; f++) {
            score += 0.5f * (sum_emb[f] * sum_emb[f] - sum_sq_emb[f]);
        }
        unsigned char cls = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```