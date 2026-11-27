```hypothesis
A factorization machine (FM) with per-feature-value embeddings captures pairwise interactions
that a linear model misses, while staying lightweight: O(d·k) parameters and inference cost
for d=14 features and k=8. On this tabular task with mixed ordinal/categorical columns,
the FM should exceed label-linear's 0.8524 dev accuracy. If the FM does not beat label-linear
by >0.5% absolute accuracy, the interactions are not beneficial here.
```

```python
"""Factorization Machine for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Exports: magic 0x464D4144, d, k, w0, linear_weights[d*256], embeddings[d*256*k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
VAL_FRACTION = 0.2

# Hyperparameter grid for tuning on the last 20% of train.in
K_GRID = [4, 8, 16]
L2_GRID = [1e-4, 1e-3, 1e-2]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def load_data(card_path, in_path, key_path):
    card = json.load(open(card_path))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(in_path)), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(key_path)], dtype=np.int64)
    return X, y, d


def split_val(X, y, val_frac):
    n = X.shape[0]
    val_size = int(n * val_frac)
    # last 20% in file order
    return (X[:-val_size], y[:-val_size]), (X[-val_size:], y[-val_size:])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fm_forward(X, w0, lin_w, emb, k):
    """X: (n, d) uint8. Returns logits (n,)."""
    n, d = X.shape
    # flat indices for gathering: feature i uses rows i*256 .. (i+1)*256-1
    flat_idx = X + np.arange(d, dtype=np.uint16) * 256  # (n, d)
    # linear term
    lin = w0 + np.sum(lin_w[flat_idx], axis=1, dtype=np.float32)
    # embedding term: sum_v and sum_v_sq
    emb_gathered = emb[flat_idx]  # (n, d, k)
    sum_v = np.sum(emb_gathered, axis=1, dtype=np.float32)          # (n, k)
    sum_v_sq = np.sum(emb_gathered * emb_gathered, axis=1, dtype=np.float32)
    inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1, dtype=np.float32)
    return lin + inter


def fm_backward(X, y, logits, w0, lin_w, emb, k, l2):
    """Returns gradients for w0, lin_w, emb."""
    n, d = X.shape
    p = sigmoid(logits)
    err = (p - y).astype(np.float32)          # (n,)
    flat_idx = X + np.arange(d, dtype=np.uint16) * 256  # (n, d)
    # gradient for w0
    gw0 = np.mean(err)
    # gradient for linear weights
    glin_w = np.zeros_like(lin_w)
    np.add.at(lin_w, flat_idx.ravel(), err.repeat(d))  # but we need per-example per-feature
    # Instead, we accumulate per feature-value:
    for j in range(d):
        np.add.at(glin_w, flat_idx[:, j], err)
    glin_w /= n
    glin_w += l2 * lin_w
    # gradient for embeddings
    emb_gathered = emb[flat_idx]                # (n, d, k)
    sum_v = np.sum(emb_gathered, axis=1)        # (n, k)
    # d logit / d v_{i,val} = err[:, None] * (sum_v - v_{i,val})
    grad_emb = err[:, None, None] * (sum_v[:, None, :] - emb_gathered)  # (n, d, k)
    gemb = np.zeros_like(emb)
    for j in range(d):
        np.add.at(gemb, flat_idx[:, j], grad_emb[:, j])
    gemb /= n
    gemb += l2 * emb
    return gw0, glin_w, gemb


def train_fm(X, y, d, k, l2, lr, batch, max_epochs, samples, rng):
    n = X.shape[0]
    max_epochs = min(max_epochs, math.ceil(samples / n))
    # parameters
    w0 = np.float32(0.0)
    lin_w = np.zeros(d * 256, dtype=np.float32)
    emb = rng.uniform(-0.01, 0.01, (d * 256, k)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(lin_w), np.zeros_like(lin_w)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            logits = fm_forward(xb, w0, lin_w, emb, k)
            gw0, glin_w, gemb = fm_backward(xb, yb, logits, w0, lin_w, emb, k, l2)
            t += 1
            # Adam updates
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            m_lin = beta1 * m_lin + (1 - beta1) * glin_w
            v_lin = beta2 * v_lin + (1 - beta2) * glin_w * glin_w
            lin_w -= lr * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)
            m_emb = beta1 * m_emb + (1 - beta1) * gemb
            v_emb = beta2 * v_emb + (1 - beta2) * gemb * gemb
            emb -= lr * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)
    return w0, lin_w, emb


def evaluate(X, y, w0, lin_w, emb, k):
    logits = fm_forward(X, w0, lin_w, emb, k)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card_path, in_path, key_path, out_path = sys.argv[1:]
    X, y, d = load_data(card_path, in_path, key_path)
    (X_tr, y_tr), (X_val, y_val) = split_val(X, y, VAL_FRACTION)
    rng = np.random.default_rng(SEED)

    best_acc = -1.0
    best_k = K_GRID[0]
    best_l2 = L2_GRID[0]
    for k in K_GRID:
        for l2 in L2_GRID:
            w0, lin_w, emb = train_fm(X_tr, y_tr, d, k, l2, LR, BATCH, MAX_EPOCHS, SAMPLES, rng)
            acc = evaluate(X_val, y_val, w0, lin_w, emb, k)
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_l2 = l2

    # Retrain on full data with best hyperparameters
    rng = np.random.default_rng(SEED)
    w0, lin_w, emb = train_fm(X, y, d, best_k, best_l2, LR, BATCH, MAX_EPOCHS, SAMPLES, rng)

    # Write weights.bin
    magic = 0x464D4144  # "FMAD"
    header = struct.pack("<4i", magic, d, best_k, 0)  # last int unused (C=2 implicit)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(np.float32(w0).tobytes())
        f.write(lin_w.astype("<f4").tobytes())
        f.write(emb.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (0 or 1).
 * Weights format: magic(4), d, k, _ (4 int32), w0 (float32),
 * linear_weights[d*256] (float32), embeddings[d*256*k] (float32), all LE. */
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
    if (nw < 16 || hd[0] != 0x464D4144) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2];
    if (d <= 0 || k <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 16 + 4 + (size_t)d * 256 * 4 + (size_t)d * 256 * k * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;
    const float *lin_w = p; p += d * 256;
    const float *emb = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = (float *)calloc(k, sizeof(float));
    float *sum_v_sq = (float *)calloc(k, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float linear = w0;
        memset(sum_v, 0, k * sizeof(float));
        memset(sum_v_sq, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            uint32_t idx = j * 256 + xi[j];
            linear += lin_w[idx];
            const float *v = emb + idx * k;
            for (int u = 0; u < k; u++) {
                float val = v[u];
                sum_v[u] += val;
                sum_v_sq[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < k; u++) {
            float sv = sum_v[u];
            inter += sv * sv - sum_v_sq[u];
        }
        inter *= 0.5f;
        float logit = linear + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_v);
    free(sum_v_sq);
    return 0;
}
```