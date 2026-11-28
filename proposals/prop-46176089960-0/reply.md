```hypothesis
Family: label-fm (Factorization Machine with learned embeddings per feature value).
Why: Adult has strong pairwise interactions (e.g., education×occupation) that linear models miss. FM captures all pairwise interactions via low-rank embeddings, adding only d*256*k parameters (k=8 → ~29k floats) and ~91·k multiply-adds per item at inference. This should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at similar or lower CPU cost. If dev accuracy ≤ 0.8524, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the 14 bytes as a field with 256 possible values.
Learns a global bias, per-field-value linear weights, and per-field-value embeddings (dim k).
Trains with Adam on logistic loss, L2 regularization. Hyperparameters chosen on last 20% of train.in.
Exports: magic 0x464D0001, d, k, C; float32 w0, linear[d*256], embed[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
MAX_EPOCHS_FULL = 100
MAX_EPOCHS_TUNE = 20
SAMPLES_PER_EPOCH = 600000

# Hyperparameter grid (order fixed for determinism)
EMBED_DIMS = [4, 8, 16]
LRS = [1e-2, 3e-3, 1e-3]
L2S = [1e-4, 1e-5]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def train_fm(X, y, d, k, C, lr, l2, epochs, rng, w0_init, linear_init, embed_init):
    """Train FM on (X,y). Returns (w0, linear, embed)."""
    n = X.shape[0]
    w0 = w0_init.copy()
    linear = linear_init.copy()
    embed = embed_init.copy()
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)
            # Forward
            lin_part = w0 + np.sum(linear[xb, np.arange(d)], axis=1)  # linear[xb] shape (bs,d)
            # Efficient FM interaction: 0.5 * ( (sum_v)^2 - sum(v^2) )
            emb_x = embed[xb]  # (bs, d, k)
            sum_v = emb_x.sum(axis=1)          # (bs, k)
            sum_v_sq = (emb_x * emb_x).sum(axis=1)
            inter = 0.5 * ((sum_v * sum_v).sum(axis=1) - sum_v_sq.sum(axis=1))
            logit = lin_part + inter
            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)  # (bs,)
            # Gradients
            # w0
            g_w0 = grad.mean() + l2 * w0
            # linear: for each feature, add grad * 1
            g_lin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(g_lin, xb[:, j], grad)
            g_lin /= bs
            g_lin += l2 * linear
            # embed: for each feature, add grad * (sum_v - v_i)
            g_emb = np.zeros_like(embed)
            for j in range(d):
                v_j = embed[xb[:, j]]                 # (bs, k)
                factor = (sum_v - v_j) * grad[:, None]  # (bs, k)
                np.add.at(g_emb, xb[:, j], factor)
            g_emb /= bs
            g_emb += l2 * embed
            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (math.sqrt(v_w0 / (1 - beta2 ** t)) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= lr * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)
            # embed
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            embed -= lr * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)
    return w0, linear, embed


def predict_fm(X, d, k, w0, linear, embed):
    n = X.shape[0]
    lin_part = w0 + np.sum(linear[X, np.arange(d)], axis=1)
    emb_x = embed[X]  # (n, d, k)
    sum_v = emb_x.sum(axis=1)
    sum_v_sq = (emb_x * emb_x).sum(axis=1)
    inter = 0.5 * ((sum_v * sum_v).sum(axis=1) - sum_v_sq.sum(axis=1))
    logit = lin_part + inter
    return (logit > 0).astype(np.int64)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 14 and C == 2

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_acc = -1.0
    best_params = None
    best_model = None

    # Fixed initialization for each config (using a sub-rng per config would be ideal,
    # but we just continue the global rng sequence for determinism)
    for k in EMBED_DIMS:
        for lr in LRS:
            for l2 in L2S:
                # Initialize
                w0_init = np.array([0.0], dtype=np.float32)
                linear_init = np.zeros((d, 256), dtype=np.float32)
                embed_init = (rng.normal(0, 0.01, size=(d, 256, k))).astype(np.float32)
                # Train on tuning split
                w0, linear, embed = train_fm(X_train, y_train, d, k, C, lr, l2, MAX_EPOCHS_TUNE, rng,
                                             w0_init, linear_init, embed_init)
                pred = predict_fm(X_tune, d, k, w0, linear, embed)
                acc = accuracy(y_tune, pred)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (k, lr, l2)
                    best_model = (w0.copy(), linear.copy(), embed.copy())

    # Retrain on full data with best params
    k, lr, l2 = best_params
    w0_init = np.array([0.0], dtype=np.float32)
    linear_init = np.zeros((d, 256), dtype=np.float32)
    embed_init = (rng.normal(0, 0.01, size=(d, 256, k))).astype(np.float32)
    w0, linear, embed = train_fm(X, y, d, k, C, lr, l2, MAX_EPOCHS_FULL, rng,
                                 w0_init, linear_init, embed_init)

    # Pack weights: magic, d, k, C; float32 w0, linear[d*256], embed[d*256*k]
    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, k, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1], k = (int)hd[2], C = (int)hd[3];
    if (d != 14 || k <= 0 || C != 2) { fprintf(stderr, "bad header values\n"); return 3; }
    size_t expected = 16 + 4 + (size_t)d * 256 * 4 + (size_t)d * 256 * k * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expected %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;
    const float *linear = p; p += (size_t)d * 256;
    const float *embed = p; /* size d*256*k */

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = (float *)malloc(sizeof(float) * k);
    float *sum_v_sq = (float *)malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float lin = w0;
        memset(sum_v, 0, sizeof(float) * k);
        memset(sum_v_sq, 0, sizeof(float) * k);

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin += linear[(size_t)j * 256 + v];
            const float *e = embed + ((size_t)j * 256 + v) * k;
            for (int u = 0; u < k; u++) {
                float ev = e[u];
                sum_v[u] += ev;
                sum_v_sq[u] += ev * ev;
            }
        }

        float inter = 0.0f;
        for (int u = 0; u < k; u++) {
            inter += sum_v[u] * sum_v[u] - sum_v_sq[u];
        }
        inter *= 0.5f;

        float logit = lin + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sum_v);
    free(sum_v_sq);
    return 0;
}
```