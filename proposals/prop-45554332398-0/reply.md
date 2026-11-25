```hypothesis
Family: label-fm (factorization machine with per-value embeddings).
Why: Adult data has many categorical features where pairwise interactions (e.g., education × occupation) are known to be predictive. A degree-2 FM captures all pairwise interactions via low-rank embeddings, adding only O(d·k) parameters per feature value. With k=8 it stays tiny (≈30k floats) and inference is O(d·k) – faster than MLP and competitive with GBDT. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds label-mlp16 (2325 Ir/item), the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D0001, d, k, C; float32 bias, linear[d][256], embed[d][256][k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
K_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2_CHOICES = [1e-4, 1e-5]
EPOCHS = 30
BATCH = 64
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(X, bias, linear, embed, k):
    """X: (n, d) uint8. Returns logits (n,)."""
    n, d = X.shape
    # linear part
    lin = bias + np.sum(linear[np.arange(d), X], axis=1)  # (n,)
    # interaction part
    sum_emb = np.zeros((n, k), dtype=np.float32)
    sum_sq_emb = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        e = embed[j, X[:, j]]  # (n, k)
        sum_emb += e
        sum_sq_emb += e * e
    inter = 0.5 * np.sum(sum_emb * sum_emb - sum_sq_emb, axis=1)
    return lin + inter

def train_fm(X, y, k, lr, l2, epochs, batch, rng):
    n, d = X.shape
    # initialize
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = (rng.normal(0, 0.01, (d, 256, k)) * math.sqrt(1.0 / k)).astype(np.float32)
    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # forward
            lin_part = bias + np.sum(linear[np.arange(d), xb], axis=1)
            sum_e = np.zeros((len(xb), k), np.float32)
            sum_sq_e = np.zeros((len(xb), k), np.float32)
            for j in range(d):
                e = embed[j, xb[:, j]]
                sum_e += e
                sum_sq_e += e * e
            inter = 0.5 * np.sum(sum_e * sum_e - sum_sq_e, axis=1)
            logits = lin_part + inter
            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yb).astype(np.float32)  # (batch,)
            # gradients
            g_bias = grad.mean()
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(embed)
            for j in range(d):
                # linear gradient
                for v in range(256):
                    mask = (xb[:, j] == v)
                    if np.any(mask):
                        g_lin[j, v] = grad[mask].mean()
                # embedding gradient
                # inter gradient w.r.t. embed[j, v, f] = grad * (sum_e[:,f] - embed[j, v, f])
                for f in range(k):
                    # vectorized over batch
                    se_f = sum_e[:, f]
                    for v in range(256):
                        mask = (xb[:, j] == v)
                        if np.any(mask):
                            g_emb[j, v, f] = np.mean(grad[mask] * (se_f[mask] - embed[j, v, f]))
            # L2 regularization
            g_lin += l2 * linear
            g_emb += l2 * embed
            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= lr * (m_bias / (1 - beta1 ** t)) / (np.sqrt(v_bias / (1 - beta2 ** t)) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= lr * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)
            # embed
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            embed -= lr * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)
    return bias, linear, embed

def evaluate(X, y, bias, linear, embed, k):
    logits = fm_forward(X, bias, linear, embed, k)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "FM only for binary"
    n = X.shape[0]
    # tuning split: last 20%
    split = n - n // 5
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc, best_cfg = -1, None
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                bias, linear, embed = train_fm(X_train, y_train, k, lr, l2, EPOCHS, BATCH, rng)
                acc = evaluate(X_tune, y_tune, bias, linear, embed, k)
                if acc > best_acc:
                    best_acc, best_cfg = acc, (k, lr, l2)
    print(f"Tuning best: k={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]}, acc={best_acc:.4f}", file=sys.stderr)
    # retrain on all data
    k, lr, l2 = best_cfg
    bias, linear, embed = train_fm(X, y, k, lr, l2, EPOCHS, BATCH, rng)
    # export weights
    magic = 0x464D0001
    header = struct.pack("<4i", magic, d, k, C)
    out = header
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0 or 1).
 * Weights format: int32 magic 0x464D0001, d, k, C; float32 bias, linear[d][256], embed[d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14 || k <= 0 || k > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1L + (long)d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *embed = linear + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = malloc(sizeof(float) * k);
    float *sum_sq_e = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float logit = bias[0];
        memset(sum_e, 0, sizeof(float) * k);
        memset(sum_sq_e, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += linear[(long)j * 256 + v];
            const float *e = embed + ((long)j * 256 + v) * k;
            for (int f = 0; f < k; f++) {
                sum_e[f] += e[f];
                sum_sq_e[f] += e[f] * e[f];
            }
        }
        for (int f = 0; f < k; f++) {
            logit += 0.5f * (sum_e[f] * sum_e[f] - sum_sq_e[f]);
        }
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```