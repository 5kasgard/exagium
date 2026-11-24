```hypothesis
Family: label-fm (Factorization Machine with learned embeddings per feature value).
Why: Adult data has strong pairwise interactions (e.g., education × occupation) that linear models miss. A small FM (embedding size 8–16) captures these at O(d·K) inference cost, far cheaper than GBDT. If it fails to beat label-linear, interactions are either weak or the FM overfits despite regularization.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x464D3130 ("FM10"), int32 d, K, C; float32 bias, w[d*256], e[d*256*K]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
K_CHOICES = [8, 16]
LR_CHOICES = [1e-3, 3e-3]
L2_CHOICES = [1e-4, 1e-5]
EPOCHS_CHOICES = [10, 20]
SEED = 1
BATCH = 64
FULL_EPOCHS_MULT = 2  # retrain on full data for this many times the chosen epochs


def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, bias, w, e, K):
    """X: (n, d) uint8. Returns logits (n,) float32."""
    n, d = X.shape
    # linear part: bias + sum_j w[j, X[:, j]]
    linear = bias + np.sum(w[np.arange(d), X.T], axis=0)  # (n,)
    # interaction part
    # e has shape (d, 256, K). Gather embeddings: (d, n, K)
    emb = e[np.arange(d), X.T]  # (d, n, K)
    sum_emb = np.sum(emb, axis=0)  # (n, K)
    sum_sq_emb = np.sum(emb * emb, axis=0)  # (n, K)
    interaction = 0.5 * np.sum(sum_emb * sum_emb - sum_sq_emb, axis=1)  # (n,)
    return (linear + interaction).astype(np.float32)


def train_fm(X, y, K, lr, l2, epochs, rng):
    n, d = X.shape
    # Initialize parameters
    bias = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    e = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    # Adam moments
    m_b, v_b = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_e, v_e = np.zeros_like(e), np.zeros_like(e)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            # forward
            logits = fm_forward(xb, bias, w, e, K)
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logit = (p - yb).astype(np.float32)  # (B,)
            # gradients
            g_b = np.mean(grad_logit)
            # linear weights gradient
            g_w = np.zeros_like(w)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_w[j], vals, grad_logit)
            g_w /= BATCH
            g_w += l2 * w
            # embedding gradients
            g_e = np.zeros_like(e)
            # gather embeddings for this batch: (d, B, K)
            emb = e[np.arange(d), xb.T]  # (d, B, K)
            sum_emb = np.sum(emb, axis=0)  # (B, K)
            for j in range(d):
                vals = xb[:, j]
                # for each feature j, gradient for embedding of value v is grad_logit * (sum_emb[:, f] - emb[j, :, f])
                # vectorized over batch and K
                diff = sum_emb - emb[j]  # (B, K)
                # multiply by grad_logit[:, None]
                grad_emb_j = grad_logit[:, None] * diff  # (B, K)
                # accumulate per value
                np.add.at(g_e[j], vals, grad_emb_j)
            g_e /= BATCH
            g_e += l2 * e
            # Adam update
            t += 1
            # bias
            m_b = beta1 * m_b + (1 - beta1) * g_b
            v_b = beta2 * v_b + (1 - beta2) * g_b * g_b
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            bias -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            # e
            m_e = beta1 * m_e + (1 - beta1) * g_e
            v_e = beta2 * v_e + (1 - beta2) * g_e * g_e
            m_e_hat = m_e / (1 - beta1 ** t)
            v_e_hat = v_e / (1 - beta2 ** t)
            e -= lr * m_e_hat / (np.sqrt(v_e_hat) + eps)
    return bias, w, e


def accuracy(X, y, bias, w, e, K):
    logits = fm_forward(X, bias, w, e, K)
    pred = (logits > 0).astype(np.int64)
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_acc = -1.0
    best_cfg = None
    best_params = None
    rng = np.random.default_rng(SEED)

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                for epochs in EPOCHS_CHOICES:
                    # deterministic seed per config
                    cfg_rng = np.random.default_rng(SEED + hash((K, lr, l2, epochs)) % 2**32)
                    bias, w, e = train_fm(X_train, y_train, K, lr, l2, epochs, cfg_rng)
                    acc = accuracy(X_tune, y_tune, bias, w, e, K)
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = (K, lr, l2, epochs)
                        best_params = (bias, w, e)

    # Retrain on full data with best config
    K, lr, l2, epochs = best_cfg
    full_epochs = epochs * FULL_EPOCHS_MULT
    rng_full = np.random.default_rng(SEED)
    bias, w, e = train_fm(X, y, K, lr, l2, full_epochs, rng_full)

    # Write weights.bin
    # magic, d, K, C
    header = struct.pack("<4i", 0x464D3130, d, K, C)
    # bias (float32), w (d*256 float32), e (d*256*K float32)
    body = struct.pack("<f", bias) + w.astype("<f4").tobytes() + e.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label-fm family.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class (0 or 1). */
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
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0;
    long N = atol(argv[3]);
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0 || d > 256 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 16 + 4 * (1 + (size_t)d * 256 + (size_t)d * 256 * K);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *w = bias + 1;
    const float *e = w + (size_t)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;
    float *sum_emb = (float *)malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // linear part
        float linear = *bias;
        for (int j = 0; j < d; j++) linear += w[j * 256 + xi[j]];
        // interaction part
        memset(sum_emb, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *ej = e + ((size_t)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) sum_emb[f] += ej[f];
        }
        float interaction = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *ej = e + ((size_t)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float v = ej[f];
                interaction += sum_emb[f] * v - v * v;
            }
        }
        interaction *= 0.5f;
        float logit = linear + interaction;
        // scores: class 0 = -logit, class 1 = logit
        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    free(sum_emb);
    return 0;
}
```