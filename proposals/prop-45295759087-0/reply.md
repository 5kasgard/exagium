```hypothesis
We propose a Factorization Machine (FM) with low-rank embeddings (K=4) to capture pairwise feature interactions. FMs extend linear models with a dot-product interaction term for each feature pair, computed efficiently in O(d*K) at inference. On tabular data like Adult, FMs often outperform linear models and can match small MLPs with fewer parameters and lower latency. If dev accuracy does not exceed label-linear (0.8524) or inference cost is significantly higher, the approach fails.
```

```python
"""Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Trains on 80% of train.in, tunes hyperparameters on last 20%, then retrains on all.
Exports weights: magic 0x464D3130, d, K, C; float32 linear_w[d*256], linear_b, emb[d*256*K]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen after tuning on the last 20% of train.in)
SEED = 1
K = 4                # embedding dimension
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SAMPLES = 600000     # max samples per epoch for epoch count


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(emb, linear_w, linear_b, X, d, K):
    """X: (n, d) uint8. Returns logits (n,)"""
    n = X.shape[0]
    # linear term
    lin = linear_b + np.sum(linear_w[X + np.arange(d) * 256], axis=1)
    # interaction term: 0.5 * ( (sum e)^2 - sum e^2 )
    e = emb[X + np.arange(d) * 256]          # (n, d, K)
    sum_e = np.sum(e, axis=1)                # (n, K)
    sum_e_sq = np.sum(e * e, axis=1)         # (n, K)
    inter = 0.5 * np.sum(sum_e * sum_e - sum_e_sq, axis=1)
    return lin + inter


def train_fm(X, y, d, K, C, rng, lr, l2, batch, max_epochs, samples):
    n = X.shape[0]
    # parameters
    linear_w = np.zeros(d * 256, np.float32)
    linear_b = np.zeros(1, np.float32)
    emb = (rng.uniform(-1, 1, (d * 256, K)) * math.sqrt(6.0 / (256 * K))).astype(np.float32)
    params = [linear_w, linear_b, emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(max_epochs, math.ceil(samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # forward
            e = emb[xb + np.arange(d) * 256]          # (b, d, K)
            sum_e = np.sum(e, axis=1)                 # (b, K)
            sum_e_sq = np.sum(e * e, axis=1)          # (b, K)
            lin = linear_b + np.sum(linear_w[xb + np.arange(d) * 256], axis=1)  # (b,)
            inter = 0.5 * np.sum(sum_e * sum_e - sum_e_sq, axis=1)
            logits = lin + inter                       # (b,)
            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = p - yb                              # (b,)
            # gradients
            g_lin_w = np.zeros_like(linear_w)
            g_lin_b = np.zeros_like(linear_b)
            g_emb = np.zeros_like(emb)
            # linear gradients
            np.add.at(g_lin_w, xb + np.arange(d) * 256, grad[:, None])
            g_lin_b[0] = np.sum(grad)
            # embedding gradients: for each feature j, grad * (sum_e - e_j)
            for j in range(d):
                v = xb[:, j] + j * 256
                diff = sum_e - e[:, j]                # (b, K)
                np.add.at(g_emb, v[:, None], grad[:, None] * diff)
            # L2 regularization
            g_lin_w += l2 * linear_w
            g_lin_b += l2 * linear_b
            g_emb += l2 * emb
            grads = [g_lin_w, g_lin_b, g_emb]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return linear_w, linear_b, emb


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    split = int(0.8 * n)
    # tune on last 20%
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # simple grid search (could be expanded)
    best_acc = -1
    best_params = None
    for K_try in [2, 4, 8]:
        for LR_try in [1e-3, 3e-3]:
            for L2_try in [1e-4, 1e-3]:
                rng_tune = np.random.default_rng(SEED)
                lw, lb, emb = train_fm(X_tr, y_tr, d, K_try, C, rng_tune, LR_try, L2_try,
                                       BATCH, MAX_EPOCHS, SAMPLES)
                logits = fm_forward(emb, lw, lb, X_tune, d, K_try)
                pred = (logits > 0).astype(np.float32)
                acc = np.mean(pred == y_tune)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K_try, LR_try, L2_try)
                    print(f"Tune: K={K_try}, LR={LR_try}, L2={L2_try}, acc={acc:.4f}")

    K_best, LR_best, L2_best = best_params
    print(f"Best: K={K_best}, LR={LR_best}, L2={L2_best}, acc={best_acc:.4f}")

    # retrain on all data
    rng_final = np.random.default_rng(SEED)
    linear_w, linear_b, emb = train_fm(X, y, d, K_best, C, rng_final, LR_best, L2_best,
                                       BATCH, MAX_EPOCHS, SAMPLES)

    # write weights
    magic = 0x464D3130  # 'FM10'
    out = struct.pack("<4i", magic, d, K_best, C)
    out += linear_w.astype("<f4").tobytes()
    out += linear_b.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (d * 256 + 1 + d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *linear_w = (const float *)(hd + 4);
    const float *linear_b = linear_w + d * 256;
    const float *emb = linear_b + 1;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = malloc(sizeof(float) * K);
    float *sum_e_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float lin = linear_b[0];
        for (int u = 0; u < K; u++) { sum_e[u] = 0.0f; sum_e_sq[u] = 0.0f; }
        for (int j = 0; j < d; j++) {
            int idx = xi[j] + j * 256;
            lin += linear_w[idx];
            const float *e = emb + (long)idx * K;
            for (int u = 0; u < K; u++) {
                float val = e[u];
                sum_e[u] += val;
                sum_e_sq[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += sum_e[u] * sum_e[u] - sum_e_sq[u];
        inter *= 0.5f;
        float logit = lin + inter;
        unsigned char best = logit > 0.0f ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```