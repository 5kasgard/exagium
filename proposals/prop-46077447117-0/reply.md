```hypothesis
Family: label-fm (Factorization Machine with low-rank embeddings).
Why: Adult income prediction benefits from feature interactions (e.g., education × occupation). FMs model all pairwise interactions via factorized embeddings, adding expressive power over linear models while keeping inference O(d·K) with small K (4–16). This should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at lower CPU cost than GBDT/MLP.
Failure signal: Dev accuracy ≤ 0.8524 (label-linear) or CPU cost per item ≥ label-mlp16 (≈2300 Ir/item) without accuracy gain.
```

```python
"""Factorization Machine for tabular label tasks (binary).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encoding: all 14 columns treated as categorical with 256 values (ordinal are quantile bins).
Model: bias + linear weights per (feature, value) + embeddings per (feature, value) of dim K.
Interaction: 0.5 * (||sum_v||^2 - sum||v||^2). Trained with Adam on logistic loss.
Hyperparameters (K, lr, L2) chosen on last 20% of train.in (tuning split), then retrain on all.
Weights format: magic 0x464D4C31, int32 d, K, C; float32 bias; float32 linear[d*256]; float32 emb[d*256*K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
TUNE_EPOCHS = 10
FINAL_EPOCHS = 20
BATCH = 256
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03]
L2_CHOICES = [0.0, 1e-4]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, K, rng):
    bias = np.float32(0.0)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    return bias, linear, emb


def adam_step(params, grads, states, lr, beta1, beta2, eps, t):
    for p, g, (m, v) in zip(params, grads, states):
        m *= beta1; m += (1 - beta1) * g
        v *= beta2; v += (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_epoch(X, y, bias, linear, emb, lr, L2, batch_size, rng, d, K):
    n = X.shape[0]
    order = rng.permutation(n)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    # Adam states
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        xb = X[idx]          # (b, d)
        yb = y[idx]          # (b,)
        b = xb.shape[0]

        # Gather embeddings and linear weights
        # linear part: bias + sum_j linear[j, xb[:,j]]
        lin_sum = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (b,)

        # Embeddings: (b, d, K)
        emb_batch = emb[np.arange(d), xb.T].transpose(1, 0, 2)
        sum_v = np.sum(emb_batch, axis=1)                 # (b, K)
        sum_sq = np.sum(emb_batch * emb_batch, axis=(1, 2))  # (b,)
        inter = 0.5 * (np.sum(sum_v * sum_v, axis=1) - sum_sq)  # (b,)

        logits = lin_sum + inter
        p = 1.0 / (1.0 + np.exp(-logits))
        g = p - yb.astype(np.float32)  # (b,)

        # Gradients
        # bias
        g_bias = np.sum(g)
        # linear: for each feature j, value v, add g for each sample where xb[:,j]==v
        g_linear = np.zeros_like(linear)
        for j in range(d):
            np.add.at(g_linear[j], xb[:, j], g)
        g_linear += L2 * linear

        # embeddings: for each feature j, value v, add g * (sum_v - v_j)
        g_emb = np.zeros_like(emb)
        for j in range(d):
            v_j = emb_batch[:, j, :]  # (b, K)
            diff = sum_v - v_j        # (b, K)
            # g[:, None] * diff  -> (b, K)
            grad_j = g[:, None] * diff
            # scatter add to g_emb[j, xb[:, j], :]
            np.add.at(g_emb[j], xb[:, j], grad_j)
        g_emb += L2 * emb

        t += 1
        # Update bias
        m_bias = beta1 * m_bias + (1 - beta1) * g_bias
        v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
        m_hat = m_bias / (1 - beta1 ** t)
        v_hat = v_bias / (1 - beta2 ** t)
        bias -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Update linear and emb via adam_step helper
        adam_step([linear], [g_linear], [(m_lin, v_lin)], lr, beta1, beta2, eps, t)
        adam_step([emb], [g_emb], [(m_emb, v_emb)], lr, beta1, beta2, eps, t)

    return bias, linear, emb


def evaluate(X, y, bias, linear, emb, d, K):
    # vectorized evaluation
    lin_sum = bias + np.sum(linear[np.arange(d), X.T], axis=0)
    emb_batch = emb[np.arange(d), X.T].transpose(1, 0, 2)
    sum_v = np.sum(emb_batch, axis=1)
    sum_sq = np.sum(emb_batch * emb_batch, axis=(1, 2))
    inter = 0.5 * (np.sum(sum_v * sum_v, axis=1) - sum_sq)
    logits = lin_sum + inter
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    n = X.shape[0]
    n_train = int(0.8 * n)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    best_acc = -1.0
    best_params = None
    best_hp = None

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for L2 in L2_CHOICES:
                rng = np.random.default_rng(SEED)
                bias, linear, emb = init_params(d, K, rng)
                for _ in range(TUNE_EPOCHS):
                    bias, linear, emb = train_epoch(X_tr, y_tr, bias, linear, emb, lr, L2, BATCH, rng, d, K)
                acc = evaluate(X_val, y_val, bias, linear, emb, d, K)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (bias.copy(), linear.copy(), emb.copy())
                    best_hp = (K, lr, L2)

    print(f"Best tuning: K={best_hp[0]}, lr={best_hp[1]}, L2={best_hp[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters
    K, lr, L2 = best_hp
    rng = np.random.default_rng(SEED)
    bias, linear, emb = init_params(d, K, rng)
    for _ in range(FINAL_EPOCHS):
        bias, linear, emb = train_epoch(X, y, bias, linear, emb, lr, L2, BATCH, rng, d, K)

    # Write weights.bin
    # magic, d, K, C
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0 or 1). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d <= 0 || K <= 0 || C != 2) { fprintf(stderr, "bad weights dims\n"); return 3; }

    size_t expected = 16 + 4 + (size_t)d * 256 * 4 + (size_t)d * 256 * K * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *emb = linear + (size_t)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = (float *)malloc(K * sizeof(float));
    unsigned char pred;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = *bias;
        for (int j = 0; j < d; j++) {
            lin_sum += linear[(size_t)j * 256 + xi[j]];
        }

        memset(sum_v, 0, K * sizeof(float));
        float sum_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = e[kk];
                sum_v[kk] += v;
                sum_sq += v * v;
            }
        }
        float dot_sum = 0.0f;
        for (int kk = 0; kk < K; kk++) dot_sum += sum_v[kk] * sum_v[kk];
        float inter = 0.5f * (dot_sum - sum_sq);
        float logit = lin_sum + inter;

        pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```