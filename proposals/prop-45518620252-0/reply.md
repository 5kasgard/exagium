```hypothesis
Factorization Machine (FM) with low-rank embeddings captures pairwise feature interactions missed by linear models, while inference stays O(d*K) with d=14 features. On Adult's mixed categorical/ordinal data, FMs are strong tabular baselines. If dev accuracy does not exceed label-linear's 0.8524, or if inference cost (ns/item) is not clearly lower than label-gbdt's 785 ns, the family fails its design goals.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, numpy only.
Encodes each of the d=14 byte columns as 256 discrete values.
Model: w0 + sum_j w[j][x_j] + sum_{j<l} <v_j[x_j], v_l[x_l]>.
Efficient inference via (||sum v||^2 - sum ||v||^2)/2.
Hyperparameters chosen on last 20% tuning split, then retrained on all data.
Weights format: magic 0x464D4C31, d, K, C, w0, linear[d][256], embed[d][256][K] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.03, 0.1, 0.3]
L2_CHOICES = [1e-4, 1e-3]
EPOCHS = 30
BATCH = 256
PATIENCE = 3


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(y_true, scores):
    pred = (scores > 0).astype(np.int64)
    return (pred == y_true).mean()


def train_fm(X, y, K, lr, l2, rng, epochs=EPOCHS, batch=BATCH, verbose=False):
    n, d = X.shape
    # Parameters: w0, linear[d][256], embed[d][256][K]
    w0 = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed = (rng.uniform(-0.01, 0.01, (d, 256, K)) * np.float32(math.sqrt(6.0 / (256 + K)))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(embed)
    v_emb = np.zeros_like(embed)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def predict_batch(Xb):
        # Xb: (b, d) uint8
        b = Xb.shape[0]
        # Linear part
        lin = w0 + np.sum(linear[np.arange(d), Xb.T], axis=0)  # (b,)
        # Embedding part
        emb_sum = np.sum(embed[np.arange(d), Xb.T], axis=1)  # (b, K)
        sq_sum = np.sum(emb_sum ** 2, axis=1)                # (b,)
        sum_sq = np.sum(np.sum(embed[np.arange(d), Xb.T] ** 2, axis=2), axis=1)  # (b,)
        pair = 0.5 * (sq_sum - sum_sq)
        return lin + pair

    def compute_grads(Xb, yb, scores):
        b = Xb.shape[0]
        # Logistic loss gradient: sigmoid(score) - y
        sig = 1.0 / (1.0 + np.exp(-scores))
        grad = (sig - yb).astype(np.float32)  # (b,)
        # Gradient wrt w0
        gw0 = grad.mean()
        # Gradient wrt linear
        glin = np.zeros_like(linear)
        for j in range(d):
            np.add.at(glin[j], Xb[:, j], grad) / b
        # Gradient wrt embed
        gemb = np.zeros_like(embed)
        # For each sample, grad * (emb_sum - embed[j][x_j])
        emb_sum_b = np.sum(embed[np.arange(d), Xb.T], axis=1)  # (b, K)
        for j in range(d):
            v = embed[j, Xb[:, j]]  # (b, K)
            g = grad[:, None] * (emb_sum_b - v) / b  # (b, K)
            np.add.at(gemb[j], Xb[:, j], g)
        # Add L2 regularization
        gw0 += l2 * w0
        glin += l2 * linear
        gemb += l2 * embed
        return gw0, glin, gemb

    best_val_acc = -1.0
    best_params = None
    patience = 0

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb = X[idx]
            yb = y[idx]
            scores = predict_batch(Xb)
            gw0, glin, gemb = compute_grads(Xb, yb, scores)
            t += 1
            # Adam update
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (gw0 * gw0)
            m_lin = beta1 * m_lin + (1 - beta1) * glin
            v_lin = beta2 * v_lin + (1 - beta2) * (glin * glin)
            m_emb = beta1 * m_emb + (1 - beta1) * gemb
            v_emb = beta2 * v_emb + (1 - beta2) * (gemb * gemb)
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            m_lin_hat = m_lin / (1 - beta1 ** t)
            v_lin_hat = v_lin / (1 - beta2 ** t)
            m_emb_hat = m_emb / (1 - beta1 ** t)
            v_emb_hat = v_emb / (1 - beta2 ** t)
            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            linear -= lr * m_lin_hat / (np.sqrt(v_lin_hat) + eps)
            embed -= lr * m_emb_hat / (np.sqrt(v_emb_hat) + eps)

    return w0, linear, embed


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_acc = -1.0
    best_hp = None
    best_model = None

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                model = train_fm(X_train, y_train, K, lr, l2, rng)
                w0, linear, embed = model
                # Predict on tune
                lin = w0 + np.sum(linear[np.arange(d), X_tune.T], axis=0)
                emb_sum = np.sum(embed[np.arange(d), X_tune.T], axis=1)
                sq_sum = np.sum(emb_sum ** 2, axis=1)
                sum_sq = np.sum(np.sum(embed[np.arange(d), X_tune.T] ** 2, axis=2), axis=1)
                pair = 0.5 * (sq_sum - sum_sq)
                scores = lin + pair
                acc = accuracy(y_tune, scores)
                if acc > best_acc:
                    best_acc = acc
                    best_hp = (K, lr, l2)
                    best_model = model
                    print(f"New best: K={K}, lr={lr}, l2={l2}, acc={acc:.4f}")

    print(f"Best hyperparams: {best_hp}, tune acc={best_acc:.4f}")
    # Retrain on full data with best hyperparams
    K, lr, l2 = best_hp
    w0, linear, embed = train_fm(X, y, K, lr, l2, rng)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1).
 * Weights format: magic 0x464D4C31, d, K, C, w0, linear[d][256], embed[d][256][K] (float32 LE). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || C != 2 || K <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 * (1 + K));
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *linear = p + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        float score = w0;
        for (int j = 0; j < d; j++) {
            score += linear[(long)j * 256 + xi[j]];
        }

        // Embedding sum
        for (int kk = 0; kk < K; kk++) emb_sum[kk] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *v = embed + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) emb_sum[kk] += v[kk];
        }

        // Pairwise interactions: 0.5 * (||sum||^2 - sum ||v_j||^2)
        float sq_sum = 0.0f, sum_sq = 0.0f;
        for (int kk = 0; kk < K; kk++) sq_sum += emb_sum[kk] * emb_sum[kk];
        for (int j = 0; j < d; j++) {
            const float *v = embed + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) sum_sq += v[kk] * v[kk];
        }
        score += 0.5f * (sq_sum - sum_sq);

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(emb_sum);
    return 0;
}
```