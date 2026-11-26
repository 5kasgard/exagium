```hypothesis
Family: label-fm (Factorization Machine on raw byte features). 
Why: Adult tabular data benefits from pairwise feature interactions (e.g., education × occupation). A low-rank FM captures these with O(d·K) parameters (K=8 → ~32k params) and O(d·K) inference, faster than MLPs. It should surpass label-linear (0.8524) and approach label-mlp16 (0.8591) while being cheaper than MLP at inference. 
Failure: dev accuracy ≤ 0.8524 or inference cost >2× label-linear.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then trained on all data.
Weights format: int32 magic 0x464D4C31, d, K, C; float32 bias; float32 linear[d][256]; float32 emb[d][256][K]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid (searched on tuning split)
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03]
L2_CHOICES = [1e-4, 1e-3]
MAX_EPOCHS = 20
BATCH = 256
SAMPLES = 600000
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train_fm(X, y, K, lr, l2, max_epochs, batch, seed, verbose=False):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    # Parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(max_epochs, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,) int64
            B = len(idx)

            # Forward
            lin_part = np.zeros(B, dtype=np.float32)
            for f in range(d):
                lin_part += linear[f, xb[:, f]]
            emb_gathered = emb[np.arange(d), xb.T]  # (d, B, K)
            sum_emb = np.sum(emb_gathered, axis=0)   # (B, K)
            sum_emb_sq = np.sum(emb_gathered ** 2, axis=0)
            fm_part = 0.5 * np.sum(sum_emb ** 2 - sum_emb_sq, axis=1)
            logits = bias + lin_part + fm_part
            probs = sigmoid(logits)
            g = probs - yb  # (B,) gradient of loss w.r.t logits

            # Gradients
            db = np.mean(g)
            dlin = np.zeros_like(linear)
            demb = np.zeros_like(emb)

            for f in range(d):
                idx_f = xb[:, f]
                # linear gradient
                np.add.at(dlin[f], idx_f, g)
                dlin[f] /= B
                # embedding gradient
                # sum_g per value
                sum_g = np.bincount(idx_f, weights=g, minlength=256)
                # sum_g_sum_emb per value per k
                for k in range(K):
                    w = g * sum_emb[:, k]
                    sum_g_se = np.bincount(idx_f, weights=w, minlength=256)
                    demb[f, :, k] = (sum_g_se - emb[f, :, k] * sum_g) / B

            # L2 regularization
            dlin += l2 * linear
            demb += l2 * emb
            db += l2 * bias

            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * db
            v_bias = beta2 * v_bias + (1 - beta2) * db * db
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * dlin
            v_lin = beta2 * v_lin + (1 - beta2) * dlin * dlin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * demb
            v_emb = beta2 * v_emb + (1 - beta2) * demb * demb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if verbose:
            # compute training accuracy
            lin_part = np.zeros(n, dtype=np.float32)
            for f in range(d):
                lin_part += linear[f, X[:, f]]
            emb_gathered = emb[np.arange(d), X.T]
            sum_emb = np.sum(emb_gathered, axis=0)
            sum_emb_sq = np.sum(emb_gathered ** 2, axis=0)
            fm_part = 0.5 * np.sum(sum_emb ** 2 - sum_emb_sq, axis=1)
            logits = bias + lin_part + fm_part
            preds = (logits > 0).astype(np.int64)
            acc = np.mean(preds == y)
            print(f"epoch {epoch+1}: train acc={acc:.4f}")

    return bias, linear, emb


def evaluate(X, y, bias, linear, emb):
    n, d = X.shape
    K = emb.shape[2]
    lin_part = np.zeros(n, dtype=np.float32)
    for f in range(d):
        lin_part += linear[f, X[:, f]]
    emb_gathered = emb[np.arange(d), X.T]
    sum_emb = np.sum(emb_gathered, axis=0)
    sum_emb_sq = np.sum(emb_gathered ** 2, axis=0)
    fm_part = 0.5 * np.sum(sum_emb ** 2 - sum_emb_sq, axis=1)
    logits = bias + lin_part + fm_part
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    n = X.shape[0]

    # Tuning split: last 20%
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc = -1.0
    best_params = None
    best_model = None

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                bias, linear, emb = train_fm(X_train, y_train, K, lr, l2, MAX_EPOCHS, BATCH, SEED)
                acc = evaluate(X_val, y_val, bias, linear, emb)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K, lr, l2)
                    best_model = (bias, linear, emb)
                print(f"K={K}, lr={lr}, l2={l2}: val acc={acc:.4f}")

    K, lr, l2 = best_params
    print(f"Best: K={K}, lr={lr}, l2={l2} with val acc={best_acc:.4f}")

    # Retrain on full data with best hyperparameters
    bias, linear, emb = train_fm(X, y, K, lr, l2, MAX_EPOCHS, BATCH, SEED)

    # Write weights.bin
    magic = 0x464D4C31  # 'FML1'
    C = card["classes"]
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += np.array([bias], dtype=np.float32).astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class index (0 or 1). */
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = (float *)malloc(sizeof(float) * K);
    float *sum_emb_sq = (float *)malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias[0];
        // linear part
        for (int f = 0; f < d; f++) {
            logit += linear[(long)f * 256 + xi[f]];
        }
        // FM part
        for (int kk = 0; kk < K; kk++) {
            sum_emb[kk] = 0.0f;
            sum_emb_sq[kk] = 0.0f;
        }
        for (int f = 0; f < d; f++) {
            const float *e = emb + ((long)f * 256 + xi[f]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = e[kk];
                sum_emb[kk] += v;
                sum_emb_sq[kk] += v * v;
            }
        }
        for (int kk = 0; kk < K; kk++) {
            logit += 0.5f * (sum_emb[kk] * sum_emb[kk] - sum_emb_sq[kk]);
        }

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sum_emb);
    free(sum_emb_sq);
    return 0;
}
```