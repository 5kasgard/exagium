```hypothesis
We propose a factorization machine family "label-fm". It models pairwise feature interactions via low-rank embeddings (dimension k), which is known to work well on tabular categorical data like Adult. The FM computes a linear term plus a sum of squared dot products, capturing second-order interactions efficiently in O(d*k) time per item (d=14, k=8). This should beat label-linear (0.8524) by adding interaction modeling while staying much faster than label-gbdt. If dev accuracy does not exceed label-linear or inference cost exceeds ~2000 Ir/item, the approach fails.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Trains a binary FM with logistic loss using Adam. Hyperparameters chosen on last 20% tuning split."""
import json, math, struct, sys

import numpy as np

# Fixed seed for reproducibility
SEED = 1

# Grid for tuning on the held-out 20% of train.in (file order)
GRID = [
    {"k": 4, "lr": 0.03, "l2": 1e-4, "epochs": 15, "batch": 64},
    {"k": 8, "lr": 0.03, "l2": 1e-4, "epochs": 15, "batch": 64},
    {"k": 8, "lr": 0.01, "l2": 1e-4, "epochs": 15, "batch": 64},
    {"k": 16, "lr": 0.01, "l2": 1e-4, "epochs": 15, "batch": 64},
]

FINAL_EPOCHS = 30
FINAL_BATCH = 128


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_score(linear_w, embed, bias, x, k, d):
    """x: (n, d) uint8. Returns logits (n,)."""
    n = x.shape[0]
    # linear part: bias + sum_j linear_w[j, x_ij]
    lin = np.full(n, bias, dtype=np.float32)
    for j in range(d):
        lin += linear_w[j, x[:, j]]
    # FM part: 0.5 * sum_f ( (sum_j e_{j,f})^2 - sum_j e_{j,f}^2 )
    sum_e = np.zeros((n, k), dtype=np.float32)
    sum_e2 = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        e = embed[j, x[:, j], :]          # (n, k)
        sum_e += e
        sum_e2 += e * e
    fm = 0.5 * np.sum(sum_e * sum_e - sum_e2, axis=1)
    return lin + fm


def train_fm(X, y, d, k, lr, l2, epochs, batch, rng):
    n = X.shape[0]
    # Initialize parameters
    linear_w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    embed = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_lin, v_lin = np.zeros_like(linear_w), np.zeros_like(linear_w)
    m_emb, v_emb = np.zeros_like(embed), np.zeros_like(embed)
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = len(idx)

            # Forward
            lin = np.full(b, bias, dtype=np.float32)
            sum_e = np.zeros((b, k), dtype=np.float32)
            sum_e2 = np.zeros((b, k), dtype=np.float32)
            for j in range(d):
                lin += linear_w[j, xb[:, j]]
                e = embed[j, xb[:, j], :]
                sum_e += e
                sum_e2 += e * e
            fm = 0.5 * np.sum(sum_e * sum_e - sum_e2, axis=1)
            logits = lin + fm

            # Logistic loss gradient: sigmoid(logits) - y
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yb).astype(np.float32)  # (b,)

            # Backward
            # bias
            g_bias = grad.mean()
            # linear_w
            g_lin = np.zeros_like(linear_w)
            for j in range(d):
                np.add.at(g_lin[j], xb[:, j], grad)
            g_lin /= b
            # embed
            g_emb = np.zeros_like(embed)
            for j in range(d):
                # For each feature j, we need d(sum_e)/d e = 1, d(sum_e2)/d e = 2*e
                # fm = 0.5 * sum_f (sum_e^2 - sum_e2)
                # d(fm)/d e_{j,f} = sum_e_f - e_{j,f}
                # So gradient contribution: grad_i * (sum_e_i - e_{i,j})
                # We can compute per example and accumulate
                # Vectorized: for each example i, we have sum_e[i] and e[i,j]
                # We'll loop over batch for simplicity (batch <= 128)
                for i in range(b):
                    v = xb[i, j]
                    g_emb[j, v] += grad[i] * (sum_e[i] - embed[j, v])
            g_emb /= b

            # L2 regularization
            g_lin += l2 * linear_w
            g_emb += l2 * embed
            g_bias += l2 * bias

            # Adam updates
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * (g_bias * g_bias)
            m_bias_hat = m_bias / (1 - beta1 ** t)
            v_bias_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_bias_hat / (np.sqrt(v_bias_hat) + eps)

            # linear_w
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * (g_lin * g_lin)
            m_lin_hat = m_lin / (1 - beta1 ** t)
            v_lin_hat = v_lin / (1 - beta2 ** t)
            linear_w -= lr * m_lin_hat / (np.sqrt(v_lin_hat) + eps)

            # embed
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * (g_emb * g_emb)
            m_emb_hat = m_emb / (1 - beta1 ** t)
            v_emb_hat = v_emb / (1 - beta2 ** t)
            embed -= lr * m_emb_hat / (np.sqrt(v_emb_hat) + eps)

    return linear_w, embed, bias


def evaluate(X, y, linear_w, embed, bias, k, d):
    logits = fm_score(linear_w, embed, bias, X, k, d)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X_all)

    # Split: last 20% for tuning (file order)
    split = int(0.8 * n)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    rng = np.random.default_rng(SEED)

    # Grid search on tuning split
    best_acc = -1.0
    best_params = None
    best_model = None
    for params in GRID:
        k = params["k"]
        model = train_fm(X_train, y_train, d, k, params["lr"], params["l2"],
                         params["epochs"], params["batch"], rng)
        acc = evaluate(X_tune, y_tune, *model, k, d)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = model

    # Retrain on all data with best params
    k = best_params["k"]
    final_model = train_fm(X_all, y_all, d, k, best_params["lr"], best_params["l2"],
                           FINAL_EPOCHS, FINAL_BATCH, rng)
    linear_w, embed, bias = final_model

    # Write weights.bin
    # Format: magic (4 bytes), d (i32), k (i32), then bias (f32),
    # linear_w: d * 256 f32 (feature-major),
    # embed: d * 256 * k f32 (feature-major, value-major, dim-major)
    magic = 0x464D0001  # "FM\0\1"
    out = struct.pack("<3i", magic, d, k)
    out += struct.pack("<f", bias)
    out += linear_w.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label-fm family.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1). */
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
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D0001) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2];
    if (d <= 0 || k <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    // Expected size: 12 (header) + 4 (bias) + 4*d*256 (linear_w) + 4*d*256*k (embed)
    long expected = 12 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    float bias = *(float *)(hd + 3);
    float *linear_w = (float *)(hd + 4);                 // [d][256]
    float *embed = linear_w + (long)d * 256;             // [d][256][k]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = malloc(sizeof(float) * k);
    float *sum_e2 = malloc(sizeof(float) * k);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        // Linear part
        float score = bias;
        for (int j = 0; j < d; j++) {
            score += linear_w[(long)j * 256 + xi[j]];
        }

        // FM part
        memset(sum_e, 0, sizeof(float) * k);
        memset(sum_e2, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            float *e = embed + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                sum_e[f] += e[f];
                sum_e2[f] += e[f] * e[f];
            }
        }
        for (int f = 0; f < k; f++) {
            score += 0.5f * (sum_e[f] * sum_e[f] - sum_e2[f]);
        }

        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```