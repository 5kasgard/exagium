```hypothesis
We propose a Factorization Machine (family "label-fm") with low-rank embeddings (k=8). FMs capture pairwise feature interactions efficiently, which are known to be predictive for Adult (e.g., education × occupation). The model adds only ~130 KB weights and ~2k FLOPs/item, keeping inference faster than MLPs while being more expressive than linear. If dev accuracy does not exceed label-linear's 0.8524, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Trains an FM with embedding dimension k chosen on the tuning split (last 20%),
then re-trains on all data. Exports: magic 0x464D3132, d, k, C; float32 embeddings[d][256][k],
linear_weights[d][256], bias."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters for final training (tuning will pick k and lr)
SEED = 1
BATCH = 256
EPOCHS_TUNE = 5
EPOCHS_FULL = 20
L2 = 1e-4
K_CANDIDATES = [4, 8, 16]
LR_CANDIDATES = [0.01, 0.03, 0.1]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(X, emb, lin, bias):
    """X: (n, d) uint8. Returns logits (n,)"""
    n, d = X.shape
    k = emb.shape[2]
    linear_sum = np.full(n, bias, dtype=np.float32)
    sum_emb = np.zeros((n, k), dtype=np.float32)
    sum_sq_emb = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        idx = X[:, j]
        linear_sum += lin[j, idx]
        e = emb[j, idx]           # (n, k)
        sum_emb += e
        sum_sq_emb += e * e
    interaction = 0.5 * (np.sum(sum_emb * sum_emb, axis=1) - np.sum(sum_sq_emb, axis=1))
    return linear_sum + interaction

def fm_backward(X, y, emb, lin, bias, logits, lr, t, state_emb, state_lin, state_bias):
    """Adam update for one batch. Returns updated parameters and states."""
    n, d = X.shape
    k = emb.shape[2]
    p = 1.0 / (1.0 + np.exp(-logits))
    grad_logit = (p - y).astype(np.float32)          # (n,)
    # Linear gradients
    grad_lin = np.zeros_like(lin)
    for j in range(d):
        idx = X[:, j]
        np.add.at(grad_lin[j], idx, grad_logit)
    grad_lin += L2 * lin
    # Bias gradient
    grad_bias = np.sum(grad_logit) + L2 * bias
    # Embedding gradients
    grad_emb = np.zeros_like(emb)
    sum_emb = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        idx = X[:, j]
        sum_emb += emb[j, idx]
    for j in range(d):
        idx = X[:, j]
        e = emb[j, idx]                               # (n, k)
        # d(interaction)/de = sum_emb - e
        grad_inter = grad_logit[:, None] * (sum_emb - e)
        np.add.at(grad_emb[j], idx, grad_inter)
    grad_emb += L2 * emb

    # Adam updates
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    # embeddings
    m_emb, v_emb = state_emb
    m_emb *= beta1; m_emb += (1 - beta1) * grad_emb
    v_emb *= beta2; v_emb += (1 - beta2) * grad_emb * grad_emb
    m_hat_emb = m_emb / (1 - beta1 ** t)
    v_hat_emb = v_emb / (1 - beta2 ** t)
    emb -= lr * m_hat_emb / (np.sqrt(v_hat_emb) + eps)
    # linear
    m_lin, v_lin = state_lin
    m_lin *= beta1; m_lin += (1 - beta1) * grad_lin
    v_lin *= beta2; v_lin += (1 - beta2) * grad_lin * grad_lin
    m_hat_lin = m_lin / (1 - beta1 ** t)
    v_hat_lin = v_lin / (1 - beta2 ** t)
    lin -= lr * m_hat_lin / (np.sqrt(v_hat_lin) + eps)
    # bias
    m_bias, v_bias = state_bias
    m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
    v_bias = beta2 * v_bias + (1 - beta2) * grad_bias * grad_bias
    m_hat_bias = m_bias / (1 - beta1 ** t)
    v_hat_bias = v_bias / (1 - beta2 ** t)
    bias -= lr * m_hat_bias / (np.sqrt(v_hat_bias) + eps)

    return emb, lin, bias, (m_emb, v_emb), (m_lin, v_lin), (m_bias, v_bias)

def train_fm(X, y, k, lr, epochs, rng):
    n, d = X.shape
    C = 2
    # Initialize parameters
    emb = (rng.uniform(-0.01, 0.01, (d, 256, k)) * math.sqrt(6.0 / (256 + k))).astype(np.float32)
    lin = (rng.uniform(-0.01, 0.01, (d, 256)) * math.sqrt(6.0 / 256)).astype(np.float32)
    bias = np.float32(0.0)
    # Adam state
    state_emb = (np.zeros_like(emb), np.zeros_like(emb))
    state_lin = (np.zeros_like(lin), np.zeros_like(lin))
    state_bias = (np.float32(0.0), np.float32(0.0))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            logits = fm_forward(Xb, emb, lin, bias)
            t += 1
            emb, lin, bias, state_emb, state_lin, state_bias = fm_backward(
                Xb, yb, emb, lin, bias, logits, lr, t,
                state_emb, state_lin, state_bias)
    return emb, lin, bias

def evaluate(X, y, emb, lin, bias):
    logits = fm_forward(X, emb, lin, bias)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 14 and C == 2

    # Split: last 20% for tuning
    n = X.shape[0]
    tune_size = n // 5
    X_train, y_train = X[:-tune_size], y[:-tune_size]
    X_tune, y_tune = X[-tune_size:], y[-tune_size:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_k, best_lr = 8, 0.03

    # Grid search on tuning split
    for k in K_CANDIDATES:
        for lr in LR_CANDIDATES:
            # Derive a seed for this config to keep deterministic
            config_seed = (SEED * 1000 + k * 10 + int(lr * 100)) % (2**32)
            config_rng = np.random.default_rng(config_seed)
            emb, lin, bias = train_fm(X_train, y_train, k, lr, EPOCHS_TUNE, config_rng)
            acc = evaluate(X_tune, y_tune, emb, lin, bias)
            if acc > best_acc:
                best_acc = acc
                best_k, best_lr = k, lr

    # Retrain on full data with best hyperparameters
    final_seed = (SEED * 1000 + best_k * 10 + int(best_lr * 100)) % (2**32)
    final_rng = np.random.default_rng(final_seed)
    emb, lin, bias = train_fm(X, y, best_k, best_lr, EPOCHS_FULL, final_rng)

    # Export weights
    # Magic: 0x464D3132 ("FM12")
    magic = 0x464D3132
    k = best_k
    header = struct.pack("<4i", magic, d, k, C)
    # embeddings: d * 256 * k floats (field-major, then category, then embedding)
    emb_bytes = emb.astype("<f4").tobytes()
    # linear weights: d * 256 floats
    lin_bytes = lin.astype("<f4").tobytes()
    # bias: 1 float
    bias_bytes = np.array([bias], dtype="<f4").tobytes()
    out = header + emb_bytes + lin_bytes + bias_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (one per field). Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * k + (long)d * 256 + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *lin = emb + (long)d * 256 * k;
    const float bias = lin[(long)d * 256];

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = calloc(k, sizeof(float));
    float *sum_sq_emb = calloc(k, sizeof(float));
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float linear_sum = bias;
        memset(sum_emb, 0, k * sizeof(float));
        memset(sum_sq_emb, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *e = emb + ((long)j * 256 + v) * k;
            linear_sum += lin[(long)j * 256 + v];
            for (int u = 0; u < k; u++) {
                float val = e[u];
                sum_emb[u] += val;
                sum_sq_emb[u] += val * val;
            }
        }
        float interaction = 0.0f;
        for (int u = 0; u < k; u++) {
            interaction += sum_emb[u] * sum_emb[u] - sum_sq_emb[u];
        }
        interaction *= 0.5f;
        float logit = linear_sum + interaction;
        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    free(sum_emb); free(sum_sq_emb);
    return 0;
}
```