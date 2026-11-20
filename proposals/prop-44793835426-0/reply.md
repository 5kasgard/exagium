```hypothesis
Family: label-fm (Factorization Machine on raw 14-byte features).
Why: Adult has 14 mixed-type features; label-linear ignores pairwise interactions. FM models all pairwise interactions via low-rank factor vectors (k=4..8), adding only O(d*k) inference cost. On tabular data with categorical features, FM often matches MLP accuracy at near-linear speed. The efficient inference formula (0.5*(sum(v)^2 - sum(v^2))) computes all pairwise interactions in one pass.
Failure signal: dev accuracy <= 0.8524 (label-linear) or Ir/item > 2000 (2.8x linear cost).
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, numpy only.
Encodes each of the d input bytes as a categorical feature (256 values).
Model: bias + linear[d][256] + factor[d][256][k] (k tuned on last 20% of train).
Exported as int32 magic 0x464D3100, d, k, C; float32 bias, linear[d][256], factor[d][256][k]."""
import json, math, struct, sys

import numpy as np

# Tuning candidates: (k, lr, l2, epochs_factor)
CONFIGS = [
    (2, 1e-2, 1e-5, 1.0),
    (4, 5e-3, 1e-5, 1.0),
    (8, 3e-3, 1e-5, 1.0),
    (16, 2e-3, 1e-5, 1.0),
]
SEED = 1
BATCH = 256
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X, bias, linear, factor):
    """X: (n, d) uint8. Returns logits (n, C). Here C=2 but write generally."""
    n, d = X.shape
    k = factor.shape[2]
    C = linear.shape[2]  # (d, 256, C)
    # Linear term: (n, C)
    lin = np.zeros((n, C), dtype=np.float32)
    for j in range(d):
        lin += linear[j, X[:, j], :]
    lin += bias
    if k == 0:
        return lin
    # Interaction term: 0.5 * (sum_sq - sq_sum)
    # sum_v: (n, k)
    sum_v = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        sum_v += factor[j, X[:, j], :]
    # sum_sq = sum_v^2 summed over k -> (n,)
    sum_sq = np.sum(sum_v * sum_v, axis=1, keepdims=True)  # (n, 1)
    # sq_sum = sum(v^2) over k -> (n,)
    sq_sum = 0.0
    for j in range(d):
        sq_sum += np.sum(factor[j, X[:, j], :] ** 2, axis=1, keepdims=True)
    inter = 0.5 * (sum_sq - sq_sum)  # (n, 1)
    # For C=2, we can add same interaction to both classes (shifts logits equally, no effect on argmax).
    # To allow class-specific interactions, we'd need factor[d][256][k][C] which is larger.
    # Instead, we add a per-class linear interaction weight? Simpler: use factor[d][256][k] shared,
    # and learn a class-specific scaling vector gamma[C] for the interaction term.
    # But for binary classification, shared interaction doesn't change decision boundary.
    # So we need class-specific factors or a different approach.
    # Let's make factor[d][256][k][C] but that's 4x larger. Instead, do FM per class: factor[d][256][k] for each class?
    # Actually, standard FM for multi-class uses factor[d][256][k] and the interaction is computed per class.
    # Wait: the interaction term in FM is sum_{i<j} <v_i, v_j> x_i x_j. For multi-class, v depends on class.
    # So we need factor[C][d][256][k]. For C=2, that's 2x. Let's do that.
    # But the forward function above assumes shared. Let me redesign.
    return lin  # placeholder


def train_fm(X, y, C, d, k, lr, l2, max_epochs, rng):
    n = X.shape[0]
    # Parameters: bias[C], linear[d][256][C], factor[C][d][256][k]
    bias = np.zeros(C, dtype=np.float32)
    linear = np.zeros((d, 256, C), dtype=np.float32)
    factor = np.zeros((C, d, 256, k), dtype=np.float32)
    # Init factors small
    if k > 0:
        scale = math.sqrt(2.0 / (d * k))
        factor[:] = rng.normal(0, scale, factor.shape).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_fac, v_fac = np.zeros_like(factor), np.zeros_like(factor)
    t = 0

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]  # (b, d)
            yb = y[idx]  # (b,)
            b = len(idx)

            # Forward
            # Linear: (b, C)
            logits = bias + np.sum(linear[np.arange(d)[None, :], xb[:, :, None], :], axis=1)
            if k > 0:
                # Interaction per class
                for c in range(C):
                    sum_v = np.sum(factor[c, np.arange(d)[None, :], xb[:, :, None], :], axis=1)  # (b, k)
                    sum_sq = np.sum(sum_v * sum_v, axis=1, keepdims=True)  # (b, 1)
                    sq_sum = np.sum(factor[c, np.arange(d)[None, :], xb[:, :, None], :] ** 2, axis=(1, 2), keepdims=True)
                    logits[:, c] += 0.5 * (sum_sq - sq_sum).ravel()

            # Softmax + grad
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(b), yb] -= 1.0
            grad_logits = probs / b  # (b, C)

            # Gradients
            g_bias = grad_logits.sum(axis=0)
            g_linear = np.zeros_like(linear)
            for j in range(d):
                # xb[:, j] is (b,) values
                for c in range(C):
                    np.add.at(g_linear[j, :, c], xb[:, j], grad_logits[:, c])

            g_factor = np.zeros_like(factor)
            if k > 0:
                for c in range(C):
                    g_logit_c = grad_logits[:, c:c+1]  # (b, 1)
                    # sum_v for this class: (b, k)
                    sum_v = np.sum(factor[c, np.arange(d)[None, :], xb[:, :, None], :], axis=1)
                    # grad wrt factor[c, j, val, f] = g_logit * (sum_v[:, f] - factor[c, j, val, f])
                    for j in range(d):
                        vals = xb[:, j]
                        fac_j = factor[c, j, vals, :]  # (b, k)
                        # g_factor[c, j, vals, f] += g_logit * (sum_v[:, f] - fac_j[:, f])
                        for f in range(k):
                            diff = (sum_v[:, f] - fac_j[:, f]) * g_logit_c.ravel()
                            np.add.at(g_factor[c, j, :, f], vals, diff)

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps) + lr * l2 * bias

            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_linear
            v_lin = beta2 * v_lin + (1 - beta2) * g_linear * g_linear
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= lr * m_hat / (np.sqrt(v_hat) + eps) + lr * l2 * linear

            # factor
            if k > 0:
                m_fac = beta1 * m_fac + (1 - beta1) * g_factor
                v_fac = beta2 * v_fac + (1 - beta2) * g_factor * g_factor
                m_hat = m_fac / (1 - beta1 ** t)
                v_hat = v_fac / (1 - beta2 ** t)
                factor -= lr * m_hat / (np.sqrt(v_hat) + eps) + lr * l2 * factor

    return bias, linear, factor


def eval_acc(X, y, bias, linear, factor):
    n = X.shape[0]
    C = bias.shape[0]
    d = X.shape[1]
    k = factor.shape[3] if factor.ndim == 4 else 0
    logits = bias + np.sum(linear[np.arange(d)[None, :], X[:, :, None], :], axis=1)
    if k > 0:
        for c in range(C):
            sum_v = np.sum(factor[c, np.arange(d)[None, :], X[:, :, None], :], axis=1)
            sum_sq = np.sum(sum_v * sum_v, axis=1, keepdims=True)
            sq_sum = np.sum(factor[c, np.arange(d)[None, :], X[:, :, None], :] ** 2, axis=(1, 2), keepdims=True)
            logits[:, c] += 0.5 * (sum_sq - sq_sum).ravel()
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_all.shape[0]

    # Tuning split: last 20%
    split = int(0.8 * n)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_config = None

    for k, lr, l2, ep_factor in CONFIGS:
        max_ep = min(MAX_EPOCHS, max(1, int(math.ceil(SAMPLES / len(X_train)) * ep_factor)))
        rstate = np.random.default_rng(SEED)  # fresh for each config
        bias, linear, factor = train_fm(X_train, y_train, C, d, k, lr, l2, max_ep, rstate)
        acc = eval_acc(X_tune, y_tune, bias, linear, factor)
        if acc > best_acc:
            best_acc = acc
            best_config = (k, lr, l2, ep_factor)
            best_params = (bias.copy(), linear.copy(), factor.copy())

    # Retrain on all data with best config
    k, lr, l2, ep_factor = best_config
    max_ep = min(MAX_EPOCHS, max(1, int(math.ceil(SAMPLES / n) * ep_factor)))
    rng = np.random.default_rng(SEED)
    bias, linear, factor = train_fm(X_all, y_all, C, d, k, lr, l2, max_ep, rng)

    # Export: magic, d, k, C; bias[C]; linear[d][256][C]; factor[C][d][256][k]
    magic = 0x464D3100  # 'FM1\0'
    out = struct.pack("<4i", magic, d, k, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += factor.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class index (argmax).
 * Weights format: int32 magic=0x464D3100, d, k, C; float32 bias[C], linear[d][256][C], factor[C][d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || d > 256 || k < 0 || k > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L
        + 4L * C
        + 4L * (long)d * 256 * C
        + 4L * (long)C * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *factor = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *sum_v = k ? malloc(sizeof(float) * k) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        // Linear term + bias
        for (int c = 0; c < C; c++) logits[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *w = linear + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }

        // Interaction term per class
        if (k > 0) {
            for (int c = 0; c < C; c++) {
                // sum_v = sum_j factor[c, j, xi[j], :]
                memset(sum_v, 0, sizeof(float) * k);
                for (int j = 0; j < d; j++) {
                    const float *v = factor + ((long)c * d * 256 * k) + ((long)j * 256 + xi[j]) * k;
                    for (int f = 0; f < k; f++) sum_v[f] += v[f];
                }
                // 0.5 * (sum_v^2 - sum(v^2))
                float sum_sq = 0.0f, sq_sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    const float *v = factor + ((long)c * d * 256 * k) + ((long)j * 256 + xi[j]) * k;
                    for (int f = 0; f < k; f++) sq_sum += v[f] * v[f];
                }
                for (int f = 0; f < k; f++) sum_sq += sum_v[f] * sum_v[f];
                logits[c] += 0.5f * (sum_sq - sq_sum);
            }
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```