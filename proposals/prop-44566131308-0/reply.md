```hypothesis
Family: label-fm (Factorization Machine with low-rank pairwise interactions).
Why: Adult has 14 mixed ordinal/categorical features; FM captures feature crosses with O(d*K) parameters, avoiding the O(d^2) blowup. With K=4 it adds only ~18K floats vs label-linear's ~3.6K, but models interactions crucial for income prediction (e.g., education × occupation). FMs are known strong on tabular data.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear by >2×, the model is not an improvement.
```

```python
import json, math, struct, sys

import numpy as np

K = 4
LR = 0.01
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 50
PATIENCE = 5
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    train_idx = idx[:split]
    val_idx = idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    rng = np.random.default_rng(SEED)
    linear = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    factor = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)

    m_linear = np.zeros_like(linear)
    v_linear = np.zeros_like(linear)
    m_factor = np.zeros_like(factor)
    v_factor = np.zeros_like(factor)
    m_bias = np.float32(0.0)
    v_bias = np.float32(0.0)
    t = 0

    def predict_batch(X_batch):
        bs = X_batch.shape[0]
        lin = bias
        for j in range(d):
            lin += linear[j, X_batch[:, j]]
        s = np.zeros((bs, K), dtype=np.float32)
        sq = np.zeros((bs, K), dtype=np.float32)
        for j in range(d):
            vals = X_batch[:, j]
            f_j = factor[j, vals, :]
            s += f_j
            sq += f_j * f_j
        fm = 0.5 * np.sum(s * s - sq, axis=1)
        return lin + fm

    def train_epoch(X_data, y_data):
        nonlocal t, linear, factor, bias, m_linear, v_linear, m_factor, v_factor, m_bias, v_bias
        order = rng.permutation(len(X_data))
        for start in range(0, len(X_data), BATCH):
            idx_batch = order[start:start + BATCH]
            xb = X_data[idx_batch]
            yb = y_data[idx_batch]
            bs = len(yb)
            lin = bias
            for j in range(d):
                lin += linear[j, xb[:, j]]
            s = np.zeros((bs, K), dtype=np.float32)
            sq = np.zeros((bs, K), dtype=np.float32)
            for j in range(d):
                vals = xb[:, j]
                f_j = factor[j, vals, :]
                s += f_j
                sq += f_j * f_j
            fm = 0.5 * np.sum(s * s - sq, axis=1)
            y_hat = lin + fm
            p = 1.0 / (1.0 + np.exp(-y_hat))
            grad = p - yb
            g_bias = grad.mean()
            g_linear = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_linear[j], vals, grad)
            g_linear /= bs
            g_factor = np.zeros_like(factor)
            for j in range(d):
                vals = xb[:, j]
                diff = s - factor[j, vals, :]
                g = grad[:, None] * diff
                for f in range(K):
                    np.add.at(g_factor[j, :, f], vals, g[:, f])
            g_factor /= bs
            g_linear += L2 * linear
            g_factor += L2 * factor
            g_bias += L2 * bias
            t += 1
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            m_bias_hat = m_bias / (1 - 0.9 ** t)
            v_bias_hat = v_bias / (1 - 0.999 ** t)
            bias -= LR * m_bias_hat / (np.sqrt(v_bias_hat) + 1e-8)
            m_linear = 0.9 * m_linear + 0.1 * g_linear
            v_linear = 0.999 * v_linear + 0.001 * g_linear * g_linear
            m_linear_hat = m_linear / (1 - 0.9 ** t)
            v_linear_hat = v_linear / (1 - 0.999 ** t)
            linear -= LR * m_linear_hat / (np.sqrt(v_linear_hat) + 1e-8)
            m_factor = 0.9 * m_factor + 0.1 * g_factor
            v_factor = 0.999 * v_factor + 0.001 * g_factor * g_factor
            m_factor_hat = m_factor / (1 - 0.9 ** t)
            v_factor_hat = v_factor / (1 - 0.999 ** t)
            factor -= LR * m_factor_hat / (np.sqrt(v_factor_hat) + 1e-8)

    def evaluate(X_eval, y_eval):
        y_hat = predict_batch(X_eval)
        p = 1.0 / (1.0 + np.exp(-y_hat))
        pred = (p > 0.5).astype(np.int64)
        return np.mean(pred == y_eval)

    best_val_acc = 0.0
    best_epoch = 0
    for epoch in range(MAX_EPOCHS):
        train_epoch(X_train, y_train)
        val_acc = evaluate(X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        elif epoch - best_epoch >= PATIENCE:
            break

    # Retrain on full data for best_epoch epochs
    rng = np.random.default_rng(SEED)
    linear = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    factor = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    bias = np.float32(0.0)
    m_linear = np.zeros_like(linear)
    v_linear = np.zeros_like(linear)
    m_factor = np.zeros_like(factor)
    v_factor = np.zeros_like(factor)
    m_bias = np.float32(0.0)
    v_bias = np.float32(0.0)
    t = 0
    for epoch in range(best_epoch):
        train_epoch(X, y)

    magic = 0x464D3231
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += factor.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x464D3231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1];
    int K = (int)hd[2];
    int C = (int)hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }
    const float *bias = (const float *)(wb + 16);
    const float *linear = (const float *)(wb + 16 + 4);
    const float *factor = (const float *)(wb + 16 + 4 + 4L * d * 256);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_f[64], sum_sq_f[64];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float score = bias[0];
        for (int j = 0; j < d; j++) {
            score += linear[j * 256 + xi[j]];
        }
        for (int f = 0; f < K; f++) {
            sum_f[f] = 0.0f;
            sum_sq_f[f] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            const float *fj = factor + ((j * 256 + xi[j]) * K);
            for (int f = 0; f < K; f++) {
                float v = fj[f];
                sum_f[f] += v;
                sum_sq_f[f] += v * v;
            }
        }
        for (int f = 0; f < K; f++) {
            score += 0.5f * (sum_f[f] * sum_f[f] - sum_sq_f[f]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```