```hypothesis
We propose a Factorization Machine (FM) with embedding size 8, treating all 14 features as categorical (256 values each). FM captures pairwise feature interactions efficiently, which should improve over the linear model on this tabular dataset. If dev accuracy does not exceed 0.8524 (label-linear), the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Uses last 20% of train for validation/early stopping, then retrains on all.
Exports: magic 0x464D3131, d, k, C=1; float32 w0, linear[d*256], emb[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 8
LR = 0.01
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 30
VAL_FRAC = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    n, d = X.shape
    assert d == 14
    C = 1  # binary logit

    rng = np.random.default_rng(SEED)

    # Split for validation (last 20% in file order)
    split = int(n * (1 - VAL_FRAC))
    idx = np.arange(n)
    tr_idx, val_idx = idx[:split], idx[split:]

    # Initialize parameters
    w0 = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.uniform(-0.01, 0.01, (d, 256, K)) * np.sqrt(6.0 / (256 + K))).astype(np.float32)

    # Adam state
    def adam_state(shape):
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    m_w0, v_w0 = 0.0, 0.0
    m_lin, v_lin = adam_state(linear.shape)
    m_emb, v_emb = adam_state(emb.shape)

    t = 0
    best_val_acc = -1.0
    best_epoch = 0
    best_params = None

    def forward(Xb):
        # Xb: (batch, d) uint8
        batch = Xb.shape[0]
        logit = np.full(batch, w0, dtype=np.float32)
        # linear part
        for j in range(d):
            logit += linear[j, Xb[:, j]]
        # interaction part
        sum_v = np.zeros((batch, K), dtype=np.float32)
        sum_v_sq = np.zeros((batch, K), dtype=np.float32)
        for j in range(d):
            e = emb[j, Xb[:, j], :]  # (batch, K)
            sum_v += e
            sum_v_sq += e * e
        interaction = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
        logit += interaction
        return logit

    def backward(Xb, grad_logit):
        # grad_logit: (batch,) dL/dlogit
        batch = Xb.shape[0]
        nonlocal m_w0, v_w0, m_lin, v_lin, m_emb, v_emb, w0, linear, emb, t
        t += 1
        # gradient wrt w0
        g_w0 = np.sum(grad_logit)
        # gradient wrt linear
        g_lin = np.zeros_like(linear)
        for j in range(d):
            np.add.at(g_lin[j], Xb[:, j], grad_logit)
        # gradient wrt emb
        g_emb = np.zeros_like(emb)
        # compute sum_v for each sample again
        sum_v = np.zeros((batch, K), dtype=np.float32)
        for j in range(d):
            sum_v += emb[j, Xb[:, j], :]
        for j in range(d):
            e = emb[j, Xb[:, j], :]  # (batch, K)
            # dL/demb = grad_logit[:, None] * (sum_v - e)
            g_emb[j, Xb[:, j], :] += grad_logit[:, None] * (sum_v - e)

        # Add L2 regularization
        g_w0 += L2 * w0
        g_lin += L2 * linear
        g_emb += L2 * emb

        # Adam updates
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        # w0
        m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
        v_w0 = beta2 * v_w0 + (1 - beta2) * (g_w0 * g_w0)
        m_hat = m_w0 / (1 - beta1 ** t)
        v_hat = v_w0 / (1 - beta2 ** t)
        w0 -= LR * m_hat / (np.sqrt(v_hat) + eps)
        # linear
        m_lin = beta1 * m_lin + (1 - beta1) * g_lin
        v_lin = beta2 * v_lin + (1 - beta2) * (g_lin * g_lin)
        m_hat = m_lin / (1 - beta1 ** t)
        v_hat = v_lin / (1 - beta2 ** t)
        linear -= LR * m_hat / (np.sqrt(v_hat) + eps)
        # emb
        m_emb = beta1 * m_emb + (1 - beta1) * g_emb
        v_emb = beta2 * v_emb + (1 - beta2) * (g_emb * g_emb)
        m_hat = m_emb / (1 - beta1 ** t)
        v_hat = v_emb / (1 - beta2 ** t)
        emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Training loop with validation
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(len(tr_idx))
        for s in range(0, len(tr_idx), BATCH):
            batch_idx = tr_idx[order[s:s + BATCH]]
            Xb = X[batch_idx]
            yb = y[batch_idx]
            logit = forward(Xb)
            p = sigmoid(logit)
            grad_logit = (p - yb).astype(np.float32)  # dL/dlogit
            backward(Xb, grad_logit)

        # Validation accuracy
        val_logit = forward(X[val_idx])
        val_pred = (val_logit > 0).astype(np.int64)
        val_acc = np.mean(val_pred == y[val_idx])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_params = (w0.copy(), linear.copy(), emb.copy())

    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}", file=sys.stderr)

    # Retrain on full data for best_epoch epochs
    w0, linear, emb = best_params
    # Reset Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_lin, v_lin = adam_state(linear.shape)
    m_emb, v_emb = adam_state(emb.shape)
    t = 0

    for epoch in range(best_epoch):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            Xb = X[batch_idx]
            yb = y[batch_idx]
            logit = forward(Xb)
            p = sigmoid(logit)
            grad_logit = (p - yb).astype(np.float32)
            backward(Xb, grad_logit)

    # Write weights
    magic = 0x464D3131
    header = struct.pack("<IIII", magic, d, K, C)
    out = header
    out += np.array(w0, dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K <= 0 || K > 64 || C != 1) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = w0;
        for (int j = 0; j < d; j++) {
            logit += linear[j * 256 + xi[j]];
        }

        memset(sum_v, 0, K * sizeof(float));
        memset(sum_v_sq, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                sum_v[f] += val;
                sum_v_sq[f] += val * val;
            }
        }
        float interaction = 0.0f;
        for (int f = 0; f < K; f++) {
            interaction += sum_v[f] * sum_v[f] - sum_v_sq[f];
        }
        logit += 0.5f * interaction;

        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```