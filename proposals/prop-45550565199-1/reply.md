```hypothesis
Family: label-gam (Generalized Additive Model with per-feature lookup tables).
Why: Optdigits pixels have non-linear intensity effects; a GAM captures per-pixel non-linearities without interactions, offering more expressivity than linear but far fewer parameters than MLP, so it should beat label-linear accuracy at similar inference cost.
Failure: If dev accuracy <= 0.9613 (label-linear) or inference cost exceeds label-mlp16.
```

```python
import json
import math
import struct
import sys

import numpy as np

# Fixed seed for reproducibility
SEED = 1
# Hyperparameter grid for tuning split
LRS = [0.1, 0.03, 0.01]
L2S = [1e-4, 1e-3, 1e-2]
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 5  # early stopping patience


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_grad(logits, y):
    """Gradient of cross-entropy w.r.t. logits (already softmaxed)."""
    probs = softmax(logits)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def train_gam(X, y, C, d, lr, l2, batch, max_epochs, rng, verbose=False):
    n = X.shape[0]
    # Initialize tables: (C, d, 256) and bias (C,)
    tables = rng.normal(0, 0.01, size=(C, d, 256)).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)
    # Adam state
    m_tables, v_tables = np.zeros_like(tables), np.zeros_like(tables)
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_val_acc = -1.0
    best_tables, best_bias = None, None
    patience_counter = 0

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # Forward: compute logits (batch, C)
            logits = bias[None, :] + np.sum(tables[:, xb.T, np.arange(d)], axis=1).T  # (batch, C)
            # Gradient
            grad_logits = cross_entropy_grad(logits, yb)  # (batch, C)
            # Gradient w.r.t tables and bias
            grad_bias = grad_logits.sum(axis=0)  # (C,)
            grad_tables = np.zeros_like(tables)
            # For each feature j, each class c, each value v: accumulate grad_logits[b, c] where xb[b, j] == v
            # Vectorized: for each j, we can use np.add.at
            for j in range(d):
                vals = xb[:, j]  # (batch,)
                for c in range(C):
                    np.add.at(grad_tables[c, j], vals, grad_logits[:, c])
            # L2 regularization
            grad_tables += l2 * tables
            grad_bias += l2 * bias
            # Adam update
            t += 1
            for param, grad, m, v in [
                (tables, grad_tables, m_tables, v_tables),
                (bias, grad_bias, m_bias, v_bias)
            ]:
                m *= beta1
                m += (1 - beta1) * grad
                v *= beta2
                v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # Validation will be done outside this function
    return tables, bias


def evaluate(X, y, tables, bias, C, d):
    logits = bias[None, :] + np.sum(tables[:, X.T, np.arange(d)], axis=1).T
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # Split: last 20% for tuning (validation)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)

    best_config = None
    best_val_acc = -1.0
    best_tables = None
    best_bias = None

    # Grid search on tuning split
    for lr in LRS:
        for l2 in L2S:
            # Train on training split with early stopping based on validation
            tables = rng.normal(0, 0.01, size=(C, d, 256)).astype(np.float32)
            bias = np.zeros(C, dtype=np.float32)
            m_tables, v_tables = np.zeros_like(tables), np.zeros_like(tables)
            m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
            t = 0
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            patience = 0
            local_best_val = -1.0
            local_best_tables, local_best_bias = None, None

            for epoch in range(MAX_EPOCHS):
                order = rng.permutation(len(X_train))
                for s in range(0, len(X_train), BATCH):
                    idx = order[s:s + BATCH]
                    xb, yb = X_train[idx], y_train[idx]
                    logits = bias[None, :] + np.sum(tables[:, xb.T, np.arange(d)], axis=1).T
                    grad_logits = cross_entropy_grad(logits, yb)
                    grad_bias = grad_logits.sum(axis=0)
                    grad_tables = np.zeros_like(tables)
                    for j in range(d):
                        vals = xb[:, j]
                        for c in range(C):
                            np.add.at(grad_tables[c, j], vals, grad_logits[:, c])
                    grad_tables += l2 * tables
                    grad_bias += l2 * bias
                    t += 1
                    for param, grad, m, v in [
                        (tables, grad_tables, m_tables, v_tables),
                        (bias, grad_bias, m_bias, v_bias)
                    ]:
                        m *= beta1
                        m += (1 - beta1) * grad
                        v *= beta2
                        v += (1 - beta2) * grad * grad
                        m_hat = m / (1 - beta1 ** t)
                        v_hat = v / (1 - beta2 ** t)
                        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
                # Validate
                val_acc = evaluate(X_val, y_val, tables, bias, C, d)
                if val_acc > local_best_val:
                    local_best_val = val_acc
                    local_best_tables = tables.copy()
                    local_best_bias = bias.copy()
                    patience = 0
                else:
                    patience += 1
                    if patience >= PATIENCE:
                        break
            if local_best_val > best_val_acc:
                best_val_acc = local_best_val
                best_config = (lr, l2)
                best_tables = local_best_tables
                best_bias = local_best_bias

    # Retrain on full data with best config
    lr, l2 = best_config
    rng = np.random.default_rng(SEED)  # reset seed for final training
    tables = rng.normal(0, 0.01, size=(C, d, 256)).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)
    m_tables, v_tables = np.zeros_like(tables), np.zeros_like(tables)
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    # Train for a fixed number of epochs (or until convergence) on full data
    # We'll use the same early stopping but on a small held-out? Since we must train on all, we just run a fixed number of epochs.
    # Use the same max_epochs but no validation.
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits = bias[None, :] + np.sum(tables[:, xb.T, np.arange(d)], axis=1).T
            grad_logits = cross_entropy_grad(logits, yb)
            grad_bias = grad_logits.sum(axis=0)
            grad_tables = np.zeros_like(tables)
            for j in range(d):
                vals = xb[:, j]
                for c in range(C):
                    np.add.at(grad_tables[c, j], vals, grad_logits[:, c])
            grad_tables += l2 * tables
            grad_bias += l2 * bias
            t += 1
            for param, grad, m, v in [
                (tables, grad_tables, m_tables, v_tables),
                (bias, grad_bias, m_bias, v_bias)
            ]:
                m *= beta1
                m += (1 - beta1) * grad
                v *= beta2
                v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    # Format: magic (0x47414D31), d (int32), C (int32), tables (C*d*256 float32), bias (C float32)
    magic = 0x47414D31  # "GAM1"
    header = struct.pack("<3i", magic, d, C)
    tables_bytes = tables.astype("<f4").tobytes()
    bias_bytes = bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + tables_bytes + bias_bytes)


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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x47414D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 12 + 4L * C * d * 256 + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *tables = (const float *)(hd + 3);
    const float *bias = tables + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *t = tables + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```