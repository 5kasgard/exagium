```hypothesis
We propose label-fm, a factorization machine with rank-k pairwise interactions. The Adult dataset has strong feature interactions (e.g., education × occupation, marital-status × relationship) that a linear model cannot capture. A low-rank FM adds O(d·k) parameters and computes interactions in O(d·k) time, so it should improve accuracy over label-linear while staying fast. If dev accuracy does not exceed label-linear (0.8524) or inference cost per item grows by more than 2×, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid (chosen on the tuning split, last 20% of train.in)
GRID = {
    'k': [4, 8, 16],
    'lr': [1e-3, 3e-3],
    'l2': [1e-4, 1e-3],
    'epochs': [30, 50],
    'batch': [64, 256]
}
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types):
    """Return Z (n x d_full), encodings list (256 x dim_j), and offsets list."""
    encodings = []
    offsets = []
    off = 0
    for j, t in enumerate(types):
        col = X[:, j]
        if t == "ordinal":
            v = col.astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            scaled = (np.arange(256) - mu) / sd
            E = scaled[:, None]                     # (256, 1)
            dim = 1
        else:
            vals = np.unique(col)
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            dim = len(vals)
        encodings.append(E)
        offsets.append((off, dim))
        off += dim
    n = X.shape[0]
    Z = np.zeros((n, off), dtype=np.float32)
    for j, (off_j, dim_j) in enumerate(offsets):
        Z[:, off_j:off_j + dim_j] = encodings[j][X[:, j]]
    return Z, encodings, offsets


def train_fm(Z, y, k, lr, l2, epochs, batch, rng):
    """Train binary FM (labels -1/+1) with Adam. Returns w0, w, V."""
    n, d = Z.shape
    w0 = 0.0
    w = np.zeros(d, dtype=np.float32)
    V = rng.normal(0, 0.01, (d, k)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    V_sq_sum = np.sum(V * V, axis=1)   # (d,)  sum_f V_if^2

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            x = Z[idx]                 # (b, d)
            yy = y[idx]                # (b,) in {-1, 1}
            b = len(x)

            # Forward
            linear = w0 + x @ w                     # (b,)
            Vx = x @ V                              # (b, k)
            inter = 0.5 * (np.sum(Vx * Vx, axis=1) - x @ (V_sq_sum * (x * x).sum(axis=1)?))  # Wait, need correct formula
            # Actually inter = 0.5 * sum_f ( (sum_i V_if x_i)^2 - sum_i V_if^2 x_i^2 )
            # We can compute as:
            # term1 = np.sum(Vx * Vx, axis=1)  # (b,)
            # term2 = np.sum((x * x) * V_sq_sum, axis=1)  # (b,)
            # inter = 0.5 * (term1 - term2)
            term1 = np.sum(Vx * Vx, axis=1)
            term2 = np.sum((x * x) * V_sq_sum, axis=1)
            inter = 0.5 * (term1 - term2)
            scores = linear + inter

            # Logistic loss gradient
            z = -yy * scores
            sig = 1.0 / (1.0 + np.exp(-z))          # sigmoid(z)
            grad_score = -yy * sig                  # (b,)

            # Gradients
            grad_w0 = np.mean(grad_score)
            grad_w = (x.T @ grad_score) / b + l2 * w

            # dL/dV = (x.T @ (grad_score[:,None] * Vx)) / b - V * ((x**2).T @ grad_score)[:,None] / b + l2 * V
            G = grad_score
            termA = (x.T @ (G[:, None] * Vx)) / b
            termB = ((x * x).T @ G) / b             # (d,)
            grad_V = termA - V * termB[:, None] + l2 * V

            # Adam updates
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * grad_w0 * grad_w0
            w0 -= lr * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            w -= lr * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            V -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

            V_sq_sum = np.sum(V * V, axis=1)

    return w0, w, V


def evaluate(Z, y, w0, w, V):
    linear = w0 + Z @ w
    Vx = Z @ V
    term1 = np.sum(Vx * Vx, axis=1)
    V_sq_sum = np.sum(V * V, axis=1)
    term2 = np.sum((Z * Z) * V_sq_sum, axis=1)
    inter = 0.5 * (term1 - term2)
    scores = linear + inter
    preds = (scores > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    y_bin = 2 * y - 1                     # 0 -> -1, 1 -> +1
    types = card["input"]["types"]

    Z, encodings, offsets = encode_features(X, types)
    n, d_full = Z.shape

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    Z_train, Z_val = Z[:split], Z[split:]
    y_train, y_val = y_bin[:split], y_bin[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_model = None

    for k in GRID['k']:
        for lr in GRID['lr']:
            for l2 in GRID['l2']:
                for epochs in GRID['epochs']:
                    for batch in GRID['batch']:
                        w0, w, V = train_fm(Z_train, y_train, k, lr, l2, epochs, batch, rng)
                        acc = evaluate(Z_val, y_val, w0, w, V)
                        if acc > best_acc:
                            best_acc = acc
                            best_params = (k, lr, l2, epochs, batch)
                            best_model = (w0, w, V)

    # Retrain on full data
    k, lr, l2, epochs, batch = best_params
    w0, w, V = train_fm(Z, y_bin, k, lr, l2, epochs, batch, rng)

    # Build per-byte lookup tables for inference
    d_bytes = card["input"]["bytes"]
    linear_table = np.zeros((d_bytes, 256), dtype=np.float32)
    factor_table = np.zeros((d_bytes, 256, k), dtype=np.float32)
    for j, (off, dim) in enumerate(offsets):
        E = encodings[j]                          # (256, dim)
        linear_table[j] = E @ w[off:off + dim]
        factor_table[j] = E @ V[off:off + dim]

    # Write weights.bin
    magic = 0x464D4C31   # 'FML1'
    out = struct.pack("<4i", magic, d_bytes, k, 2)   # C=2
    out += struct.pack("<f", w0)
    out += linear_table.astype("<f4").tobytes()
    out += factor_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) {
        fprintf(stderr, "short read %s\n", path);
        exit(3);
    }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | (buf[*pos + 1] << 8) | (buf[*pos + 2] << 16) | ((uint32_t)buf[*pos + 3] << 24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, (k >> 8) & 255, (k >> 16) & 255, (k >> 24) & 255};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: infer weights.bin in.rec N\n");
        return 2;
    }
    long nw, nx = 0;
    long N = atol(argv[3]);
    int32_t *hdr = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hdr[0] != 0x464D4C31) {
        fprintf(stderr, "bad weights magic\n");
        return 3;
    }
    int d = hdr[1];
    int k = hdr[2];
    int C = hdr[3];
    if (C != 2 || d <= 0 || k <= 0) {
        fprintf(stderr, "bad header values\n");
        return 3;
    }
    long expected = 16 + 4L * (1 + (long)d * 256 * (1 + k));
    if (nw != expected) {
        fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected);
        return 3;
    }
    const float *p = (const float *)(hdr + 4);
    float bias = *p++;
    const float *linear_table = p;
    p += (long)d * 256;
    const float *factor_table = p;

    unsigned char *x = NULL;
    if (N > 0) {
        x = slurp(argv[2], &nx);
    }
    long pos = 0;

    float *factor_sum = malloc(k * sizeof(float));
    float *factor_sq_sum = malloc(k * sizeof(float));
    if (!factor_sum || !factor_sq_sum) {
        fprintf(stderr, "malloc failed\n");
        return 3;
    }

    for (long i = 0; i < N; i++) {
        uint32_t rec_len;
        const unsigned char *rec = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) {
            fprintf(stderr, "item %ld has %u bytes, expected %d\n", i, rec_len, d);
            return 3;
        }
        float score = bias;
        for (int f = 0; f < k; f++) {
            factor_sum[f] = 0.0f;
            factor_sq_sum[f] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            unsigned char v = rec[j];
            score += linear_table[(long)j * 256 + v];
            const float *frow = factor_table + ((long)j * 256 + v) * k;
            for (int f = 0; f < k; f++) {
                float fc = frow[f];
                factor_sum[f] += fc;
                factor_sq_sum[f] += fc * fc;
            }
        }
        for (int f = 0; f < k; f++) {
            score += 0.5f * (factor_sum[f] * factor_sum[f] - factor_sq_sum[f]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(factor_sum);
    free(factor_sq_sum);
    return 0;
}
```