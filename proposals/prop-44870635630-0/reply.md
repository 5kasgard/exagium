```hypothesis
Family: label-fm (degree-2 Factorization Machine with per-value embeddings).
Reason: Adult has 14 byte-encoded features (mixed ordinal/categorical). A linear model misses pairwise interactions; a full MLP is heavier. FM captures all pairwise interactions via low-rank embeddings (k=8) with O(d*k) inference, fitting the CPU budget. It should beat label-linear (0.8524) and approach label-gbdt (0.8664) at lower cost.
Failure signal: dev accuracy ≤ 0.8524 or Ir/item > 7570 (label-gbdt cost).
```

```python
"""Factorization Machine for tabular label tasks with byte features.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format: magic 0x464D4C31, d, k, C, w0, w[d][256], V[d][256][k] (all float32)."""
import json, math, struct, sys

import numpy as np

SEED = 1
GRID = {
    'k': [4, 8, 16],
    'lr': [0.01, 0.005],
    'epochs': [10, 20],
    'l2': [1e-4],
    'batch': [64],
}
TUNE_FRAC = 0.2  # last 20% for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(idx, w0, w, V):
    """idx: (B, d) uint8 -> logits (B,)"""
    B, d = idx.shape
    # linear part
    lin = w0 + np.sum(w[np.arange(d), idx], axis=1)  # (B,)
    # interaction part
    V_batch = V[np.arange(d), idx]  # (B, d, k)
    sum_V = np.sum(V_batch, axis=1)  # (B, k)
    sum_V_sq = np.sum(V_batch * V_batch, axis=1)
    inter = 0.5 * np.sum(sum_V * sum_V - sum_V_sq, axis=1)
    return lin + inter


def train_fm(X, y, k, lr, epochs, l2, batch, rng):
    n, d = X.shape
    # init
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, k)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # forward
            V_batch = V[np.arange(d), xb]  # (B, d, k)
            sum_V = np.sum(V_batch, axis=1)
            sum_V_sq = np.sum(V_batch * V_batch, axis=1)
            lin = w0 + np.sum(w[np.arange(d), xb], axis=1)
            inter = 0.5 * np.sum(sum_V * sum_V - sum_V_sq, axis=1)
            logit = lin + inter
            p = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = (p - yb).astype(np.float32)  # (B,)
            # gradients
            # w0
            gw0 = np.mean(grad_logit)
            # w
            gw = np.zeros_like(w)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(gw[j], vals, grad_logit)
            gw /= batch
            gw += l2 * w
            # V
            gV = np.zeros_like(V)
            for j in range(d):
                vals = xb[:, j]
                # sum_V without j-th feature
                sum_V_excl = sum_V - V_batch[:, j, :]
                # gradient for V[j, vals, :] is grad_logit[:, None] * sum_V_excl
                for b in range(len(idx)):
                    v_idx = vals[b]
                    gV[j, v_idx] += grad_logit[b] * sum_V_excl[b]
            gV /= batch
            gV += l2 * V
            # Adam updates
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m_hat = m_w0 / (1 - beta1 ** t)
            v_hat = v_w0 / (1 - beta2 ** t)
            w0 -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            m_hat = m_w / (1 - beta1 ** t)
            v_hat = v_w / (1 - beta2 ** t)
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            m_hat = m_V / (1 - beta1 ** t)
            v_hat = v_V / (1 - beta2 ** t)
            V -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return w0, w, V


def evaluate(X, y, w0, w, V):
    logit = fm_forward(X, w0, w, V)
    pred = (logit > 0).astype(np.int64)
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    tune_size = int(n * TUNE_FRAC)
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_acc = -1.0
    best_params = None
    best_model = None
    for k in GRID['k']:
        for lr in GRID['lr']:
            for epochs in GRID['epochs']:
                for l2 in GRID['l2']:
                    for batch in GRID['batch']:
                        w0, w, V = train_fm(X_train, y_train, k, lr, epochs, l2, batch, rng)
                        acc = evaluate(X_tune, y_tune, w0, w, V)
                        if acc > best_acc:
                            best_acc = acc
                            best_params = (k, lr, epochs, l2, batch)
                            best_model = (w0, w, V)
    print(f"Tuned: k={best_params[0]}, lr={best_params[1]}, epochs={best_params[2]}, l2={best_params[3]}, batch={best_params[4]}, acc={best_acc:.4f}", file=sys.stderr)
    # retrain on full data with best params
    k, lr, epochs, l2, batch = best_params
    w0, w, V = train_fm(X, y, k, lr, epochs, l2, batch, rng)
    # save weights
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, k, C)
    out = header
    out += np.array([w0], dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (feature values 0..255). Output: one byte class index (0 or 1). */
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
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || k <= 0 || d > 256 || k > 256) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(wb + 16);
    float w0 = *p++;                     // scalar bias
    const float *w = p; p += d * 256;    // w[d][256]
    const float *V = p;                  // V[d][256][k]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld: %u bytes != %d\n", i, len, d); return 3; }
        // linear part
        float logit = w0;
        for (int j = 0; j < d; j++) logit += w[j * 256 + xi[j]];
        // interaction part
        memset(sum_V, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            const float *v = V + (j * 256 + xi[j]) * k;
            for (int u = 0; u < k; u++) sum_V[u] += v[u];
        }
        float sum_sq = 0.0f, sq_sum = 0.0f;
        for (int u = 0; u < k; u++) {
            float val = sum_V[u];
            sum_sq += val * val;
            // subtract squared V for this feature
            for (int j = 0; j < d; j++) {
                float v = V[(j * 256 + xi[j]) * k + u];
                sq_sum += v * v;
            }
        }
        logit += 0.5f * (sum_sq - sq_sum);
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```