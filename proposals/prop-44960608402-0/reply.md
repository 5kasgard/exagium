```hypothesis
Family: label-fm (factorization machine). It models pairwise feature interactions via low-rank embeddings, which are crucial for tabular data like Adult (e.g., education × occupation). With a small embedding dimension (k=8) it adds only ~130 KB to the linear model and keeps inference at O(d·k). If dev accuracy does not exceed label-linear (0.8524) or if inference cost per item rises above label-linear, the approach fails.
```

```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only. Encodes each column as 256 one-hot features.
Model: logit = bias + sum_j w_j[x_j] + 0.5*( (sum_j v_j[x_j])^2 - sum_j v_j[x_j]^2 )
Trains with Adam on logistic loss, L2 on w and V. Hyperparameters (k, lr, l2, epochs) chosen on last 20% tuning split."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_fm(X, y, k, lr, l2, epochs, batch, rng):
    n, d = X.shape
    # parameters
    bias = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, k))).astype(np.float32)
    # Adam state
    mb, vb = 0.0, 0.0
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = xb.shape[0]
            # forward
            linear = bias + np.sum(w[np.arange(d), xb.T], axis=0)          # (b,)
            emb = V[np.arange(d), xb.T]                                     # (d, b, k)
            emb_sum = np.sum(emb, axis=0)                                   # (b, k)
            emb_sq_sum = np.sum(emb * emb, axis=0)                          # (b, k)
            interaction = 0.5 * np.sum(emb_sum * emb_sum - emb_sq_sum, axis=1)  # (b,)
            logits = linear + interaction
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logit = (p - yb).astype(np.float32)                        # (b,)
            # gradients
            g_bias = grad_logit.mean()
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)
            # vectorized gradient accumulation
            for j in range(d):
                vals = xb[:, j]                                             # (b,)
                np.add.at(g_w[j], vals, grad_logit)
                # V gradient: grad_logit[:, None] * (emb_sum - emb[j])
                term = grad_logit[:, None] * (emb_sum - emb[j])             # (b, k)
                np.add.at(g_V[j], vals, term)
            g_w /= b
            g_V /= b
            # L2 regularization
            g_w += 2 * l2 * w
            g_V += 2 * l2 * V
            # Adam updates
            t += 1
            # bias
            mb = beta1 * mb + (1 - beta1) * g_bias
            vb = beta2 * vb + (1 - beta2) * g_bias * g_bias
            m_bc = mb / (1 - beta1 ** t)
            v_bc = vb / (1 - beta2 ** t)
            bias -= lr * m_bc / (np.sqrt(v_bc) + eps)
            # w
            mw = beta1 * mw + (1 - beta1) * g_w
            vw = beta2 * vw + (1 - beta2) * g_w * g_w
            m_wc = mw / (1 - beta1 ** t)
            v_wc = vw / (1 - beta2 ** t)
            w -= lr * m_wc / (np.sqrt(v_wc) + eps)
            # V
            mV = beta1 * mV + (1 - beta1) * g_V
            vV = beta2 * vV + (1 - beta2) * g_V * g_V
            m_Vc = mV / (1 - beta1 ** t)
            v_Vc = vV / (1 - beta2 ** t)
            V -= lr * m_Vc / (np.sqrt(v_Vc) + eps)
    return bias, w, V

def accuracy(X, y, bias, w, V):
    n, d = X.shape
    linear = bias + np.sum(w[np.arange(d), X.T], axis=0)
    emb = V[np.arange(d), X.T]
    emb_sum = np.sum(emb, axis=0)
    emb_sq_sum = np.sum(emb * emb, axis=0)
    interaction = 0.5 * np.sum(emb_sum * emb_sum - emb_sq_sum, axis=1)
    logits = linear + interaction
    pred = (logits > 0).astype(np.int64)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    # tuning split: last 20%
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    # hyperparameter grid
    best_acc, best_params, best_model = -1, None, None
    for k in (4, 8, 16):
        for lr in (1e-2, 3e-3, 1e-3):
            for l2 in (1e-4, 1e-3):
                rng = np.random.default_rng(SEED)
                epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, split)))
                bias, w, V = train_fm(X_tr, y_tr, k, lr, l2, epochs, BATCH, rng)
                acc = accuracy(X_tune, y_tune, bias, w, V)
                if acc > best_acc:
                    best_acc, best_params, best_model = acc, (k, lr, l2, epochs), (bias, w, V)
    # retrain on full data with best params
    k, lr, l2, epochs = best_params
    rng = np.random.default_rng(SEED)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, n)))
    bias, w, V = train_fm(X, y, k, lr, l2, epochs_full, BATCH, rng)
    # write weights: magic, d, k, C=2, bias, w[d*256], V[d*256*k] (all float32 little-endian)
    magic = 0x464D3130  # "FM10"
    C = 2
    out = struct.pack("<4i", magic, d, k, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per column). Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || k <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * k);
    float *emb_sq_sum = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        float linear = bias;
        memset(emb_sum, 0, sizeof(float) * k);
        memset(emb_sq_sum, 0, sizeof(float) * k);
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            linear += w[(long)j * 256 + v];
            const float *vj = V + ((long)j * 256 + v) * k;
            for (int u = 0; u < k; u++) {
                float val = vj[u];
                emb_sum[u] += val;
                emb_sq_sum[u] += val * val;
            }
        }
        float interaction = 0.0f;
        for (int u = 0; u < k; u++) interaction += emb_sum[u] * emb_sum[u] - emb_sq_sum[u];
        interaction *= 0.5f;
        float logit = linear + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```