```hypothesis
Family: label-fm (Factorization Machine for tabular data).
Why: FMs capture pairwise feature interactions efficiently via low-rank embeddings, which suits the mixed ordinal/categorical Adult data. The linear model (label-linear) ignores interactions; GBDT and MLP are heavier. FM with k=8 adds only ~14*256*8=28k params, inference O(d*k) ~ 112 mults per item, far cheaper than MLP/GBDT.
Failure signal: If dev accuracy <= 0.8524 (label-linear) or instructions/item > 2000, the model does not meet the goal.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then trained on all data.
Weights format: int32 magic=0x464D3132, d, k; float32 w0, w[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.03, 0.1, 0.3]
L2_CHOICES = [1e-4, 1e-3]
TUNE_EPOCHS = 5
FINAL_EPOCHS = 20
BATCH = 256
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

def train_fm(X, y, d, k, lr, l2, epochs, rng):
    n = X.shape[0]
    # Initialize parameters
    w0 = np.float32(0.0)
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]
            # Forward
            # Linear part
            linear = w0 + np.sum(w[np.arange(d), xb.T].T, axis=1)  # (B,)
            # Interaction part
            V_sel = V[np.arange(d), xb.T].transpose(1, 0, 2)       # (B, d, k)
            s_v = V_sel.sum(axis=1)                                # (B, k)
            s2_v = (V_sel**2).sum(axis=1)                          # (B, k)
            interaction = 0.5 * ((s_v**2 - s2_v).sum(axis=1))      # (B,)
            logit = linear + interaction
            p = sigmoid(logit)
            # Loss gradient
            dlogit = (p - yb) / B                                   # (B,)
            # Gradients
            dw0 = dlogit.sum()
            dw = np.zeros_like(w)
            dV = np.zeros_like(V)
            for j in range(d):
                np.add.at(dw[j], xb[:, j], dlogit)
                diff = s_v - V_sel[:, j]                           # (B, k)
                grad = dlogit[:, None] * diff                      # (B, k)
                np.add.at(dV[j], xb[:, j], grad)
            # L2 regularization
            dw += 2 * l2 * w
            dV += 2 * l2 * V
            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * dw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * dw0 * dw0
            m_hat = m_w0 / (1 - beta1**t)
            v_hat = v_w0 / (1 - beta2**t)
            w0 -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # w
            m_w = beta1 * m_w + (1 - beta1) * dw
            v_w = beta2 * v_w + (1 - beta2) * dw * dw
            m_hat = m_w / (1 - beta1**t)
            v_hat = v_w / (1 - beta2**t)
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * dV * dV
            m_hat = m_V / (1 - beta1**t)
            v_hat = v_V / (1 - beta2**t)
            V -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return w0, w, V

def evaluate_fm(X, y, d, k, w0, w, V):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        xb = X[s:s+BATCH]
        yb = y[s:s+BATCH]
        B = xb.shape[0]
        linear = w0 + np.sum(w[np.arange(d), xb.T].T, axis=1)
        V_sel = V[np.arange(d), xb.T].transpose(1, 0, 2)
        s_v = V_sel.sum(axis=1)
        s2_v = (V_sel**2).sum(axis=1)
        interaction = 0.5 * ((s_v**2 - s2_v).sum(axis=1))
        logit = linear + interaction
        pred = (logit > 0).astype(np.int64)
        correct += (pred == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20%
    tune_size = int(0.2 * n)
    X_train, X_tune = X[:-tune_size], X[-tune_size:]
    y_train, y_tune = y[:-tune_size], y[-tune_size:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_weights = None
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                w0, w, V = train_fm(X_train, y_train, d, k, lr, l2, TUNE_EPOCHS, rng)
                acc = evaluate_fm(X_tune, y_tune, d, k, w0, w, V)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (k, lr, l2)
                    best_weights = (w0, w, V)
    # Retrain on full data with best params
    k, lr, l2 = best_params
    rng = np.random.default_rng(SEED)
    w0, w, V = train_fm(X, y, d, k, lr, l2, FINAL_EPOCHS, rng)
    # Write weights
    magic = 0x464D3132
    header = struct.pack("<3i", magic, d, k)
    payload = b""
    payload += np.array([w0], dtype=np.float32).tobytes()
    payload += w.astype(np.float32).tobytes()
    payload += V.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(header + payload)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label-fm family.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights header). Output: one byte class (0 or 1). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2];
    if (d <= 0 || d > 256 || k <= 0 || k > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 12 + 4L * (1 + (long)d*256 + (long)d*256*k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 3);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = p + 1 + (long)d*256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *s_v = (float *)malloc(sizeof(float) * k);
    float *s2_v = (float *)malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float logit = w0;
        for (int j = 0; j < d; j++) logit += w[(long)j*256 + xi[j]];
        memset(s_v, 0, k*sizeof(float));
        memset(s2_v, 0, k*sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j*256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float val = v[f];
                s_v[f] += val;
                s2_v[f] += val * val;
            }
        }
        for (int f = 0; f < k; f++) logit += 0.5f * (s_v[f]*s_v[f] - s2_v[f]);
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```