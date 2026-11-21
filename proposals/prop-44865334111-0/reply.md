```hypothesis
Family: label-fm (Factorization Machine with per-field embeddings).
Why: Adult has 14 fields (6 ordinal, 8 categorical) each already quantized to 256 values. A rank-4 FM captures pairwise field interactions with only 14*256*4 ≈ 14k extra parameters, staying fast at inference. It should beat linear (0.8524) and approach GBT (0.8664) at a fraction of GBT's cost.
Failure signal: dev accuracy ≤ 0.853 or Ir/item > 2000.
```

```python
"""Factorization Machine for tabular byte fields.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Trains on tuning split (last 20%) to pick embedding dim k and L2, then retrains on all data.
Weights format: magic 0x464D31, d, k, C; float32 w0, w[d*256], v[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K_CHOICES = [2, 4, 8]
L2_CHOICES = [1e-4, 1e-3, 1e-2]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(X, w0, w, v, k):
    # X: (n, d) uint8
    n, d = X.shape
    linear = w0 + np.sum(w[np.arange(d), X], axis=1).astype(np.float32)
    inter = np.zeros(n, np.float32)
    for f in range(k):
        vf = v[:, :, f]          # (d, 256)
        sum_v = np.sum(vf[np.arange(d), X], axis=1)
        sum_v2 = np.sum(vf[np.arange(d), X] ** 2, axis=1)
        inter += 0.5 * (sum_v * sum_v - sum_v2)
    return linear + inter

def train_fm(X, y, k, l2, rng, epochs, batch, lr):
    n, d = X.shape
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), np.float32)
    v = (rng.normal(0, 0.01, (d, 256, k))).astype(np.float32)
    params = [w0, w, v]
    shapes = [(), w.shape, v.shape]
    # Adam state
    m = [np.zeros(s, np.float32) for s in shapes]
    v_adam = [np.zeros(s, np.float32) for s in shapes]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xi, yi = X[idx], y[idx]
            # forward
            linear = w0 + np.sum(w[np.arange(d), xi], axis=1)
            inter = np.zeros(len(xi), np.float32)
            for f in range(k):
                vf = v[:, :, f]
                sum_v = np.sum(vf[np.arange(d), xi], axis=1)
                sum_v2 = np.sum(vf[np.arange(d), xi] ** 2, axis=1)
                inter += 0.5 * (sum_v * sum_v - sum_v2)
            logits = linear + inter
            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logits))
            g = (p - yi).astype(np.float32) / len(yi)
            # gradients
            gw0 = np.float32(g.sum())
            gw = np.zeros_like(w)
            gv = np.zeros_like(v)
            for j in range(d):
                vals = xi[:, j]
                np.add.at(gw[j], vals, g)
                for f in range(k):
                    vjf = v[j, :, f]
                    sum_v = np.sum(v[j, xi[:, j], f], axis=0)  # scalar per example
                    # derivative of 0.5*(sum_v^2 - sum_v2) w.r.t v[j,val,f] = sum_v - v[j,val,f]
                    # vectorized per example
                    for i_ex, val in enumerate(vals):
                        gv[j, val, f] += g[i_ex] * (sum_v[i_ex] - vjf[val])
            # L2
            gw += l2 * w
            gv += l2 * v
            grads = [gw0, gw, gv]
            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v_adam[i] / (1 - beta2 ** t)
                if i == 0:
                    params[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                else:
                    p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params[0], params[1], params[2]

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    tune_start = int(0.8 * n)
    X_tr, X_te = X[:tune_start], X[tune_start:]
    y_tr, y_te = y[:tune_start], y[tune_start:]

    rng = np.random.default_rng(SEED)
    best_acc, best_k, best_l2 = -1, 4, 1e-3
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
    for k in K_CHOICES:
        for l2 in L2_CHOICES:
            w0, w, v = train_fm(X_tr, y_tr, k, l2, rng, epochs, BATCH, LR)
            logits = fm_forward(X_te, w0, w, v, k)
            pred = (logits > 0).astype(np.int64)
            acc = (pred == y_te.astype(np.int64)).mean()
            if acc > best_acc:
                best_acc, best_k, best_l2 = acc, k, l2

    # retrain on all data
    w0, w, v = train_fm(X, y, best_k, best_l2, rng, min(MAX_EPOCHS, math.ceil(SAMPLES / n)), BATCH, LR)

    # write weights
    magic = 0x464D31  # "FM1"
    out = struct.pack("<4i", magic, d, best_k, C)
    out += np.array(w0, dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (field values 0..255). Output: one byte class (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k <= 0 || k > 16) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * (1 + d * 256 + d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char pred;

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "record %ld len %u != %d\n", i, rec_len, d); return 3; }

        float linear = w0;
        float sum_v[16];               // k <= 16
        float sum_v2[16];
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            linear += w[j * 256 + val];
            const float *vj = v + (j * 256 + val) * k;
            for (int f = 0; f < k; f++) {
                float vjf = vj[f];
                sum_v[f] += vjf;
                sum_v2[f] += vjf * vjf;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        float score = linear + inter;
        pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```