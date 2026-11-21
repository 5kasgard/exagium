```hypothesis
Family: label-fm (Factorization Machine). Adult income depends on feature interactions (e.g., education×occupation, marital-status×relationship) which linear models miss. FM learns 8-dim embeddings per feature value, capturing pairwise interactions via efficient dot-product trick. Expected: dev accuracy > 0.8524 (label-linear) with Ir/item < 1000. Fails if accuracy ≤ 0.8524 or Ir/item ≥ 1500.
```

```python
"""Factorization Machine for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Field-aware FM with k=8 embeddings per (field,value).
Ordinal/categorical columns both treated as 256-value fields (pre-quantized). Binary logistic loss, Adam,
L2=1e-4 on embeddings and linear weights. Early stopping on last 20% of train.in (tuning split), then
retrain on full train. Exports: magic, d, k, C; linear[d][256]; emb[d][256][k]; bias."""
import json, math, struct, sys

import numpy as np

K = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 50
VAL_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def accuracy(X, y, linear, emb, bias, d):
    n = X.shape[0]
    logits = np.full(n, bias, dtype=np.float32)
    sum_v = np.zeros((n, K), dtype=np.float32)
    sum_v2 = np.zeros((n, K), dtype=np.float32)
    for f in range(d):
        vals = X[:, f]
        logits += linear[f, vals]
        v = emb[f, vals, :]
        sum_v += v
        sum_v2 += v * v
    interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
    logits += interaction
    preds = (logits > 0).astype(np.uint8)
    return np.mean(preds == y)

def train_epoch(X, y, linear, emb, bias, linear_m, linear_v, emb_m, emb_v, bias_m, bias_v, rng, t, d):
    n = X.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, BATCH):
        idx = order[s:s+BATCH]
        xb = X[idx]
        yb = y[idx].astype(np.float32)
        bsz = len(yb)
        logits = np.full(bsz, bias, dtype=np.float32)
        sum_v = np.zeros((bsz, K), dtype=np.float32)
        sum_v2 = np.zeros((bsz, K), dtype=np.float32)
        for f in range(d):
            vals = xb[:, f]
            logits += linear[f, vals]
            v = emb[f, vals, :]
            sum_v += v
            sum_v2 += v * v
        interaction = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
        logits += interaction
        p = 1.0 / (1.0 + np.exp(-logits))
        grad = (p - yb).astype(np.float32)
        g_bias = grad.mean()
        g_linear = np.zeros_like(linear)
        g_emb = np.zeros_like(emb)
        for f in range(d):
            vals = xb[:, f]
            v = emb[f, vals, :]
            gl = np.zeros(256, dtype=np.float32)
            cnt = np.zeros(256, dtype=np.int32)
            np.add.at(gl, vals, grad)
            np.add.at(cnt, vals, 1)
            mask = cnt > 0
            gl[mask] /= cnt[mask]
            g_linear[f] = gl
            ge = np.zeros((256, K), dtype=np.float32)
            for lat in range(K):
                gvk = grad * (sum_v[:, lat] - v[:, lat])
                np.add.at(ge[:, lat], vals, gvk)
            ge[mask] /= cnt[mask, None]
            g_emb[f] = ge
        g_linear += L2 * linear
        g_emb += L2 * emb
        t += 1
        bias_m = 0.9 * bias_m + 0.1 * g_bias
        bias_v = 0.999 * bias_v + 0.001 * g_bias * g_bias
        bias -= LR * (bias_m / (1 - 0.9**t)) / (np.sqrt(bias_v / (1 - 0.999**t)) + 1e-8)
        linear_m = 0.9 * linear_m + 0.1 * g_linear
        linear_v = 0.999 * linear_v + 0.001 * g_linear * g_linear
        linear -= LR * (linear_m / (1 - 0.9**t)) / (np.sqrt(linear_v / (1 - 0.999**t)) + 1e-8)
        emb_m = 0.9 * emb_m + 0.1 * g_emb
        emb_v = 0.999 * emb_v + 0.001 * g_emb * g_emb
        emb -= LR * (emb_m / (1 - 0.9**t)) / (np.sqrt(emb_v / (1 - 0.999**t)) + 1e-8)
    return t, bias_m, bias_v, linear_m, linear_v, emb_m, emb_v

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)
    n = X.shape[0]
    n_val = int(n * VAL_FRAC)
    n_tr = n - n_val
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr:], y[n_tr:]
    rng = np.random.default_rng(SEED)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    bias = np.float32(0.0)
    linear_m = np.zeros((d, 256), dtype=np.float32)
    linear_v = np.zeros((d, 256), dtype=np.float32)
    emb_m = np.zeros((d, 256, K), dtype=np.float32)
    emb_v = np.zeros((d, 256, K), dtype=np.float32)
    bias_m = np.float32(0.0)
    bias_v = np.float32(0.0)
    t = 0
    best_val_acc = 0.0
    best_epoch = 0
    best_linear = None
    best_emb = None
    best_bias = 0.0
    for epoch in range(MAX_EPOCHS):
        t, bias_m, bias_v, linear_m, linear_v, emb_m, emb_v = train_epoch(
            X_tr, y_tr, linear, emb, bias, linear_m, linear_v, emb_m, emb_v, bias_m, bias_v, rng, t, d)
        val_acc = accuracy(X_val, y_val, linear, emb, bias, d)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_linear = linear.copy()
            best_emb = emb.copy()
            best_bias = bias
    linear, emb, bias = best_linear, best_emb, best_bias
    linear_m = np.zeros((d, 256), dtype=np.float32)
    linear_v = np.zeros((d, 256), dtype=np.float32)
    emb_m = np.zeros((d, 256, K), dtype=np.float32)
    emb_v = np.zeros((d, 256, K), dtype=np.float32)
    bias_m = np.float32(0.0)
    bias_v = np.float32(0.0)
    t = 0
    for epoch in range(best_epoch):
        t, bias_m, bias_v, linear_m, linear_v, emb_m, emb_v = train_epoch(
            X, y, linear, emb, bias, linear_m, linear_v, emb_m, emb_v, bias_m, bias_v, rng, t, d)
    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += struct.pack("<f", bias)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class 0 or 1).
 * Weights: magic 0x464D0001, d, k, C; linear[d][256]; emb[d][256][k]; bias. */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || K > 64 || d < 1 || d > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * (1 + K) + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + (long)d * 256;
    const float bias = emb[(long)d * 256 * K];
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = bias;
        for (int lat = 0; lat < K; lat++) { sum_v[lat] = 0.0f; sum_v2[lat] = 0.0f; }
        for (int f = 0; f < d; f++) {
            unsigned char val = xi[f];
            logit += linear[(long)f * 256 + val];
            const float *v = emb + ((long)f * 256 + val) * K;
            for (int lat = 0; lat < K; lat++) {
                float vf = v[lat];
                sum_v[lat] += vf;
                sum_v2[lat] += vf * vf;
            }
        }
        float inter = 0.0f;
        for (int lat = 0; lat < K; lat++) inter += sum_v[lat] * sum_v[lat] - sum_v2[lat];
        logit += 0.5f * inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```