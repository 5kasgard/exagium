```hypothesis
Family: label-fm (Factorization Machine). The Adult dataset has strong categorical interactions (education×occupation, marital-status×relationship) that a linear model misses. A low-rank FM (k=8) captures all pairwise interactions with O(d·k) extra params and compute, adding ~130 KB weights and ~112 FMA/item. It should beat label-linear (0.8524) on accuracy while staying well below label-mlp16/GBDT cost. Failure: dev accuracy ≤ 0.8524 or instructions/item > 2× label-linear.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Encodes ordinal as z-scored continuous, categorical as one-hot per training category.
FM with k=8 latent factors. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports per-byte lookup tables: magic 0x464D3130, d, k, C, w0; then for each of d features,
256 entries of (linear_weight, k latent_weights) as float32."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]
    assert d == 14 and C == 2

    rng = np.random.default_rng(SEED)

    is_ordinal = [t == "ordinal" for t in types]
    ordinal_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]

    mu = np.zeros(d, np.float64)
    sd = np.ones(d, np.float64)
    for j in ordinal_idx:
        v = X[:, j].astype(np.float64)
        mu[j] = v.mean()
        sd[j] = v.std() or 1.0

    cat_vals = []
    for j in cat_idx:
        vals = np.unique(X[:, j])
        vals = vals[vals != 255]
        cat_vals.append(vals)

    w0 = 0.0
    ord_w = np.zeros(len(ordinal_idx), np.float32)
    ord_v = rng.uniform(-0.01, 0.01, (len(ordinal_idx), K)).astype(np.float32)
    cat_w = [np.zeros(256, np.float32) for _ in cat_idx]
    cat_v = [np.zeros((256, K), np.float32) for _ in cat_idx]
    for ci, vals in enumerate(cat_vals):
        cat_v[ci][vals] = rng.uniform(-0.01, 0.01, (len(vals), K)).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_ord_w, v_ord_w = np.zeros_like(ord_w), np.zeros_like(ord_w)
    m_ord_v, v_ord_v = np.zeros_like(ord_v), np.zeros_like(ord_v)
    m_cat_w = [np.zeros(256, np.float32) for _ in cat_idx]
    v_cat_w = [np.zeros(256, np.float32) for _ in cat_idx]
    m_cat_v = [np.zeros((256, K), np.float32) for _ in cat_idx]
    v_cat_v = [np.zeros((256, K), np.float32) for _ in cat_idx]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]
            yi = y[idx]
            bs = len(yi)

            scores = np.full(bs, w0, dtype=np.float32)
            sum_f = np.zeros((bs, K), np.float32)
            sum_sq_f = np.zeros((bs, K), np.float32)

            for oi, j in enumerate(ordinal_idx):
                x = (xi[:, j].astype(np.float32) - mu[j]) / sd[j]
                scores += ord_w[oi] * x
                vx = ord_v[oi] * x[:, None]
                sum_f += vx
                sum_sq_f += vx * vx

            for ci, j in enumerate(cat_idx):
                vals = xi[:, j]
                scores += cat_w[ci][vals]
                vx = cat_v[ci][vals]
                sum_f += vx
                sum_sq_f += vx * vx

            fm = 0.5 * np.sum(sum_f * sum_f - sum_sq_f, axis=1)
            scores += fm

            p = 1.0 / (1.0 + np.exp(-scores))
            grad = (p - yi).astype(np.float32)

            gw0 = grad.mean()
            g_ord_w = np.zeros_like(ord_w)
            g_ord_v = np.zeros_like(ord_v)
            for oi, j in enumerate(ordinal_idx):
                x = (xi[:, j].astype(np.float32) - mu[j]) / sd[j]
                x2 = x * x
                g_ord_w[oi] = np.mean(grad * x)
                for f in range(K):
                    g_ord_v[oi, f] = np.mean(grad * (sum_f[:, f] * x - ord_v[oi, f] * x2))

            g_cat_w = [np.zeros(256, np.float32) for _ in cat_idx]
            g_cat_v = [np.zeros((256, K), np.float32) for _ in cat_idx]
            for ci, j in enumerate(cat_idx):
                vals = xi[:, j]
                np.add.at(g_cat_w[ci], vals, grad)
                g_cat_w[ci] /= bs
                for f in range(K):
                    diff = sum_f[:, f] - cat_v[ci][vals, f]
                    np.add.at(g_cat_v[ci][:, f], vals, grad * diff)
                g_cat_v[ci] /= bs

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m_hat = m_w0 / (1 - beta1 ** t)
            v_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_hat / (np.sqrt(v_hat) + eps)

            m_ord_w = beta1 * m_ord_w + (1 - beta1) * g_ord_w
            v_ord_w = beta2 * v_ord_w + (1 - beta2) * g_ord_w * g_ord_w
            m_ord_v = beta1 * m_ord_v + (1 - beta1) * g_ord_v
            v_ord_v = beta2 * v_ord_v + (1 - beta2) * g_ord_v * g_ord_v
            ord_w -= LR * (m_ord_w / (1 - beta1 ** t)) / (np.sqrt(v_ord_w / (1 - beta2 ** t)) + eps)
            ord_v -= LR * (m_ord_v / (1 - beta1 ** t)) / (np.sqrt(v_ord_v / (1 - beta2 ** t)) + eps)

            for ci in range(len(cat_idx)):
                m_cat_w[ci] = beta1 * m_cat_w[ci] + (1 - beta1) * g_cat_w[ci]
                v_cat_w[ci] = beta2 * v_cat_w[ci] + (1 - beta2) * g_cat_w[ci] * g_cat_w[ci]
                m_cat_v[ci] = beta1 * m_cat_v[ci] + (1 - beta1) * g_cat_v[ci]
                v_cat_v[ci] = beta2 * v_cat_v[ci] + (1 - beta2) * g_cat_v[ci] * g_cat_v[ci]
                cat_w[ci] -= LR * (m_cat_w[ci] / (1 - beta1 ** t)) / (np.sqrt(v_cat_w[ci] / (1 - beta2 ** t)) + eps)
                cat_v[ci] -= LR * (m_cat_v[ci] / (1 - beta1 ** t)) / (np.sqrt(v_cat_v[ci] / (1 - beta2 ** t)) + eps)

    linear_table = np.zeros((d, 256), np.float32)
    latent_table = np.zeros((d, 256, K), np.float32)

    for oi, j in enumerate(ordinal_idx):
        for v in range(256):
            x = (v - mu[j]) / sd[j]
            linear_table[j, v] = ord_w[oi] * x
            latent_table[j, v] = ord_v[oi] * x

    for ci, j in enumerate(cat_idx):
        linear_table[j] = cat_w[ci]
        latent_table[j] = cat_v[ci]

    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += struct.pack("<f", w0)
    out += linear_table.astype("<f4").tobytes()
    out += latent_table.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (0 or 1).
 * Weight format: magic 0x464D3130, int32 d, k, C, float32 w0,
 *   float32 linear[d][256], float32 latent[d][256][k]. */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
