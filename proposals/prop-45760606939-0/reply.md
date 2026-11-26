```hypothesis
Factorization Machine (FM) with small factor dimension (k=8) captures pairwise feature interactions
efficiently, which is crucial for tabular data with mixed ordinal/categorical features like Adult.
It adds only ~14*256*8 ≈ 28K parameters over linear, so inference stays fast.
If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D3130, d, k, C, w0, w[d][256], v[d][256][k] (all float32 little-endian)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 30


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "Binary classification only"

    rng = np.random.default_rng(SEED)
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0

    n = X.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bsz = xb.shape[0]

            # Forward
            linear = w0 + np.sum(w[np.arange(d), xb.T], axis=0)  # (bsz,)
            # FM interaction term: 0.5 * sum_f ( (sum_i v_i,f)^2 - sum_i v_i,f^2 )
            v_x = v[np.arange(d), xb.T]  # (d, bsz, K)
            sum_v = np.sum(v_x, axis=0)          # (bsz, K)
            sum_v2 = np.sum(v_x * v_x, axis=0)   # (bsz, K)
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)  # (bsz,)
            logits = linear + inter
            probs = sigmoid(logits)

            # Gradients (binary cross-entropy)
            grad = (probs - yb).astype(np.float32)  # (bsz,)
            # w0
            g_w0 = grad.mean()
            # w
            g_w = np.zeros_like(w)
            np.add.at(g_w, (np.arange(d), xb.T), grad / bsz)
            g_w += L2 * w
            # v
            g_v = np.zeros_like(v)
            # For each feature j, each example i, each factor f:
            # dL/dv_j,f = grad_i * (sum_v_i,f - v_j,xi_j,f)
            for j in range(d):
                vals = xb[:, j]  # (bsz,)
                v_j = v[j, vals]  # (bsz, K)
                # sum_v is (bsz, K)
                diff = (sum_v - v_j) * grad[:, None] / bsz  # (bsz, K)
                np.add.at(g_v[j], vals, diff)
            g_v += L2 * v

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # Adam updates
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v

            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)

            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # Pack weights
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (0 or 1). */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1 || d > 256) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + 4 + (size_t)d * 256 * 4 + (size_t)d * 256 * K * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;
    const float *w = p; p += (size_t)d * 256;
    const float *v = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float score = w0;
        for (int f = 0; f < K; f++) sum_v[f] = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            score += w[(size_t)j * 256 + val];
            const float *vj = v + ((size_t)j * 256 + val) * K;
            for (int f = 0; f < K; f++) sum_v[f] += vj[f];
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            float sv = sum_v[f];
            // subtract sum of squares
            float sq = 0.0f;
            for (int j = 0; j < d; j++) {
                unsigned char val = xi[j];
                float vjf = v[((size_t)j * 256 + val) * K + f];
                sq += vjf * vjf;
            }
            inter += 0.5f * (sv * sv - sq);
        }
        score += inter;
        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char pred = prob > 0.5f ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```