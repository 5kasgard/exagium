```hypothesis
Family: label-fm (Factorization Machine, degree 2). All 14 features are quantized to 255 bins, so pairwise interactions are abundant and low-rank latent vectors capture them efficiently. It should beat label-linear by modeling interactions while keeping inference O(d*K) per item. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds ~2× label-linear, the hypothesis fails.
```
```python
"""Factorization Machine (degree 2) for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: int32 magic 0x464D4C42, d, K, C; float32 bias; float32 w[d][256]; float32 V[d][256][K]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    d = X.shape[1]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    bias = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    mb, mw, mV = np.float32(0.0), np.zeros_like(w), np.zeros_like(V)
    vb, vw, vV = np.float32(0.0), np.zeros_like(w), np.zeros_like(V)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward
            linear = bias + np.sum(w[np.arange(d), xb.T], axis=0)  # (bs,)
            sum_V = np.zeros((bs, K), dtype=np.float32)
            sum_V_sq = np.zeros((bs, K), dtype=np.float32)
            for j in range(d):
                vj = V[j, xb[:, j], :]  # (bs, K)
                sum_V += vj
                sum_V_sq += vj * vj
            inter = 0.5 * np.sum(sum_V * sum_V - sum_V_sq, axis=1)
            logit = linear + inter

            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = (p - yb).astype(np.float32)  # (bs,)

            # gradients
            g_bias = grad_logit.mean()
            g_w = np.zeros_like(w)
            g_V = np.zeros_like(V)

            for j in range(d):
                xj = xb[:, j]
                np.add.at(g_w[j], xj, grad_logit)
                vj = V[j, xj, :]
                # d_inter/d_vj = sum_V - vj
                g_vj = grad_logit[:, None] * (sum_V - vj)
                np.add.at(g_V[j], xj, g_vj)

            # L2
            g_w += L2 * w
            g_V += L2 * V

            t += 1
            # Adam updates
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # bias
            mb = beta1 * mb + (1 - beta1) * g_bias
            vb = beta2 * vb + (1 - beta2) * g_bias * g_bias
            bias -= LR * (mb / (1 - beta1 ** t)) / (np.sqrt(vb / (1 - beta2 ** t)) + eps)
            # w
            mw = beta1 * mw + (1 - beta1) * g_w
            vw = beta2 * vw + (1 - beta2) * g_w * g_w
            w -= LR * (mw / (1 - beta1 ** t)) / (np.sqrt(vw / (1 - beta2 ** t)) + eps)
            # V
            mV = beta1 * mV + (1 - beta1) * g_V
            vV = beta2 * vV + (1 - beta2) * g_V * g_V
            V -= LR * (mV / (1 - beta1 ** t)) / (np.sqrt(vV / (1 - beta2 ** t)) + eps)

    # write weights
    magic = 0x464D4C42  # 'BLMF' little-endian
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", bias)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree 2) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (feature values 0-255). Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D4C42) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || d < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = (float *)malloc(sizeof(float) * K);
    float *sum_V_sq = (float *)malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias;
        for (int j = 0; j < d; j++) logit += w[(long)j * 256 + xi[j]];

        for (int kk = 0; kk < K; kk++) { sum_V[kk] = 0.0f; sum_V_sq[kk] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *vj = V + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = vj[kk];
                sum_V[kk] += v;
                sum_V_sq[kk] += v * v;
            }
        }
        for (int kk = 0; kk < K; kk++) logit += 0.5f * (sum_V[kk] * sum_V[kk] - sum_V_sq[kk]);

        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    free(sum_V); free(sum_V_sq);
    return 0;
}
```