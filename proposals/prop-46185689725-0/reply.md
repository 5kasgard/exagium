```hypothesis
Factorization Machine (label-fm) with field-wise embeddings (k=8) captures pairwise feature interactions crucial for tabular data like Adult, where combinations (education×occupation, marital-status×relationship) are predictive. It adds only ~130K parameters over linear, trained via Adam on binary cross-entropy. If dev accuracy ≤ 0.8524 (label-linear) or inference cost ≥ 719 Ir/item, the approach fails.
```
```python
"""Factorization Machine for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Each of d byte fields gets a linear weight and k-dim embedding.
Logit = bias + Σ w[f][val] + ½(‖Σ v[f][val]‖² − Σ ‖v[f][val]‖²). Binary cross-entropy, Adam lr=1e-3, L2=1e-4."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1e-3, 1e-4, 256, 600000, 100
K = 8  # embedding dimension
MAGIC = 0x464D5F31  # "FM_1"

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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2, "FM binary only"
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # params: bias (1), w[d,256], v[d,256,K]
    bias = np.float32(0.0)
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    v = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    mb, mw, mv = np.float32(0.0), np.zeros_like(w), np.zeros_like(v)
    vb, vw, vv = np.float32(0.0), np.zeros_like(w), np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward
            linear = np.full(bsz, bias, dtype=np.float32)
            S = np.zeros((bsz, K), dtype=np.float32)
            sum_sq_v = np.zeros(bsz, dtype=np.float32)
            for f in range(d):
                vals = xb[:, f]
                linear += w[f, vals]
                v_f = v[f, vals]  # [bsz, K]
                S += v_f
                sum_sq_v += np.sum(v_f * v_f, axis=1)
            inter = 0.5 * (np.sum(S * S, axis=1) - sum_sq_v)
            logit = linear + inter

            # binary cross-entropy gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)  # [bsz]

            # gradients
            g_bias = grad.mean()
            g_w = np.zeros_like(w)
            g_v = np.zeros_like(v)
            for f in range(d):
                vals = xb[:, f]
                # linear part
                np.add.at(g_w[f], vals, grad)
                # interaction part
                # ∂inter/∂v[f,val] = (S - v[f,val]) * grad[:,None]
                for i in range(bsz):
                    val = vals[i]
                    g_v[f, val] += (S[i] - v[f, val]) * grad[i]
            g_w /= bsz
            g_v /= bsz
            g_w += L2 * w
            g_v += L2 * v

            t += 1
            # Adam update
            mb = beta1 * mb + (1 - beta1) * g_bias
            vb = beta2 * vb + (1 - beta2) * g_bias * g_bias
            bias -= LR * (mb / (1 - beta1 ** t)) / (np.sqrt(vb / (1 - beta2 ** t)) + eps)

            mw = beta1 * mw + (1 - beta1) * g_w
            vw = beta2 * vw + (1 - beta2) * g_w * g_w
            w -= LR * (mw / (1 - beta1 ** t)) / (np.sqrt(vw / (1 - beta2 ** t)) + eps)

            mv = beta1 * mv + (1 - beta1) * g_v
            vv = beta2 * vv + (1 - beta2) * g_v * g_v
            v -= LR * (mv / (1 - beta1 ** t)) / (np.sqrt(vv / (1 - beta2 ** t)) + eps)

    # pack weights: magic, d, K, C, bias, w[d*256], v[d*256*K]
    out = struct.pack("<4if", MAGIC, d, K, C, bias)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (0/1). */
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
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x464D5F31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t d = *(int32_t*)(wb + 4);
    int32_t K = *(int32_t*)(wb + 8);
    int32_t C = *(int32_t*)(wb + 12);
    float bias = *(float*)(wb + 16);
    if (C != 2 || d <= 0 || K <= 0 || d > 256 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *w = (const float *)(wb + 20);
    const float *v = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = (float *)malloc(sizeof(float) * K);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias;
        memset(S, 0, sizeof(float) * K);
        float sum_sq_v = 0.0f;

        for (int f = 0; f < d; f++) {
            unsigned char val = xi[f];
            const float *w_f = w + (long)f * 256;
            const float *v_f = v + (long)f * 256 * K;
            logit += w_f[val];
            const float *v_fv = v_f + (long)val * K;
            for (int kk = 0; kk < K; kk++) {
                float vv = v_fv[kk];
                S[kk] += vv;
                sum_sq_v += vv * vv;
            }
        }
        float dotS = 0.0f;
        for (int kk = 0; kk < K; kk++) dotS += S[kk] * S[kk];
        logit += 0.5f * (dotS - sum_sq_v);

        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```