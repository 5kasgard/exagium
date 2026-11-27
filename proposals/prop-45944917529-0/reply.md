```hypothesis
label-fm: Factorization Machine with per-field per-value factors (rank 8) capturing all pairwise feature interactions. Adult income depends on feature crosses (e.g., education×occupation, capital-gain×hours); FM learns these efficiently via low-rank dot products. Precomputed 256-entry lookup tables per field make inference ~250 FLOPs/item (vs ~70 for linear). If dev accuracy ≤ 0.855 (label-linear +0.003), interactions don't help this encoding.
```
```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    w_lin = np.zeros((d, 256, C), dtype=np.float32)
    w_fac = (rng.normal(0, 0.01, (d, 256, C, K))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    params = [w_lin, w_fac, bias]
    shapes = [p.shape for p in params]
    m = [np.zeros(s, dtype=np.float32) for s in shapes]
    v = [np.zeros(s, dtype=np.float32) for s in shapes]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            s_lin = np.broadcast_to(bias, (B, C)).copy()
            s_fac = np.zeros((B, C, K), dtype=np.float32)
            sq_fac = np.zeros((B, C, K), dtype=np.float32)

            for j in range(d):
                vj = xb[:, j]
                s_lin += w_lin[j, vj, :]
                s_fac += w_fac[j, vj, :, :]
                sq_fac += w_fac[j, vj, :, :] ** 2

            interaction = 0.5 * np.sum(s_fac * s_fac - sq_fac, axis=2)
            score = s_lin + interaction

            score_max = score.max(axis=1, keepdims=True)
            exp_score = np.exp(score - score_max)
            probs = exp_score / exp_score.sum(axis=1, keepdims=True)

            grad_score = probs
            grad_score[np.arange(B), yb] -= 1.0
            grad_score /= B

            dbias = grad_score.sum(axis=0)
            dw_lin = np.zeros_like(w_lin)
            dw_fac = np.zeros_like(w_fac)

            for j in range(d):
                vj = xb[:, j]
                for c in range(C):
                    np.add.at(dw_lin[j, :, c], vj, grad_score[:, c])
                    for f in range(K):
                        diff = s_fac[:, c, f] - w_fac[j, vj, c, f]
                        np.add.at(dw_fac[j, :, c, f], vj, grad_score[:, c] * diff)

            dw_lin += L2 * w_lin
            dw_fac += L2 * w_fac
            dbias += L2 * bias

            grads = [dw_lin, dw_fac, dbias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= beta1
                m[i] += (1 - beta1) * gr
                v[i] *= beta2
                v[i] += (1 - beta2) * gr * gr
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    out = struct.pack("<4i", 0x464D3131, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += w_lin.astype("<f4").tobytes()
    out += w_fac.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (C + (long)d * 256 * C + (long)d * 256 * C * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *w_lin = bias + C;
    const float *w_fac = w_lin + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *s_lin = malloc(sizeof(float) * C);
    float *s_fac = malloc(sizeof(float) * C * K);
    float *sq_fac = malloc(sizeof(float) * C * K);
    float score[256];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            s_lin[c] = bias[c];
            for (int f = 0; f < K; f++) {
                s_fac[c * K + f] = 0.0f;
                sq_fac[c * K + f] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *lin_ptr = w_lin + ((long)j * 256 + v) * C;
            const float *fac_ptr = w_fac + (((long)j * 256 + v) * C) * K;

            for (int c = 0; c < C; c++) {
                s_lin[c] += lin_ptr[c];
                for (int f = 0; f < K; f++) {
                    float val = fac_ptr[c * K + f];
                    s_fac[c * K + f] += val;
                    sq_fac[c * K + f] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float sf = s_fac[c * K + f];
                inter += sf * sf - sq_fac[c * K + f];
            }
            score[c] = s_lin[c] + 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    return 0;
}
```