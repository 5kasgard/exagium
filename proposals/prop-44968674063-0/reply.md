```hypothesis
Factorization Machine (degree-2, rank-4) on the 14 one-hot fields. Adult has strong pairwise interactions (education×occupation, marital-status×relationship, capital-gain×hours) that linear models miss. With only 14 fields, the 91 pairwise dot-products per item (rank 4) add ~364 FMA ops – still near-linear latency. If dev accuracy ≤ 0.8524 or Ir/item > 1000, the hypothesis fails.
```
```python
"""Factorization Machine (degree-2, rank-k) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights: magic 0x464D3231, d, k, C; float32 w0, w[d][256], v[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 4                 # factor rank
SEED, LR, L2, BATCH, EPOCHS = 1, 0.02, 1e-5, 256, 12


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)   # 0 / 1
    d = X.shape[1]
    n = X.shape[0]
    C = card["classes"]                        # 2
    rng = np.random.default_rng(SEED)

    # Parameters
    w0 = np.float32(0.0)
    w  = np.zeros((d, 256), dtype=np.float32)
    v  = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    mw0, vw0 = 0.0, 0.0
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mv, vv = np.zeros_like(v), np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]                     # (b, d)
            yi = y[idx]                     # (b,)
            b = len(yi)

            # Forward
            linear = w0 + np.sum(w[np.arange(d), xi.T], axis=0)          # (b,)
            inter = 0.0
            for j in range(d):
                vj = v[j, xi[:, j]]           # (b, K)
                for l in range(j + 1, d):
                    vl = v[l, xi[:, l]]       # (b, K)
                    inter += np.sum(vj * vl, axis=1)
            logit = linear + inter
            p = sigmoid(logit)
            grad = (p - yi).astype(np.float32)           # (b,)

            # Gradients
            gw0 = grad.mean()
            gw  = np.zeros_like(w)
            gv  = np.zeros_like(v)
            for j in range(d):
                gj = grad
                np.add.at(gw[j], xi[:, j], gj)
                for l in range(j + 1, d):
                    vl = v[l, xi[:, l]]       # (b, K)
                    vj = v[j, xi[:, j]]
                    gv[j, xi[:, j]] += gj[:, None] * vl
                    gv[l, xi[:, l]] += gj[:, None] * vj
            gw /= b
            gv /= b
            gw += L2 * w
            gv += L2 * v

            # Adam update
            t += 1
            mw0 = beta1 * mw0 + (1 - beta1) * gw0
            vw0 = beta2 * vw0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (mw0 / (1 - beta1 ** t)) / (math.sqrt(vw0 / (1 - beta2 ** t)) + eps)

            mw = beta1 * mw + (1 - beta1) * gw
            vw = beta2 * vw + (1 - beta2) * gw * gw
            w  -= LR * (mw / (1 - beta1 ** t)) / (np.sqrt(vw / (1 - beta2 ** t)) + eps)

            mv = beta1 * mv + (1 - beta1) * gv
            vv = beta2 * vv + (1 - beta2) * gv * gv
            v  -= LR * (mv / (1 - beta1 ** t)) / (np.sqrt(vv / (1 - beta2 ** t)) + eps)

    # Pack weights: magic, d, k, C, w0, w[d][256], v[d][256][k]
    out = struct.pack("<4i", 0x464D3231, d, K, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree-2, rank-k) inference.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (one per field, value 0-255). Output: 1 byte class (0/1). */
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
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || K > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expect = 16 + 4 + 4L*d*256 + 4L*d*256*K;
    if (nw != expect) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;
    const float *w = p; p += (long)d * 256;
    const float *v = p;                    // [d][256][K]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *vj = malloc(sizeof(float) * K);
    float *vl = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int j = 0; j < d; j++) linear += w[(long)j * 256 + xi[j]];

        float inter = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *vj_ptr = v + ((long)j * 256 + xi[j]) * K;
            for (int l = j + 1; l < d; l++) {
                const float *vl_ptr = v + ((long)l * 256 + xi[l]) * K;
                float dot = 0.0f;
                for (int kk = 0; kk < K; kk++) dot += vj_ptr[kk] * vl_ptr[kk];
                inter += dot;
            }
        }
        float logit = linear + inter;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```