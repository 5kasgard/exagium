```hypothesis
Family: label-fm (factorization machine, order 2).
Why: The Adult dataset has strong pairwise interactions (e.g., education × occupation, marital-status × sex). A linear model misses these; a full MLP overfits or costs more. A rank-8 FM adds only 14×256×8 ≈ 29k parameters, captures all pairwise feature interactions, and inference is just 91 dot-products of length 8 per item — far cheaper than an MLP.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 1500 Ir/item, the FM is not justified.
```

```python
"""Factorization Machine (order 2) for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D3231, d, r, C; float32 w[d][256], e[d][256][r] (row-major)."""
import json, math, struct, sys

import numpy as np

R = 8                 # embedding rank
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 10


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)   # 0 / 1
    d = X.shape[1]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    # parameters: w[d, 256], e[d, 256, R]
    w = np.zeros((d, 256), dtype=np.float32)
    e = (rng.normal(0, 0.01, (d, 256, R))).astype(np.float32)

    # Adam state
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    me, ve = np.zeros_like(e), np.zeros_like(e)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n = X.shape[0]
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (B, d)
            yb = y[idx]                 # (B,)
            B = len(idx)

            # gather embeddings
            w_b = w[:, xb.T].T          # (B, d)
            e_b = e[:, xb.T, :].transpose(1, 0, 2)   # (B, d, R)

            # logit = sum_j w_j + sum_{j<k} <e_j, e_k>
            sum_e = e_b.sum(axis=1)                 # (B, R)
            sum_e2 = (e_b * e_b).sum(axis=1)        # (B, R)
            pairwise = 0.5 * ((sum_e * sum_e - sum_e2).sum(axis=1))   # (B,)
            logit = w_b.sum(axis=1) + pairwise

            # gradient of binary cross-entropy: p - y, where p = sigmoid(logit)
            p = 1.0 / (1.0 + np.exp(-logit))
            dlogit = (p - yb).astype(np.float32)    # (B,)

            # gradients
            # w: for each example, add dlogit to the used w_j
            gw = np.zeros_like(w)
            np.add.at(gw.T, xb.T, dlogit)           # (d, 256)  (transpose trick)
            gw += L2 * w

            # e: grad_e_j = dlogit * (sum_e - e_j)
            ge = np.zeros_like(e)
            for j in range(d):
                vals = xb[:, j]                     # (B,)
                diff = (dlogit[:, None] * (sum_e - e_b[:, j, :])).astype(np.float32)  # (B, R)
                np.add.at(ge[j], vals, diff)
            ge += L2 * e

            # Adam update
            t += 1
            for param, grad, m, v in [(w, gw, mw, vw), (e, ge, me, ve)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights.bin
    magic = 0x464D3231
    out = struct.pack("<4i", magic, d, R, C)
    out += w.astype("<f4").tobytes()
    out += e.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (order 2) inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C != 2 || d < 1 || R < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * ( (long)d * 256 * (1 + R) );
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w = (const float *)(hd + 4);
    const float *e = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *vecs = malloc(sizeof(float) * d * R);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += w[(long)j * 256 + v];
            const float *src = e + ((long)j * 256 + v) * R;
            float *dst = vecs + (long)j * R;
            memcpy(dst, src, sizeof(float) * R);
        }
        for (int j = 0; j < d; j++) {
            const float *ej = vecs + (long)j * R;
            for (int k2 = j + 1; k2 < d; k2++) {
                const float *ek = vecs + (long)k2 * R;
                for (int r = 0; r < R; r++) logit += ej[r] * ek[r];
            }
        }
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```