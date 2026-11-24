```hypothesis
Family: factorization machine (FM) with per-field/value embeddings. Captures pairwise feature interactions via low-rank factors (K=8), crucial for tabular data with categorical fields (education×occupation, marital×relationship, etc.). Beats linear on Adult while keeping O(d·K) inference. Fails if dev accuracy ≤ 0.8524 (label-linear).
```
```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 50


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
    n, d = X.shape
    assert d == card["input"]["bytes"]

    rng = np.random.default_rng(SEED)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    factor = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_fac, v_fac = np.zeros_like(factor), np.zeros_like(factor)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(idx)

            # Forward
            lin_part = np.sum(linear[np.arange(d), xb.T], axis=0)
            fm_part = np.zeros(B, np.float32)
            for f in range(K):
                v = factor[np.arange(d), xb.T, f]
                sum_v = np.sum(v, axis=0)
                sum_v2 = np.sum(v * v, axis=0)
                fm_part += 0.5 * (sum_v * sum_v - sum_v2)
            logits = lin_part + fm_part

            # Logistic gradient
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad = probs - yb

            # Gradients
            grad_lin = np.zeros_like(linear)
            grad_fac = np.zeros_like(factor)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(grad_lin[j], vals, grad)
                for f in range(K):
                    v_all = factor[np.arange(d), xb.T, f]
                    sum_v = np.sum(v_all, axis=0)
                    v_j = v_all[j]
                    g = grad * (sum_v - v_j)
                    np.add.at(grad_fac[j, :, f], vals, g)

            grad_lin += L2 * linear
            grad_fac += L2 * factor

            t += 1
            m_lin = beta1 * m_lin + (1 - beta1) * grad_lin
            v_lin = beta2 * v_lin + (1 - beta2) * grad_lin * grad_lin
            linear -= LR * (m_lin / (1 - beta1**t)) / (np.sqrt(v_lin / (1 - beta2**t)) + eps)

            m_fac = beta1 * m_fac + (1 - beta1) * grad_fac
            v_fac = beta2 * v_fac + (1 - beta2) * grad_fac * grad_fac
            factor -= LR * (m_fac / (1 - beta1**t)) / (np.sqrt(v_fac / (1 - beta2**t)) + eps)

    # Save: magic, d, K, C; linear[d][256]; factor[d][256][K]
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<4i", magic, d, K, 2)
    out += linear.astype("<f4").tobytes()
    out += factor.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || K > 64 || d < 1 || d > 100) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *factor = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = 0.0f;
        for (int j = 0; j < d; j++)
            logit += linear[(long)j * 256 + xi[j]];

        for (int f = 0; f < K; f++) {
            float sum_v = 0.0f, sum_v2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = factor[((long)j * 256 + xi[j]) * K + f];
                sum_v += v;
                sum_v2 += v * v;
            }
            logit += 0.5f * (sum_v * sum_v - sum_v2);
        }

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```