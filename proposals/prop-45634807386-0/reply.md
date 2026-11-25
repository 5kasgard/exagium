```hypothesis
Family: label-fm (Factorization Machine with rank-4 pairwise interactions).
Why: Adult has strong feature interactions (education×occupation, marital×relationship) that linear models miss. FM captures all pairwise interactions via low-rank factors with O(d·K) inference cost. At K=4, params ≈72KB, inference ~500 float ops/item — faster than label-mlp16 (2325 Ir/item) while modeling interactions.
Failure signal: dev accuracy ≤ label-linear (0.8524) or Ir/item ≥ label-mlp16 (2325).
```
```python
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 128, 600000, 100


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    linear = (rng.normal(0, 0.01, (d, 256))).astype(np.float32)
    factors = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    bias = np.float32(0.0)

    params = [linear, factors, np.array([bias], dtype=np.float32)]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    y01 = y  # {0,1}
    rows = np.arange(d)[:, None]

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y01[idx]
            B = len(idx)

            # forward
            lin_part = np.sum(linear[rows, xb.T], axis=0) + bias
            f = factors[rows, xb.T, :]
            sum_v = np.sum(f, axis=0)
            sum_v2 = np.sum(f * f, axis=0)
            fm_part = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
            score = lin_part + fm_part

            # logistic gradient
            score_clipped = np.clip(score, -20.0, 20.0)
            sig = 1.0 / (1.0 + np.exp(-score_clipped))
            grad = (sig - yb).astype(np.float32)

            # gradients
            d_linear = np.zeros_like(linear)
            d_factors = np.zeros_like(factors)
            d_bias = np.float32(grad.sum())

            for j in range(d):
                vals = xb[:, j]
                np.add.at(d_linear[j], vals, grad)
                fv = factors[j, vals, :]
                grad_f = grad[:, None] * (sum_v - fv)
                np.add.at(d_factors[j], vals, grad_f)

            d_linear += L2 * linear
            d_factors += L2 * factors

            grads = [d_linear, d_factors, np.array([d_bias], dtype=np.float32)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x464D3130
    out = struct.pack("<4i", magic, d, K, 1)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += factors.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 1 || d != 14 || K < 1 || K > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * d * 256 * (1L + K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *factors = linear + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias[0];
        for (int j = 0; j < d; j++)
            score += linear[j * 256 + xi[j]];

        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *fv = factors + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float v = fv[f];
                sum_v[f] += v;
                sum_v2[f] += v * v;
            }
        }
        for (int f = 0; f < K; f++)
            score += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```