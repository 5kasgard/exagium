```hypothesis
Factorization Machine (FM) with latent dimension k=8. Adult income has strong feature interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. FM models all pairwise interactions via low-rank dot products, adding only O(d·k) params and compute. With k=8, it should exceed label-mlp16 (0.8591) while staying fast. If dev accuracy ≤0.8524 (label-linear), interactions aren't captured or k is too small.
```
```python
"""Factorization Machine for tabular label tasks: fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread. FM with k latent factors per feature value.
Linear term + pairwise interactions via dot product. SGD with momentum, L2 reg."""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 8
LR = 0.01
MOMENTUM = 0.9
L2_W = 1e-4
L2_V = 1e-3
BATCH = 256
EPOCHS = 30
MAGIC = 0x464D3130  # "FM10"


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
    assert d == 14
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    # params
    w0 = 0.0
    w = np.zeros((d, 256), dtype=np.float32)
    v = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    # velocities
    w0_vel = 0.0
    w_vel = np.zeros_like(w)
    v_vel = np.zeros_like(v)

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # forward
            w_x = np.zeros(B, dtype=np.float32)
            v_sum = np.zeros((B, K), dtype=np.float32)
            v_sq_sum = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                vals = xb[:, j]
                w_x += w[j, vals]
                v_j = v[j, vals, :]
                v_sum += v_j
                v_sq_sum += v_j * v_j
            interaction = 0.5 * np.sum(v_sum * v_sum - v_sq_sum, axis=1)
            score = w0 + w_x + interaction
            p = 1.0 / (1.0 + np.exp(-score))
            g = p - yb  # (B,)

            # gradients
            grad_w0 = g.mean()
            grad_w = np.zeros_like(w)
            grad_v = np.zeros_like(v)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(grad_w[j], vals, g)
                diff = v_sum - v[j, vals, :]
                weighted_diff = g[:, None] * diff
                np.add.at(grad_v[j], vals, weighted_diff)

            # momentum + weight decay update
            w0_vel = MOMENTUM * w0_vel - LR * (grad_w0 + L2_W * w0)
            w0 += w0_vel

            w_vel = MOMENTUM * w_vel - LR * (grad_w + L2_W * w)
            w += w_vel

            v_vel = MOMENTUM * v_vel - LR * (grad_v + L2_V * v)
            v += v_vel

    # pack weights: magic, d, K, C; w0; w[d*256]; v[d*256*K]
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += np.array([w0], dtype=np.float32).astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (0 or 1).
 * Weights: magic 0x464D3130, int32 d, K, C; float32 w0, w[d*256], v[d*256*K] */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int j = 0; j < d; j++) score += w[j * 256 + xi[j]];

        for (int f = 0; f < K; f++) {
            float sum_v = 0.0f, sum_v2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float vj = v[((long)j * 256 + xi[j]) * K + f];
                sum_v += vj;
                sum_v2 += vj * vj;
            }
            score += 0.5f * (sum_v * sum_v - sum_v2);
        }

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```