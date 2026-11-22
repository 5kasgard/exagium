```hypothesis
Factorization Machine (FM) with rank 4 on the 14 raw features. Adult income depends on feature interactions (e.g., education × occupation); FM captures all pairwise interactions in O(d·k) time and space, adding only ~140 KiB to the model. It should surpass label-linear (0.8524) and approach label-gbdt (0.8664) while staying an order of magnitude faster than GBDT. Failure: dev accuracy ≤ 0.8524.
```
```python
"""Factorization Machine family for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D3130, d, k, C; float32 linear[d][256][C], latent[d][256][k][C], bias[C]."""
import json, math, struct, sys

import numpy as np

K = 4                 # FM rank
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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # Parameters
    linear = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    latent = (rng.normal(0, 0.01, (d, 256, K, C))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [linear, latent, bias]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    n = X.shape[0]
    steps_per_epoch = max(1, n // BATCH)
    max_steps = min(MAX_EPOCHS * steps_per_epoch, math.ceil(SAMPLES / BATCH))

    t = 0
    for step in range(max_steps):
        idx = rng.choice(n, BATCH, replace=False)
        xb, yb = X[idx], y[idx]

        # Forward
        lin_part = np.zeros((BATCH, C), np.float32)
        S = np.zeros((BATCH, K, C), np.float32)
        Q = np.zeros((BATCH, K, C), np.float32)
        for j in range(d):
            ids = xb[:, j]
            lin_part += linear[j, ids]                     # (B, C)
            lat = latent[j, ids]                           # (B, K, C)
            S += lat
            Q += lat * lat

        scores = bias + lin_part + 0.5 * np.sum(S * S - Q, axis=1)   # (B, C)

        # Softmax + cross-entropy gradient
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        probs[np.arange(BATCH), yb] -= 1.0
        dscores = probs / BATCH                              # (B, C)

        # Gradients
        dbias = dscores.sum(axis=0)
        dlinear = np.zeros_like(linear)
        dlatent = np.zeros_like(latent)

        for j in range(d):
            ids = xb[:, j]
            # linear gradient
            np.add.at(dlinear[j], ids, dscores)
            # latent gradient
            lat = latent[j, ids]                             # (B, K, C)
            # dlatent = dscores[:, None, :] * (S[:, :, :] - lat)
            # S includes current lat, so we need S - lat for each feature
            # Recompute S without current feature? Instead use the identity:
            # d/dv_{j} 0.5*(S^2 - Q) = S - v_{j}
            # where S is sum over all features. We have S already (with all features).
            # So for feature j, the term is (S - lat).
            grad_lat = dscores[:, None, :] * (S - lat)       # (B, K, C)
            np.add.at(dlatent[j], ids, grad_lat)

        # L2 regularization
        dlinear += L2 * linear
        dlatent += L2 * latent
        dbias   += L2 * bias

        grads = [dlinear, dlatent, dbias]
        t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            m[i] *= 0.9; m[i] += 0.1 * gr
            v[i] *= 0.999; v[i] += 0.001 * gr * gr
            m_hat = m[i] / (1 - 0.9 ** t)
            v_hat = v[i] / (1 - 0.999 ** t)
            p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D3130  # "FM10"
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += linear.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (C < 1 || C > 256 || K < 1 || K > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }

    size_t linear_sz = (size_t)d * 256 * C;
    size_t latent_sz = (size_t)d * 256 * K * C;
    size_t bias_sz = C;
    size_t total_floats = linear_sz + latent_sz + bias_sz;
    if (nw != 16 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *latent = linear + linear_sz;
    const float *bias   = latent + latent_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float scores[256];  // C <= 256
        for (int c = 0; c < C; c++) scores[c] = bias[c];

        float S[16][256] = {{0}};   // K x C, K<=16, C<=256
        float Q[16][256] = {{0}};

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin = linear + ((size_t)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += lin[c];

            const float *lat = latent + (((size_t)j * 256 + v) * K + 0) * C;
            for (int f = 0; f < K; f++) {
                for (int c = 0; c < C; c++) {
                    float val = lat[f * C + c];
                    S[f][c] += val;
                    Q[f][c] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float sf = S[f][c];
                inter += 0.5f * (sf * sf - Q[f][c]);
            }
            scores[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```