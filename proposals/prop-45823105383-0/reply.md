```hypothesis
Factorization machines (FM) capture pairwise feature interactions via low-rank latent vectors, which are known to be effective on tabular data like Adult (e.g., education × occupation). This model adds a latent dimension k=4 per class, increasing parameters by ~28k floats over label-linear but keeping inference at O(d·k) operations. It should exceed label-linear's 0.8524 accuracy (target >0.855) while staying under ~1000 Ir/item. Failure: dev accuracy ≤ 0.8524 or cost > 1500 Ir/item.
```

```python
"""Factorization Machine for tabular label tasks: linear + pairwise interactions via low-rank latent vectors.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exported weights: magic 0x464D4C31, d, C, k; float32 linear[d][256][C], bias[C], latent[d][256][C][k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    rng = np.random.default_rng(SEED)

    # Parameters
    linear = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    latent = (rng.normal(0, 0.01, (d, 256, C, K))).astype(np.float32)

    params = [linear, bias, latent]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            lin_out = bias.copy() + np.sum(linear[np.arange(d)[:, None], xb.T, :], axis=0)  # (B, C)
            lat_sum = np.zeros((B, C, K), np.float32)
            lat_sum_sq = np.zeros((B, C, K), np.float32)
            for j in range(d):
                v = latent[j, xb[:, j], :, :]            # (B, C, K)
                lat_sum += v
                lat_sum_sq += v * v
            inter = 0.5 * np.sum(lat_sum * lat_sum - lat_sum_sq, axis=2)  # (B, C)
            logits = lin_out + inter

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # Backward
            grad_lin = grad_logits                           # (B, C)
            grad_inter = grad_logits                         # (B, C)
            grad_bias = grad_logits.sum(axis=0)              # (C,)

            # Gradient for linear
            grad_linear = np.zeros_like(linear)
            for j in range(d):
                np.add.at(grad_linear[j], xb[:, j], grad_lin)

            # Gradient for latent
            grad_latent = np.zeros_like(latent)
            # dL/dlatent = dL/dinter * dinter/dlatent
            # dinter/dlatent = lat_sum - latent   (per feature)
            for j in range(d):
                v = latent[j, xb[:, j], :, :]            # (B, C, K)
                # dL/dv = grad_inter[:, :, None] * (lat_sum - v)
                g = grad_inter[:, :, None] * (lat_sum - v)
                np.add.at(grad_latent[j], xb[:, j], g)

            # L2 regularization
            grad_linear += L2 * linear
            grad_latent += L2 * latent

            grads = [grad_linear, grad_bias, grad_latent]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Pack weights
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, C, K)
    out += linear.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (d from weights). Output: one byte class index (argmax logit). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t linear_sz = (size_t)d * 256 * C;
    size_t bias_sz = C;
    size_t latent_sz = (size_t)d * 256 * C * K;
    size_t total_floats = linear_sz + bias_sz + latent_sz;
    if (nw != 16 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *bias = linear + linear_sz;
    const float *latent = bias + bias_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lat_sum = calloc(C * K, sizeof(float));
    float *lat_sum_sq = calloc(C * K, sizeof(float));
    float logits[256];  // C <= 256

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(lat_sum, 0, C * K * sizeof(float));
        memset(lat_sum_sq, 0, C * K * sizeof(float));
        for (int c = 0; c < C; c++) logits[c] = bias[c];

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_ptr = linear + ((size_t)j * 256 + v) * C;
            const float *lat_ptr = latent + ((size_t)j * 256 * C + v * C) * K;
            for (int c = 0; c < C; c++) {
                logits[c] += lin_ptr[c];
                for (int l = 0; l < K; l++) {
                    float w = lat_ptr[(size_t)c * K + l];
                    lat_sum[c * K + l] += w;
                    lat_sum_sq[c * K + l] += w * w;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int l = 0; l < K; l++) {
                float s = lat_sum[c * K + l];
                inter += s * s - lat_sum_sq[c * K + l];
            }
            logits[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```