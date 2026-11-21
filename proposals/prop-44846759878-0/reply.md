```hypothesis
Factorization machines (label-fm) capture pairwise feature interactions via low-rank latent vectors, which are crucial for tabular datasets like Adult where feature combinations (e.g., education × occupation) strongly predict income. With a latent dimension k=8, the model adds only ~115k parameters over linear, keeping inference fast (O(d·k) per item). If dev accuracy does not exceed label-linear (0.8524) or inference cost rises above ~1500 Ir/item, the family fails its goal.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib + numpy.
Exported weights: magic 0x464D4C31, d, k, C, w0, linear[d*256], latent[d*256*k] (all float32)."""
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
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Parameters
    w0 = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    latent = (rng.uniform(-1, 1, (d, 256, K)) * math.sqrt(6.0 / (256 * K))).astype(np.float32)

    params = [w0, linear, latent]
    # Adam state: for each param, (m, v)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = xb.shape[0]

            # Forward
            # linear part: w0 + sum_j linear[j, xb[:, j]]
            lin_part = w0 + np.sum(linear[np.arange(d), xb.T], axis=0)   # (b,)

            # latent part: sum_v and sum_v2
            # gather latent vectors for each feature: shape (b, d, K)
            lat = latent[np.arange(d), xb.T].transpose(1, 0, 2)   # (b, d, K)
            sum_v = np.sum(lat, axis=1)            # (b, K)
            sum_v2 = np.sum(lat * lat, axis=1)     # (b, K)
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)  # (b,)

            score = lin_part + inter                # (b,)
            prob = 1.0 / (1.0 + np.exp(-score))     # (b,)
            # gradient of BCE w.r.t score
            e = prob - yb                           # (b,)

            # Backward
            # w0
            grad_w0 = np.float32(np.sum(e))
            # linear
            grad_linear = np.zeros_like(linear)
            for j in range(d):
                np.add.at(grad_linear[j], xb[:, j], e)
            grad_linear += L2 * linear

            # latent
            grad_latent = np.zeros_like(latent)
            # For each example, gradient for latent[j, xb[i,j]] is (sum_v[i] - lat[i,j]) * e[i]
            # Vectorized accumulation:
            for j in range(d):
                v_sel = lat[:, j, :]                     # (b, K)
                diff = (sum_v - v_sel) * e[:, None]      # (b, K)
                # accumulate into grad_latent[j, xb[:, j], :]
                for i in range(b):
                    grad_latent[j, xb[i, j]] += diff[i]
            grad_latent += L2 * latent

            grads = [grad_w0, grad_linear, grad_latent]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # header: magic, d, K, C
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0 or 1).
 * Weights format: magic 0x464D4C31, int32 d, int32 K, int32 C,
 * float32 w0, float32 linear[d*256], float32 latent[d*256*K]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *linear = p + 1;
    const float *latent = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int j = 0; j < d; j++) score += linear[(long)j * 256 + xi[j]];

        memset(sum_v, 0, sizeof(float) * K);
        memset(sum_v2, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *v = latent + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = v[kk];
                sum_v[kk] += val;
                sum_v2[kk] += val * val;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += sum_v[kk] * sum_v[kk] - sum_v2[kk];
        score += 0.5f * inter;

        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char cls = prob > 0.5f ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```