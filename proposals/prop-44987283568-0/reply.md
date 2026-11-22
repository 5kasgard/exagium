```hypothesis
Family: label-fm (Factorization Machine). Adult benefits from pairwise feature interactions (e.g., education × occupation, capital-gain × hours). A degree-2 FM with small latent dimension (k=4) captures these interactions with O(d·k) parameters and O(d·k) inference cost, staying near label-linear speed while improving accuracy. If dev accuracy ≤ label-linear (0.8524) or Ir/item > 2× label-linear, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Each of d=14 bytes is a categorical feature (256 values). Model: bias + linear[d,256] + latent[d,256,k].
Binary cross-entropy with Adam (lr=0.01, L2=1e-4, batch=256, epochs=20). Weights: magic 0x464D31, d, k, C; float32 bias, linear[d*256], latent[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 4
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 20

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
    assert C == 2
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    bias = np.float32(0.0)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    latent = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    mb, vb = 0.0, 0.0
    ml, vl = np.zeros_like(linear), np.zeros_like(linear)
    mlat, vlat = np.zeros_like(latent), np.zeros_like(latent)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Linear term: (B,)
            lin = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (B,)

            # Interaction term: 0.5 * (sum_v^2 - sum_v2)
            v = latent[np.arange(d), xb.T]  # (d, B, K)
            sum_v = np.sum(v, axis=0)       # (B, K)
            sum_v2 = np.sum(v * v, axis=0)  # (B, K)
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)  # (B,)

            logit = lin + inter
            # BCE gradient: sigmoid(logit) - y
            sig = 1.0 / (1.0 + np.exp(-logit))
            grad = sig - yb.astype(np.float32)  # (B,)

            # Gradients
            g_bias = np.mean(grad)
            # linear gradient: for each feature j, value v, average grad over batch where xb[:,j]==v
            g_linear = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                for val in range(256):
                    mask = (vals == val)
                    if np.any(mask):
                        g_linear[j, val] = np.mean(grad[mask])
            # latent gradient: for each feature j, value v, grad * (sum_v - v_jv)
            g_latent = np.zeros_like(latent)
            for j in range(d):
                vals = xb[:, j]
                for val in range(256):
                    mask = (vals == val)
                    if np.any(mask):
                        g_latent[j, val] = np.mean(grad[mask, np.newaxis] * (sum_v[mask] - latent[j, val]), axis=0)

            # L2 regularization
            g_linear += L2 * linear
            g_latent += L2 * latent

            t += 1
            # Adam updates
            mb = beta1 * mb + (1 - beta1) * g_bias
            vb = beta2 * vb + (1 - beta2) * g_bias * g_bias
            bias -= LR * (mb / (1 - beta1 ** t)) / (np.sqrt(vb / (1 - beta2 ** t)) + eps)

            ml = beta1 * ml + (1 - beta1) * g_linear
            vl = beta2 * vl + (1 - beta2) * g_linear * g_linear
            linear -= LR * (ml / (1 - beta1 ** t)) / (np.sqrt(vl / (1 - beta2 ** t)) + eps)

            mlat = beta1 * mlat + (1 - beta1) * g_latent
            vlat = beta2 * vlat + (1 - beta2) * g_latent * g_latent
            latent -= LR * (mlat / (1 - beta1 ** t)) / (np.sqrt(vlat / (1 - beta2 ** t)) + eps)

    # Write weights
    out = struct.pack("<4i", 0x464D31, d, K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype(np.float32).tobytes()
    out += latent.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (0/1). Weights: magic 0x464D31, d, k, C; float32 bias, linear[d*256], latent[d*256*k]. */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    size_t expected = 16 + (1 + (size_t)d * 256 + (size_t)d * 256 * K) * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *latent = linear + (size_t)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(K * sizeof(float));
    float *sum_v2 = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = bias;
        for (int j = 0; j < K; j++) { sum_v[j] = 0.0f; sum_v2[j] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin += linear[(size_t)j * 256 + v];
            const float *lv = latent + ((size_t)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = lv[kk];
                sum_v[kk] += val;
                sum_v2[kk] += val * val;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += sum_v[kk] * sum_v[kk] - sum_v2[kk];
        inter *= 0.5f;

        float logit = lin + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```