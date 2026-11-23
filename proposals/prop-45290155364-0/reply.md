```hypothesis
Family: label-fm (factorization machine with latent dim 2). Adult has strong feature interactions (e.g., education × occupation, marital-status × relationship) that a linear model misses. A tiny FM adds only O(d·k) per-class parameters and inference cost ~2× linear. If dev accuracy does not exceed label-linear (0.8524), pairwise interactions are not the bottleneck.
```

```python
"""Factorization machine (FM) for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: each of the 14 input bytes is used directly as an index (0-255) into
per-class linear weights and per-class latent vectors (dim k=2).
Weights layout (little-endian):
  magic(0x464D4C31), d, k, C
  bias[C] (float32)
  for each class c: linear[d][256] (float32), latent[d][256][k] (float32)"""
import json, math, struct, sys

import numpy as np

K = 2
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
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters per class: linear [d,256], latent [d,256,K], bias [C]
    linear = np.zeros((C, d, 256), dtype=np.float32)
    latent = (rng.uniform(-1, 1, (C, d, 256, K)) * math.sqrt(6.0 / (256 * K))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Adam state
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_lat, v_lat = np.zeros_like(latent), np.zeros_like(latent)
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: compute logits for each class
            logits = np.zeros((B, C), dtype=np.float32)
            for c in range(C):
                # linear part
                lin = bias[c] + np.sum(linear[c, np.arange(d), xb.T], axis=0)  # (B,)
                # FM interaction part
                # sum_latent = sum_j latent[c, j, xb[:,j], :]  -> (B, K)
                sum_lat = np.zeros((B, K), dtype=np.float32)
                sum_sq = np.zeros((B, K), dtype=np.float32)
                for j in range(d):
                    lat = latent[c, j, xb[:, j], :]  # (B, K)
                    sum_lat += lat
                    sum_sq += lat * lat
                inter = 0.5 * np.sum(sum_lat * sum_lat - sum_sq, axis=1)  # (B,)
                logits[:, c] = lin + inter

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B  # (B, C)

            # Gradients
            grad_linear = np.zeros_like(linear)
            grad_latent = np.zeros_like(latent)
            grad_bias = np.zeros_like(bias)

            for c in range(C):
                g = grad_logits[:, c]  # (B,)
                grad_bias[c] = g.sum()
                # linear gradient
                for j in range(d):
                    np.add.at(grad_linear[c, j], xb[:, j], g)
                # latent gradient
                # For each feature j, we need sum_{i!=j} latent[c, i, xb[:,i], :]
                # We have sum_lat (B, K). For feature j, other_sum = sum_lat - latent[c, j, xb[:,j], :]
                for j in range(d):
                    lat_j = latent[c, j, xb[:, j], :]  # (B, K)
                    other = sum_lat - lat_j
                    grad_lat_j = g[:, None] * other    # (B, K)
                    np.add.at(grad_latent[c, j], xb[:, j], grad_lat_j)

            # L2 regularization
            grad_linear += L2 * linear
            grad_latent += L2 * latent

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * grad_linear
            v_lin = beta2 * v_lin + (1 - beta2) * grad_linear * grad_linear
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # latent
            m_lat = beta1 * m_lat + (1 - beta1) * grad_latent
            v_lat = beta2 * v_lat + (1 - beta2) * grad_latent * grad_latent
            m_hat = m_lat / (1 - beta1 ** t)
            v_hat = v_lat / (1 - beta2 ** t)
            latent -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1 - beta2) * grad_bias * grad_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    expected_floats = C + C * d * 256 * (1 + K)
    out = struct.pack("<4i", magic, d, K, C)
    out += bias.astype("<f4").tobytes()
    for c in range(C):
        out += linear[c].astype("<f4").tobytes()
        out += latent[c].astype("<f4").tobytes()
    if len(out) != 16 + 4 * expected_floats:
        raise RuntimeError("weights size mismatch")
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=14). Output is one byte: argmax class.
 * Weights format: magic(0x464D4C31), d, k, C; bias[C]; for each class: linear[d][256], latent[d][256][k]. */
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
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * (C + (long)C * d * 256 * (1 + K));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *latent = linear + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_lat = malloc(sizeof(float) * K);
    float *sum_sq = malloc(sizeof(float) * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            const float *lin_c = linear + (long)c * d * 256;
            const float *lat_c = latent + (long)c * d * 256 * K;

            float lin = bias[c];
            for (int j = 0; j < d; j++) lin += lin_c[(long)j * 256 + xi[j]];

            for (int f = 0; f < K; f++) sum_lat[f] = 0.0f, sum_sq[f] = 0.0f;
            for (int j = 0; j < d; j++) {
                const float *lat_j = lat_c + ((long)j * 256 + xi[j]) * K;
                for (int f = 0; f < K; f++) {
                    float v = lat_j[f];
                    sum_lat[f] += v;
                    sum_sq[f] += v * v;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < K; f++) inter += 0.5f * (sum_lat[f] * sum_lat[f] - sum_sq[f]);
            logits[c] = lin + inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```