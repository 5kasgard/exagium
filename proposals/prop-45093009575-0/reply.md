```hypothesis
Family: label-fm (Factorization Machine). 
Why: FM models pairwise feature interactions via low-rank latent vectors, which are known to be effective on tabular data like Adult (e.g., education × occupation). With K=8 latent factors per class, it adds modest capacity over linear (O(d·K·C) ≈ 200 ops/item) while staying far cheaper than GBDT/MLP. 
Failure: If dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ 719 (label-linear), the hypothesis is disproven.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: int32 magic 0x464D4C31, d, K, C; float32 L[d][256][C], V[d][256][K][C], bias[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, no tuning on the tuning split for simplicity)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 8          # latent dimension

MAGIC = 0x464D4C31  # 'FML1'

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
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters: L (linear), V (latent), bias
    L = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K, C))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    params = [L, V, bias]
    # Adam state
    state = [(np.zeros_like(p, dtype=np.float32), np.zeros_like(p, dtype=np.float32)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]          # (B, d)
            yy = y[idx]          # (B,)
            B = len(idx)

            # Forward pass: compute scores per class
            # linear part: (B, C)
            linear = bias + np.sum(L[np.arange(d), xi.T, :], axis=0).T   # (B, C)

            # interaction part: for each class and factor
            # S_fc = sum_i V[i, xi, f, c]  -> (B, K, C)
            # We can compute by looping over features (d=14 small)
            S = np.zeros((B, K, C), dtype=np.float32)
            Q = np.zeros((B, K, C), dtype=np.float32)
            for i in range(d):
                v = V[i, xi[:, i], :, :]   # (B, K, C)
                S += v
                Q += v * v
            interaction = 0.5 * np.sum(S * S - Q, axis=1)   # (B, C)

            scores = linear + interaction   # (B, C)

            # Softmax and loss gradient
            max_s = np.max(scores, axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grads_s = probs / B   # (B, C)

            # Gradients
            # bias
            g_bias = np.sum(grads_s, axis=0).astype(np.float32)   # (C,)

            # L: (d, 256, C)
            g_L = np.zeros_like(L, dtype=np.float32)
            for i in range(d):
                # xi[:, i] is (B,) indices 0..255
                # grads_s is (B, C)
                np.add.at(g_L[i], xi[:, i], grads_s)

            # V: (d, 256, K, C)
            g_V = np.zeros_like(V, dtype=np.float32)
            for i in range(d):
                v = V[i, xi[:, i], :, :]   # (B, K, C)
                s = S[:, :, :]             # (B, K, C)
                # grad for V: grads_s[:, None, :] * (s - v)
                # grads_s: (B, C) -> (B, 1, C)
                g_v = grads_s[:, None, :] * (s - v)   # (B, K, C)
                np.add.at(g_V[i], xi[:, i], g_v)

            # L2 regularization
            g_L += L2 * L
            g_V += L2 * V

            grads = [g_L, g_V, g_bias]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += L.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (argmax). */
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
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *L = (const float *)(hd + 4);
    const float *V = L + (long)d * 256 * C;
    const float *bias = V + (long)d * 256 * K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *lj = L + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += lj[c];
        }

        // Interaction part
        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float sum_v = 0.0f, sum_v2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    float v = V[((long)j * 256 + xi[j]) * K * C + f * C + c];
                    sum_v += v;
                    sum_v2 += v * v;
                }
                inter += 0.5f * (sum_v * sum_v - sum_v2);
            }
            scores[c] += inter;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```