```hypothesis
We propose a factorization machine with class-specific embeddings (label-fm). The Adult dataset has strong feature interactions (e.g., education × occupation) that linear models miss. FM adds a low-rank (K=8) pairwise interaction term per class, keeping parameters modest (~65k floats). Training uses Adam on cross-entropy with the same hyperparameters as label-linear. If dev accuracy does not exceed label-linear's 0.8524 by at least 0.005, the hypothesis fails.
```
```python
"""Factorization Machine with class-specific embeddings for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: int32 magic 0x464D5F31, d, K, C; float32 bias[C], L[d][256][C], E[C][d][256][K]."""
import json, math, struct, sys

import numpy as np

K = 8
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
    bias = np.zeros(C, np.float32)
    L = np.zeros((d, 256, C), np.float32)
    E = rng.normal(0, 0.01, (C, d, 256, K)).astype(np.float32)

    params = [bias, L, E]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward pass
            logits = np.zeros((B, C), np.float32)
            logits += bias

            # Linear part
            for j in range(d):
                v = xb[:, j]
                logits += L[j, v, :]

            # Interaction part per class
            for c in range(C):
                sum_E = np.zeros((B, K), np.float32)
                sum_E_sq = np.zeros((B, K), np.float32)
                for j in range(d):
                    v = xb[:, j]
                    e = E[c, j, v, :]          # (B, K)
                    sum_E += e
                    sum_E_sq += e * e
                inter = 0.5 * np.sum(sum_E * sum_E - sum_E_sq, axis=1)
                logits[:, c] += inter

            # Softmax and loss gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)

            # Gradients
            grad_bias = grad_logits.sum(axis=0)
            grad_L = np.zeros_like(L)
            grad_E = np.zeros_like(E)

            # Linear gradients
            for j in range(d):
                v = xb[:, j]
                for c in range(C):
                    np.add.at(grad_L[j, :, c], v, grad_logits[:, c])

            # Embedding gradients
            for c in range(C):
                sum_E = np.zeros((B, K), np.float32)
                for j in range(d):
                    v = xb[:, j]
                    e = E[c, j, v, :]
                    sum_E += e
                for j in range(d):
                    v = xb[:, j]
                    e = E[c, j, v, :]
                    # d(inter)/dE = sum_E - e
                    g = grad_logits[:, c][:, None] * (sum_E - e)
                    np.add.at(grad_E[c, j, :, :], v, g)

            # L2 regularization
            grad_L += L2 * L
            grad_E += L2 * E

            grads = [grad_bias, grad_L, grad_E]

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
    magic = 0x464D5F31  # "FM_1"
    out = struct.pack("<4i", magic, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += L.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class.
 * Weights layout: int32 magic(0x464D5F31), d, K, C; float32 bias[C], L[d][256][C], E[C][d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * C + 4L * d * 256 * C + 4L * C * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *L = bias + C;
    const float *E = L + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_E = malloc(sizeof(float) * K);
    float *sum_E_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // Linear part
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *lj = L + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += lj[c];
        }

        // Interaction part per class
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                sum_E[kk] = 0.0f;
                sum_E_sq[kk] = 0.0f;
            }
            for (int j = 0; j < d; j++) {
                int v = xi[j];
                const float *e = E + ((long)c * d * 256 * K + (long)j * 256 * K + v * K);
                for (int kk = 0; kk < K; kk++) {
                    float val = e[kk];
                    sum_E[kk] += val;
                    sum_E_sq[kk] += val * val;
                }
            }
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) inter += sum_E[kk] * sum_E[kk] - sum_E_sq[kk];
            scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```