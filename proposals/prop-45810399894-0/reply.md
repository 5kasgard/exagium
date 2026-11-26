```hypothesis
Factorization Machine (label-fm) with k=8 embeddings captures pairwise feature interactions crucial for Adult
(e.g., education×occupation, marital-status×relationship) while staying lightweight. It shares embeddings
across classes with class-specific linear terms, yielding O(d·k) inference. Expect >0.855 accuracy at <1000 Ir/item.
Failure: accuracy ≤0.8524 (label-linear) or cost ≥2000 Ir/item.
```
```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Learns per-feature-value embeddings (k=8) + linear weights per class.
Adam lr=1e-2, L2=1e-4, batch=256, epochs=min(100, ceil(600000/n)). Exports: magic 0x464D4C31, d, k, C;
float32 V[d][256][k], W[d][256][C], b[C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 256, 600000, 100


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
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)   # embeddings
    W = np.zeros((d, 256, C), np.float32)                         # linear weights
    b = np.zeros(C, np.float32)                                   # bias
    params = [V, W, b]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                     # (B, d)
            yb = y[idx]                     # (B,)
            B = xb.shape[0]

            # Gather embeddings and linear weights
            Vb = V[xb]                      # (B, d, K)
            Wb = W[xb]                      # (B, d, C)

            # Interaction term
            Sv = Vb.sum(axis=1)             # (B, K)
            Qv = (Vb * Vb).sum(axis=1)      # (B, K)
            interaction = 0.5 * ((Sv * Sv - Qv).sum(axis=1))  # (B,)

            # Linear term
            linear = b + Wb.sum(axis=1)     # (B, C)

            # Scores and softmax
            scores = linear + interaction[:, None]  # (B, C)
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (B, C)

            # Gradient on scores
            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B  # (B, C)

            # Gradients
            grad_b = grad_scores.sum(axis=0)                     # (C,)
            grad_inter = grad_scores.sum(axis=1)                 # (B,)

            # Grad W: (d, 256, C)
            grad_W = np.zeros_like(W)
            for j in range(d):
                np.add.at(grad_W[j], xb[:, j], grad_scores)

            # Grad V: (d, 256, K)
            grad_V = np.zeros_like(V)
            # For each feature j: grad_V[j, xb[:,j], :] += grad_inter[:,None] * (Sv - Vb[:,j,:])
            for j in range(d):
                diff = Sv - Vb[:, j, :]               # (B, K)
                update = grad_inter[:, None] * diff   # (B, K)
                np.add.at(grad_V[j], xb[:, j], update)

            # L2 regularization
            grad_V += L2 * V
            grad_W += L2 * W

            grads = [grad_V, grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, K, C, then V, W, b (all float32 little-endian)
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    body = V.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || C != 2 || K <= 0 || K > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * K + (long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *V = (const float *)(hd + 4);
    const float *W = V + (long)d * 256 * K;
    const float *b = W + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);
    float *linear = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        for (int c = 0; c < C; c++) linear[c] = b[c];

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Vjv = V + ((long)j * 256 + v) * K;
            const float *Wjv = W + ((long)j * 256 + v) * C;
            for (int f = 0; f < K; f++) {
                float val = Vjv[f];
                sum_v[f] += val;
                sum_v2[f] += val * val;
            }
            for (int c = 0; c < C; c++) linear[c] += Wjv[c];
        }

        float interaction = 0.0f;
        for (int f = 0; f < K; f++) interaction += sum_v[f] * sum_v[f] - sum_v2[f];
        interaction *= 0.5f;

        unsigned char best = 0;
        float best_score = linear[0] + interaction;
        for (int c = 1; c < C; c++) {
            float sc = linear[c] + interaction;
            if (sc > best_score) { best_score = sc; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```