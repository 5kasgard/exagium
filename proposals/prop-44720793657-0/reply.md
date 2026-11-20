```hypothesis
Family: label-fm (factorization machine). It models pairwise feature interactions via low-rank embeddings, which are crucial for tabular data like Adult (e.g., education × occupation). The model uses class-specific embeddings so interactions affect the decision boundary. If dev accuracy does not exceed label-linear by ≥0.5% absolute, the added parameters and compute are not justified.
```

```python
"""Factorization machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Exports: int32 magic 0x464D4C31, d, k, C; float32 bias[C], linear[C][d][256], emb[C][d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 50
K = 8  # embedding dimension


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
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.zeros(C, np.float32)
    linear = np.zeros((C, d, 256), np.float32)
    emb = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)

    # Adam state
    params = [bias, linear, emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            scores = np.zeros((B, C), np.float32)
            sum_emb = np.zeros((B, C, K), np.float32)
            sum_emb_sq = np.zeros((B, C, K), np.float32)

            for c in range(C):
                scores[:, c] = bias[c]
                for j in range(d):
                    v = xb[:, j]
                    scores[:, c] += linear[c, j, v]
                    e = emb[c, j, v, :]           # (B, K)
                    sum_emb[:, c, :] += e
                    sum_emb_sq[:, c, :] += e * e
                inter = 0.5 * np.sum(sum_emb[:, c, :]**2 - sum_emb_sq[:, c, :], axis=1)
                scores[:, c] += inter

            # Softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            # Loss gradient w.r.t scores
            grad_scores = probs.copy()
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B

            # Gradients
            grad_bias = np.zeros(C, np.float32)
            grad_linear = np.zeros((C, d, 256), np.float32)
            grad_emb = np.zeros((C, d, 256, K), np.float32)

            for c in range(C):
                g = grad_scores[:, c]               # (B,)
                grad_bias[c] = g.sum()
                # linear gradient
                for j in range(d):
                    v = xb[:, j]
                    np.add.at(grad_linear[c, j], v, g)
                # embedding gradient
                # dL/demb = g * (sum_emb - e)
                for j in range(d):
                    v = xb[:, j]
                    e = emb[c, j, v, :]             # (B, K)
                    # sum_emb for this class is sum_emb[:, c, :] (B, K)
                    factor = g[:, None] * (sum_emb[:, c, :] - e)   # (B, K)
                    np.add.at(grad_emb[c, j], v, factor)

            # L2 regularization
            grad_linear += L2 * linear
            grad_emb += L2 * emb

            grads = [grad_bias, grad_linear, grad_emb]

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
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)C * d * 256 + (long)C * d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_emb = malloc(sizeof(float) * C * K);
    float *sum_emb_sq = malloc(sizeof(float) * C * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            scores[c] = bias[c];
            for (int f = 0; f < K; f++) {
                sum_emb[c * K + f] = 0.0f;
                sum_emb_sq[c * K + f] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                scores[c] += linear[((long)c * d + j) * 256 + v];
                const float *e = emb + ((long)c * d * 256 * K + (long)j * 256 * K + v * K);
                for (int f = 0; f < K; f++) {
                    float val = e[f];
                    sum_emb[c * K + f] += val;
                    sum_emb_sq[c * K + f] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float s = sum_emb[c * K + f];
                inter += 0.5f * (s * s - sum_emb_sq[c * K + f]);
            }
            scores[c] += inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```