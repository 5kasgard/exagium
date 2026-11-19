```hypothesis
Family: label-fm (Factorization Machine). Captures pairwise feature interactions via low-rank embeddings (k=4), which should improve accuracy over linear on tabular data like Adult. Inference cost is O(d*k) ~ 14*4 ops per class, comparable to linear. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds label-linear significantly, the approach fails.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Model: per-column (14) per-value (256) weights and embeddings (k=4) for each class (2).
Score_c = sum_j w_{j,x_j,c} + 0.5*(||sum_j e_{j,x_j,c}||^2 - sum_j ||e_{j,x_j,c}||^2).
Trained with Adam, L2=1e-4, lr=0.01, batch=64, epochs=30.
Weights format: int32 magic 0x464D5F31, d, k, C; float32 weights[d*256*C], embeddings[d*256*C*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 30
K = 4  # embedding rank

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 14
    C = card["classes"]         # 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # weights: (d, 256, C)
    weights = np.zeros((d, 256, C), dtype=np.float32)
    # embeddings: (d, 256, C, K)
    embeddings = (rng.normal(0, 0.01, (d, 256, C, K))).astype(np.float32)

    # Adam state
    m_w, v_w = np.zeros_like(weights), np.zeros_like(weights)
    m_e, v_e = np.zeros_like(embeddings), np.zeros_like(embeddings)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x = X[idx]          # (B, d)
            yb = y[idx]         # (B,)
            B = x.shape[0]

            # Forward
            scores = np.zeros((B, C), dtype=np.float32)
            sum_emb = np.zeros((B, C, K), dtype=np.float32)
            sum_emb_sq = np.zeros((B, C), dtype=np.float32)
            for j in range(d):
                v = x[:, j]                 # (B,)
                scores += weights[j, v, :]  # (B, C)
                emb = embeddings[j, v, :, :]  # (B, C, K)
                sum_emb += emb
                sum_emb_sq += np.sum(emb * emb, axis=2)
            interaction = 0.5 * (np.sum(sum_emb * sum_emb, axis=2) - sum_emb_sq)
            scores += interaction

            # Softmax loss gradient
            max_s = np.max(scores, axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dscores = probs.astype(np.float32) / B  # (B, C)

            # Backward
            dweights = np.zeros_like(weights)
            dembeddings = np.zeros_like(embeddings)
            for j in range(d):
                v = x[:, j]
                np.add.at(dweights[j], v, dscores)
                emb = embeddings[j, v, :, :]          # (B, C, K)
                diff = sum_emb - emb                  # (B, C, K)
                grad_emb = dscores[:, :, None] * diff # (B, C, K)
                np.add.at(dembeddings[j], v, grad_emb)

            # L2 regularization
            dweights += L2 * weights
            dembeddings += L2 * embeddings

            # Adam update
            t += 1
            for param, grad, m, v in [(weights, dweights, m_w, v_w), (embeddings, dembeddings, m_e, v_e)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    magic = 0x464D5F31  # 'FM_1'
    header = struct.pack("<4i", magic, d, K, C)
    w_bytes = weights.astype("<f4").tobytes()
    e_bytes = embeddings.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + w_bytes + e_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes (14); output is one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K != 4 || C != 2) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + (long)d * 256 * C * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *weights = (const float *)(hd + 4);
    const float *embeddings = weights + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float scores[2] = {0.0f, 0.0f};
        float sum_emb[2][4] = {{0}};
        float sum_emb_sq[2] = {0.0f, 0.0f};

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            long base_w = ((long)j * 256 + v) * C;
            scores[0] += weights[base_w];
            scores[1] += weights[base_w + 1];

            long base_e = ((long)j * 256 + v) * C * K;
            for (int c = 0; c < C; c++) {
                const float *emb = embeddings + base_e + c * K;
                for (int dim = 0; dim < K; dim++) {
                    float e = emb[dim];
                    sum_emb[c][dim] += e;
                    sum_emb_sq[c] += e * e;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float sq_sum = 0.0f;
            for (int dim = 0; dim < K; dim++) sq_sum += sum_emb[c][dim] * sum_emb[c][dim];
            scores[c] += 0.5f * (sq_sum - sum_emb_sq[c]);
        }

        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```