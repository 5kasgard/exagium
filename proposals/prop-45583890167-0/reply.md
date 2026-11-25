```hypothesis
The label-fm family uses a factorization machine on the raw 14 byte features (each treated as a 256-category field) with embedding dimension 4. It captures pairwise feature interactions with O(d*k) inference cost, which should improve accuracy over the linear model (label-linear) while remaining faster than label-mlp16. If dev accuracy does not exceed 0.855 (label-linear is 0.8524) or inference cost exceeds 1500 Ir/item, the approach is not worthwhile.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: seed 1, single thread. Uses Adam, L2, minibatch 64, epochs min(100, ceil(600000/n)).
Exports: magic 0x464D4C31, d, k, C; float32 bias[C], W[C][d*256], V[C][d*256][k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


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
    C = card["classes"]
    d = X.shape[1]
    n_cat = 256
    rng = np.random.default_rng(SEED)

    # Flattened feature index: feature i uses categories [i*256, (i+1)*256)
    feat_off = np.arange(d, dtype=np.int32) * n_cat

    # Parameters
    bias = np.zeros(C, dtype=np.float32)
    W = rng.normal(0, 0.01, size=(C, d * n_cat)).astype(np.float32)
    V = rng.normal(0, 0.01, size=(C, d * n_cat, K)).astype(np.float32)

    params = [bias, W, V]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                     # (B, d) uint8
            yb = y[idx]                     # (B,)
            B = len(idx)

            flat_x = xb + feat_off          # (B, d)
            flat_x_flat = flat_x.ravel()    # (B*d,)

            # Forward
            scores = np.zeros((B, C), dtype=np.float32)
            sum_v_all = np.zeros((B, C, K), dtype=np.float32)
            for c in range(C):
                w = W[c, flat_x]            # (B, d)
                v = V[c, flat_x]            # (B, d, K)
                sum_w = w.sum(axis=1)
                sum_v = v.sum(axis=1)
                sum_v2 = (v * v).sum(axis=1)
                interaction = 0.5 * ((sum_v * sum_v - sum_v2).sum(axis=1))
                scores[:, c] = bias[c] + sum_w + interaction
                sum_v_all[:, c] = sum_v

            # Softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            # Loss gradient
            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B

            # Backward: accumulate gradients
            grad_bias = np.zeros_like(bias)
            grad_W = np.zeros_like(W)
            grad_V = np.zeros_like(V)

            for c in range(C):
                g = grad_scores[:, c]                   # (B,)
                grad_bias[c] = g.sum()

                # W gradient
                g_rep = np.repeat(g, d)                 # (B*d,)
                np.add.at(grad_W[c], flat_x_flat, g_rep)

                # V gradient
                v = V[c, flat_x]                        # (B, d, K)
                sum_v = sum_v_all[:, c]                 # (B, K)
                grad_v = g[:, None, None] * (sum_v[:, None, :] - v)  # (B, d, K)
                np.add.at(grad_V[c], flat_x_flat[:, None], grad_v.reshape(-1, K))

            # L2 regularization
            grad_W += L2 * W
            grad_V += L2 * V

            # Adam update
            t += 1
            grads = [grad_bias, grad_W, grad_V]
            for i, (param, (m, v)) in enumerate(zip(params, states)):
                gr = grads[i]
                m *= BETA1; m += (1 - BETA1) * gr
                v *= BETA2; v += (1 - BETA2) * gr * gr
                m_hat = m / (1 - BETA1 ** t)
                v_hat = v / (1 - BETA2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Export
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += bias.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class.
 * Weights format: magic 0x464D4C31, int32 d, K, C; float32 bias[C], W[C][d*256], V[C][d*256][K]. */
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
    if (C < 1 || C > 256 || K < 1 || d < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)C * d * 256 + (long)C * d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *W = bias + C;
    const float *V = W + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = (float *)malloc(sizeof(float) * C * K);
    float *sum_v2 = (float *)malloc(sizeof(float) * C * K);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            scores[c] = bias[c];
            for (int f = 0; f < K; f++) {
                sum_v[c * K + f] = 0.0f;
                sum_v2[c * K + f] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            long base = (long)j * 256 + v;
            for (int c = 0; c < C; c++) {
                scores[c] += W[c * d * 256 + base];
                const float *Vc = V + c * d * 256 * K + base * K;
                for (int f = 0; f < K; f++) {
                    float val = Vc[f];
                    sum_v[c * K + f] += val;
                    sum_v2[c * K + f] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float sv = sum_v[c * K + f];
                inter += 0.5f * (sv * sv - sum_v2[c * K + f]);
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