```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression).
Why: The 8x8 optdigits pixels have strong pairwise interactions (strokes, loops) that a linear model misses. Explicitly adding all 2016 pairwise products plus squares gives a rich 2144-d feature space where a linear classifier becomes a quadratic decision boundary. Training is still convex (multinomial logistic regression with L2), so it’s deterministic and fast. Inference is a single 2144×10 matrix multiply (~21k FLOPs), cheaper than label-mlp16’s two layers.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or CPU cost ≥ label-mlp16 (8685 Ir/item).
```

```python
"""label-poly2: multinomial logistic regression on degree-2 polynomial features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Expands each 64-byte sample (z-scored per pixel) to 2144 features:
  64 linear, 64 squares, 2016 unique pairwise products.
Trains with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Exports: magic 0x504F4C32, d=64, D=2144, C=10; float32 mean[64], std[64], W[2144][10], b[10]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # "POL2"
D = 64 + 64 + 64 * 63 // 2  # 2144


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def expand(Z):  # Z: (n, 64) float32 normalized
    n = Z.shape[0]
    out = np.empty((n, D), np.float32)
    out[:, :64] = Z
    out[:, 64:128] = Z * Z
    idx = 128
    for i in range(64):
        for j in range(i + 1, 64):
            out[:, idx] = Z[:, i] * Z[:, j]
            idx += 1
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n, d = X.shape
    assert d == 64

    rng = np.random.default_rng(SEED)

    # z-score per pixel on full training set
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    # expand to polynomial features
    Z2 = expand(Z)
    n, k = Z2.shape
    assert k == D

    # initialize weights
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z2[idx], y[idx]
            # softmax gradient
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            G = (P / len(yy)).astype(np.float32)
            # gradients with L2
            grads = [z.T @ G + L2 * W, G.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights: magic, d, D, C; mean[64], std[64], W[D*C], b[C] (all float32 LE)
    out = struct.pack("<4i", MAGIC, d, D, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels). Output record: 1 byte (class 0-9).
 * Weights format: int32 magic(0x504F4C32), d, D, C; float32 mean[d], std[d], W[D][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 64 || D != 2144 || C != 10) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * (2L * d + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *W    = std + d;
    const float *b    = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(D * sizeof(float));
    float *s = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // normalize and compute linear + square terms
        for (int j = 0; j < d; j++) {
            float v = (xi[j] - mean[j]) / std[j];
            z[j] = v;
            z[64 + j] = v * v;
        }
        // pairwise products
        int idx = 128;
        for (int i1 = 0; i1 < d; i1++) {
            for (int i2 = i1 + 1; i2 < d; i2++) {
                z[idx++] = z[i1] * z[i2];
            }
        }

        // linear classification
        for (int c = 0; c < C; c++) s[c] = b[c];
        for (int j = 0; j < D; j++) {
            float zv = z[j];
            const float *w = W + (long)j * C;
            for (int c = 0; c < C; c++) s[c] += zv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```