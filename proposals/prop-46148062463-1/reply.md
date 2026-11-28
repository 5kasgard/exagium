```hypothesis
Family: label-linear-onehot (linear model on per-pixel one-hot encoding).
Why: Optical digits have only 17 distinct pixel values (0..16). One-hot encoding each pixel gives a sparse
     1088-d feature space where a linear model can learn arbitrary per-pixel non-linearities (e.g. "ink vs paper")
     without a hidden layer. Inference stays extremely cheap: 64 table look-ups and 10 adds per class.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear), per-pixel non-linearities are insufficient and
                pixel interactions (requiring a hidden layer or trees) are necessary.
```

```python
"""label-linear-onehot: linear model on one-hot encoded pixels (0..16).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x4F4E4548 (ONEH), d=64, V=17, C=10;
float32 T[d][V][C] (row-major), float32 bias[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
V = 17  # pixel values 0..16


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
    assert d == 64 and C == 10

    # One-hot encoding: each pixel -> 17-d unit vector
    n = X.shape[0]
    Z = np.zeros((n, d * V), dtype=np.float32)
    for j in range(d):
        Z[np.arange(n), j * V + X[:, j]] = 1.0

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (d * V, C)) * math.sqrt(6.0 / (d * V))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            g = softmax_grad(z @ W + b, yy)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Reshape W to (d, V, C) for the table
    T = W.reshape(d, V, C).astype("<f4")
    bias = b.astype("<f4")
    out = struct.pack("<4i", 0x4F4E4548, d, V, C) + T.tobytes() + bias.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-linear-onehot infer: ./infer weights.bin in.rec N
 * Input: 64 bytes per record (pixel counts 0..16). Output: 1 byte class per record. */
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
    if (nw < 16 || hd[0] != 0x4F4E4548) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], V = hd[2], C = hd[3];
    if (d != 64 || V != 17 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L + 4L * d * V * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *bias = T + (long)d * V * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[10];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v >= V) v = V - 1;              // clamp just in case
            const float *row = T + ((long)j * V + v) * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```