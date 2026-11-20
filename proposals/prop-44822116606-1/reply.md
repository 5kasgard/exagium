```hypothesis
We propose label-poly2: polynomial logistic regression with all pairwise pixel products (including squares) as features.
On optdigits, pixel interactions capture stroke patterns that a linear model misses; degree-2 expansion yields 2144 features
from 64 pixels. Training uses the same deterministic Adam as label-linear. We expect dev accuracy > 0.9613 (label-linear)
with ~20k multiply-adds/item. If accuracy does not exceed label-linear, the hypothesis fails.
```
```python
"""label-poly2: polynomial logistic regression (degree 2) for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Expands each input byte vector x (d bytes) into all pairs (i<=j): x[i]*x[j], giving D=d(d+1)/2 features.
Standardizes features (zero mean, unit variance) on training set, then trains softmax regression with Adam.
Exports: magic 0x504F4C32, d, C, D; float32 mean[D], std[D], W[D*C], b[C] (all little-endian)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # "POL2"


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # generate all pairs (i<=j) in fixed order
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    D = len(pairs)

    # build expanded feature matrix (float32)
    Z = np.empty((n, D), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        Z[:, k] = X[:, i].astype(np.float32) * X[:, j].astype(np.float32)

    # standardize per feature
    mean = Z.mean(axis=0)
    std = Z.std(axis=0)
    std[std == 0] = 1.0
    Z = (Z - mean) / std

    # initialize weights
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # training loop
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)
            # gradients
            gW = z.T @ grad + L2 * W
            gb = grad.sum(axis=0)
            grads = [gW, gb]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    out = struct.pack("<4i", MAGIC, d, C, D)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Expands each d-byte record into D=d(d+1)/2 pairwise products, standardizes,
 * computes scores = b + z * W, outputs argmax class (one byte per record). */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], D = hd[3];
    if (d <= 0 || C <= 0 || C > 256 || D != d * (d + 1) / 2) {
        fprintf(stderr, "bad header dimensions\n"); return 3;
    }
    long expected = 16L + 4L * (2L * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + D;
    const float *W = std + D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    int *pair_i = malloc(sizeof(int) * D);
    int *pair_j = malloc(sizeof(int) * D);

    /* generate pairs in same order as Python: for i in 0..d-1, for j in i..d-1 */
    int k = 0;
    for (int i = 0; i < d; i++) {
        for (int j = i; j < d; j++) {
            pair_i[k] = i;
            pair_j[k] = j;
            k++;
        }
    }

    for (long rec_idx = 0; rec_idx < N; rec_idx++) {
        uint32_t klen; const unsigned char *xi = next_rec(x, nx, &pos, &klen);
        if ((int)klen != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", rec_idx, klen, d); return 3; }

        /* compute expanded features */
        for (int f = 0; f < D; f++) {
            float val = (float)xi[pair_i[f]] * (float)xi[pair_j[f]];
            z[f] = (val - mean[f]) / std[f];
        }

        /* compute scores */
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < D; f++) {
            float zv = z[f];
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += zv * wf[c];
        }

        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```