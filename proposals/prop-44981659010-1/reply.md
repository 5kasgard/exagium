```hypothesis
We propose "label-poly2", a polynomial logistic regression of degree 2 on z-scored pixel features. It adds all 2016 pairwise interactions between the 64 ordinal inputs, capturing local stroke correlations in 8x8 digits. With ~21k features, it should significantly beat the linear baseline (0.9613) while keeping inference cost moderate (~20k multiply-adds/item). If dev accuracy does not exceed 0.966 (0.5% above linear), the family does not justify its cost.
```

```python
"""Polynomial degree-2 logistic regression for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # "POL2" little-endian


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
    # read data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # z-score per feature (ordinal)
    X = X_bytes.astype(np.float64)
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mean) / sd  # (n, d)

    # build polynomial features: linear + pairwise products
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    num_pairs = len(pairs)
    K = d + num_pairs
    Phi = np.empty((n, K), dtype=np.float32)
    Phi[:, :d] = Z.astype(np.float32)
    for idx, (i, j) in enumerate(pairs):
        Phi[:, d + idx] = Z[:, i] * Z[:, j]

    # initialize weights
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # training loop (Adam)
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            phi, yy = Phi[idx], y[idx]
            # forward
            logits = phi @ W + b  # (batch, C)
            # softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)
            # gradients
            grad_W = phi.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grads = [grad_W, grad_b]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # split weights for storage
    W_linear = W[:d]
    W_inter = W[d:]

    # write weights.bin
    header = struct.pack("<4i", MAGIC, d, C, num_pairs)
    body = (
        mean.astype("<f4").tobytes() +
        sd.astype("<f4").tobytes() +
        W_linear.astype("<f4").tobytes() +
        W_inter.astype("<f4").tobytes() +
        b.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Polynomial degree-2 logistic regression inference.
 * ./infer weights.bin in.rec N
 * Each input record: 64 bytes (pixels). Output: one byte class. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAGIC 0x504F4C32

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
    unsigned char *w = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)w;
    if (nw < 16 || hd[0] != MAGIC) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], num_pairs = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (2L * d + (long)C * (d + num_pairs + 1));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ptr = (float *)(hd + 4);
    float *mean = ptr; ptr += d;
    float *sd = ptr; ptr += d;
    float *W_linear = ptr; ptr += (long)d * C;
    float *W_inter = ptr; ptr += (long)num_pairs * C;
    float *bias = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *acc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mean[j]) / sd[j];
        }
        for (int c = 0; c < C; c++) acc[c] = bias[c];

        // linear part
        for (int j = 0; j < d; j++) {
            float zj = z[j];
            for (int c = 0; c < C; c++) acc[c] += zj * W_linear[(long)j * C + c];
        }
        // interaction part
        long pair_idx = 0;
        for (int i1 = 0; i1 < d - 1; i1++) {
            for (int i2 = i1 + 1; i2 < d; i2++) {
                float prod = z[i1] * z[i2];
                for (int c = 0; c < C; c++) acc[c] += prod * W_inter[pair_idx * C + c];
                pair_idx++;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```