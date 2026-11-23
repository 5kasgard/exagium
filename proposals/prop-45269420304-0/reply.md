```hypothesis
Family: label-fm (Factorization Machine with rank-8 embeddings on raw 14×256 features).
Why: Adult has strong pairwise interactions (e.g., education×occupation, marital-status×sex) that linear models miss. A rank-8 FM adds only 14×256×8 ≈ 29k interaction parameters, trains in seconds with Adam, and infers in ~14×8 FLOPs per item. It should beat label-linear (0.8524) while staying faster than label-mlp16.
Failure signal: dev accuracy ≤ 0.8524 or inference cost > label-mlp16.
```

```python
"""Factorization Machine (order-2) for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encodes each of the 14 columns as 256-value categorical (ordinal already quantile-binned).
Learns linear weights w[j][v] and embeddings E[j][v][k] (k=8).
Adam lr=0.01, L2=1e-4, batch=64, epochs=30.
Exports: magic 0x464D3132, d, k, C; float32 w[d*256], E[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 30


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
    assert C == 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # parameters: w[d,256], E[d,256,K]
    w = np.zeros((d, 256), np.float32)
    E = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward: linear + interaction
            lin = np.sum(w[np.arange(d), xb.T], axis=0)  # [bsz]
            # interaction: 0.5 * (||sum_emb||^2 - sum||emb||^2)
            emb = E[np.arange(d), xb.T]  # [d, bsz, K]
            sum_emb = emb.sum(axis=0)    # [bsz, K]
            inter = 0.5 * ((sum_emb * sum_emb).sum(axis=1) - (emb * emb).sum(axis=(0, 2)))
            logit = lin + inter
            # sigmoid + BCE gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = p - yb  # [bsz]

            # gradients for w and E
            # w gradient: for each sample, each feature j, value v = xb[i,j], add grad[i]
            gw = np.zeros_like(w)
            np.add.at(gw, (np.arange(d)[:, None], xb.T), grad)  # [d,256]

            # E gradient: for each sample i, feature j, value v, embedding emb[j,v]
            # derivative of inter wrt emb[j,v] = grad[i] * (sum_emb[i] - emb[j,v])
            gE = np.zeros_like(E)
            # loop over features for vectorization
            for j in range(d):
                v = xb[:, j]  # [bsz]
                # sum_emb[i] - emb[j, v[i]]
                diff = sum_emb - emb[j, np.arange(bsz), v]  # [bsz, K]
                # grad[i] * diff[i]
                g = grad[:, None] * diff  # [bsz, K]
                np.add.at(gE[j], v, g)

            # L2 regularization
            gw += L2 * w
            gE += L2 * E

            # Adam update
            t += 1
            for param, grad, m, v in [(w, gw, mw, vw), (E, gE, mE, vE)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights
    magic = 0x464D3132  # "FM12"
    out = struct.pack("<4i", magic, d, K, C)
    out += w.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=14); output one byte class (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t w_sz = (size_t)d * 256;
    size_t E_sz = (size_t)d * 256 * K;
    if (nw != 16 + 4L * (w_sz + E_sz)) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *w = (const float *)(hd + 4);
    const float *E = w + w_sz;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = 0.0f;
        float sum_emb[8] = {0};
        float sum_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *emb = E + ((size_t)j * 256 + v) * K;
            lin += w[j * 256 + v];
            for (int u = 0; u < K; u++) {
                float e = emb[u];
                sum_emb[u] += e;
                sum_sq += e * e;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += sum_emb[u] * sum_emb[u];
        inter = 0.5f * (inter - sum_sq);
        float logit = lin + inter;
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char pred = prob > 0.5f ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```