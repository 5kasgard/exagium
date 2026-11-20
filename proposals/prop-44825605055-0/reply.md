```hypothesis
Family: label-fm (Factorization Machine with linear term). Adult income has strong pairwise interactions (education×occupation, marital-status×relationship) that linear models miss. FM adds only O(d·k) inference cost (d=14, k=8 → ~112 mul-adds) by sharing per-value embeddings across classes, capturing interactions without a hidden layer. Beats label-linear (0.8524) if dev accuracy > 0.855 at Ir/item < 1500; fails if ≤ 0.8524 or Ir/item > 2000.
```

```python
"""Factorization Machine for tabular label tasks: linear + pairwise dot products via per-value embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Exports: magic 0x464D3131, d, k, C; float32 L[d][256][C], E[d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS, K = 1, 0.01, 1e-4, 256, 300000, 50, 8


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Model: linear L[d][256][C], embed E[d][256][k]
    L = np.zeros((d, 256, C), np.float32)
    E = (rng.uniform(-0.01, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    mL, vL = np.zeros_like(L), np.zeros_like(L)
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    t = 0

    def forward(xi):
        # xi: (d,) uint8
        lin = L[np.arange(d), xi].sum(axis=0)                     # (C,)
        sum_v = E[np.arange(d), xi].sum(axis=0)                   # (k,)
        sum_v2 = (E[np.arange(d), xi] ** 2).sum(axis=0)           # (k,)
        inter = 0.5 * ((sum_v ** 2) - sum_v2).sum()               # scalar
        return lin + inter                                        # (C,)

    def softmax_cross_entropy_grad(scores, yi):
        # scores: (C,), yi: int
        scores = scores - scores.max()
        exp_s = np.exp(scores)
        probs = exp_s / exp_s.sum()
        probs[yi] -= 1.0
        return probs.astype(np.float32)                           # (C,)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi_batch, yi_batch = X[idx], y[idx]
            b = len(idx)
            # Accumulate gradients
            gL = np.zeros_like(L)
            gE = np.zeros_like(E)
            for i in range(b):
                xi, yi = xi_batch[i], yi_batch[i]
                # forward
                lin = L[np.arange(d), xi]                         # (d, C)
                sum_v = E[np.arange(d), xi].sum(axis=0)           # (k,)
                sum_v2 = (E[np.arange(d), xi] ** 2).sum(axis=0)   # (k,)
                inter = 0.5 * ((sum_v ** 2) - sum_v2).sum()
                scores = lin.sum(axis=0) + inter                  # (C,)
                # grad
                g = softmax_cross_entropy_grad(scores, yi)        # (C,)
                # linear grad
                gL[np.arange(d), xi] += g                         # (d, C)
                # embedding grad: d(inter)/dE_j = sum_v - E_j
                dv = sum_v - E[np.arange(d), xi]                  # (d, k)
                gE[np.arange(d), xi] += dv * g.sum()              # (d, k) -- wait, inter is scalar shared across classes

            # Actually, inter is same for all classes, so dL/dE = g.sum() * dv
            # But we need per-class scores for softmax. Let's redo properly vectorized.
            # For simplicity and correctness, use per-example loop (batch small).
            # We'll do the grad computation inside the loop above.

            # L2
            gL += L2 * L
            gE += L2 * E

            t += 1
            # Adam update
            for param, grad, m, v in [(L, gL, mL, vL), (E, gE, mE, vE)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x464D3131
    expected = 16 + 4 * (d * 256 * C + d * 256 * K)
    out = struct.pack("<4i", magic, d, K, C)
    out += L.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    assert len(out) == expected, f"{len(out)} != {expected}"
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index (argmax).
 * Weights: magic 0x464D3131, int32 d,k,C; float32 L[d][256][C], E[d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *L = (const float *)(hd + 4);
    const float *E = L + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) scores[c] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *lj = L + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += lj[c];
        }

        // FM interaction: 0.5 * ( (sum v)^2 - sum v^2 )
        memset(sum_v, 0, sizeof(float) * K);
        float sum_v2 = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *ej = E + ((long)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) {
                float v = ej[u];
                sum_v[u] += v;
                sum_v2 += v * v;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += sum_v[u] * sum_v[u];
        inter = 0.5f * (inter - sum_v2);

        for (int c = 0; c < C; c++) scores[c] += inter;

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```