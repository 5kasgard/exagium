```hypothesis
Family: label-fm (Factorization Machine). Adult income has strong pairwise feature interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. A field-aware FM with latent dim k=8 adds only ~28K parameters (vs ~3.5K for linear) and captures all pairwise interactions in O(d·k) inference time. It should exceed label-linear's 0.8524 dev accuracy while staying near its 719 Ir/item cost. If dev accuracy ≤0.855, the extra complexity is not justified.
```
```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Each of d fields has 256 values; learns linear weight + k-dim
embedding per value. Binary logistic loss with Adam. Exports: magic 0x464D3100, d, k; float32 w0, W[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 600000, 50


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    w0 = np.float32(0.0)
    W = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    params = [w0, W, V]
    # Adam state: m, v for each param
    state = [
        (np.zeros_like(w0), np.zeros_like(w0)),
        (np.zeros_like(W), np.zeros_like(W)),
        (np.zeros_like(V), np.zeros_like(V))
    ]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            # Linear term: w0 + sum_j W[j, xb[:,j]]
            linear = w0 + W[np.arange(d), xb.T].T.sum(axis=1)  # (B,)

            # Interaction term: 0.5 * sum_f ( (sum_j V[j,xj,f])^2 - sum_j V[j,xj,f]^2 )
            V_sel = V[np.arange(d)[:, None], xb.T, :]  # (d, B, K)
            V_sel = V_sel.transpose(1, 0, 2)           # (B, d, K)
            sum_f = V_sel.sum(axis=1)                  # (B, K)
            sum_sq_f = (V_sel ** 2).sum(axis=1)        # (B, K)
            interaction = 0.5 * ((sum_f ** 2 - sum_sq_f).sum(axis=1))  # (B,)

            score = linear + interaction
            p = 1.0 / (1.0 + np.exp(-score))
            g = (p - yb).astype(np.float32)  # (B,)

            # Gradients
            dw0 = g.mean()
            dW = np.zeros_like(W)
            for j in range(d):
                np.add.at(dW[j], xb[:, j], g)
            dW /= B
            dW += L2 * W

            dV = np.zeros_like(V)
            for f in range(K):
                sum_f_b = sum_f[:, f]  # (B,)
                for j in range(d):
                    v_j = V_sel[:, j, f]  # (B,)
                    grad = g * (sum_f_b - v_j)  # (B,)
                    np.add.at(dV[j, :, f], xb[:, j], grad)
            dV /= B
            dV += L2 * V

            # Adam update
            grads = [dw0, dW, dV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            w0, W, V = params[0], params[1], params[2]

    # Pack weights: magic, d, k; then w0, W[d][256], V[d][256][k] all float32 little-endian
    magic = 0x464D3100
    out = struct.pack("<3i", magic, d, K)
    out += np.array(w0, dtype="<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (field values 0-255). Output: one byte class (0 or 1).
 * Weights format: magic 0x464D3100, int32 d, int32 k; float32 w0, W[d][256], V[d][256][k]. */
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
    if (nw < 12 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2];
    if (d < 1 || d > 256 || K < 1 || K > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 12L + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 3);
    float w0 = p[0];
    const float *W = p + 1;
    const float *V = W + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_f = malloc(sizeof(float) * K);
    float *sum_sq_f = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int f = 0; f < K; f++) { sum_f[f] = 0.0f; sum_sq_f[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            linear += W[(long)j * 256 + v];
            const float *Vjv = V + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = Vjv[f];
                sum_f[f] += val;
                sum_sq_f[f] += val * val;
            }
        }

        float interaction = 0.0f;
        for (int f = 0; f < K; f++) {
            interaction += 0.5f * (sum_f[f] * sum_f[f] - sum_sq_f[f]);
        }
        float score = linear + interaction;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```