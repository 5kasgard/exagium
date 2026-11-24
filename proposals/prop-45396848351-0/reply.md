```hypothesis
Family: label-fm (Factorization Machine, degree 2). The Adult task has 14 mixed-type features where pairwise interactions (e.g., education × occupation, marital-status × relationship) are known to be predictive. A degree-2 FM captures all pairwise interactions via low-rank embeddings (K=4) with only ~36K parameters, adding ~700 FLOPs/item over the linear baseline. It should exceed label-linear's 85.24% dev accuracy while staying well under label-mlp16's 2325 Ir/item cost. If dev accuracy ≤ 85.3% or Ir/item > 1500, the model fails its design goals.
```
```python
"""Factorization Machine (degree 2) for label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports per-class linear weights and per-class per-feature-value embeddings.
Format: magic 0x464D3231, d, K, C; float32 W[C][d][256], V[C][d][256][K], bias[C]."""
import json, math, struct, sys

import numpy as np

K = 4  # embedding dimension
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    d = card["input"]["bytes"]
    assert C == 2 and d == 14
    rng = np.random.default_rng(SEED)

    # W: [C, d, 256], V: [C, d, 256, K], bias: [C]
    W = (rng.normal(0, 0.01, (C, d, 256))).astype(np.float32)
    V = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [W, V, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            B = len(yy)

            scores = np.zeros((B, C), np.float32)
            v_sum = np.zeros((C, B, K), np.float32)
            v_sq_sum = np.zeros((C, B, K), np.float32)
            v_ji = np.zeros((C, d, B, K), np.float32)

            for c in range(C):
                linear = np.zeros(B, np.float32)
                for j in range(d):
                    w = W[c, j, x[:, j]]
                    linear += w
                    v = V[c, j, x[:, j]]
                    v_ji[c, j] = v
                    v_sum[c] += v
                    v_sq_sum[c] += v * v
                interaction = 0.5 * np.sum(v_sum[c]**2 - v_sq_sum[c], axis=1)
                scores[:, c] = linear + interaction + bias[c]

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            P = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            G = P.copy()
            G[np.arange(B), yy] -= 1.0
            G /= B

            grads = [np.zeros_like(W), np.zeros_like(V), np.zeros_like(bias)]

            for c in range(C):
                Gc = G[:, c]
                grads[2][c] = Gc.sum()
                for j in range(d):
                    grads[0][c, j] += np.bincount(x[:, j], weights=Gc, minlength=256).astype(np.float32)
                    diff = v_sum[c] - v_ji[c, j]
                    for f in range(K):
                        weights = Gc * diff[:, f]
                        grads[1][c, j, :, f] += np.bincount(x[:, j], weights=weights, minlength=256).astype(np.float32)

            grads[0] += L2 * W
            grads[1] += L2 * V

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    out = struct.pack("<4i", 0x464D3231, d, K, C)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree 2) for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 4 || nw != 16 + 4L * (2L * d * 256 + 2L * d * 256 * K + 2)) {
        fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 4);
    const float *V = W + 2L * d * 256;
    const float *bias = V + 2L * d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2];
    float v_sum[2][4];
    float v_sq_sum[2][4];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            scores[c] = bias[c];
            for (int f = 0; f < K; f++) {
                v_sum[c][f] = 0.0f;
                v_sq_sum[c][f] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                scores[c] += W[(long)c * d * 256 + j * 256 + v];
                const float *Vptr = V + ((long)c * d * 256 + j * 256 + v) * K;
                for (int f = 0; f < K; f++) {
                    float val = Vptr[f];
                    v_sum[c][f] += val;
                    v_sq_sum[c][f] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                inter += 0.5f * (v_sum[c][f] * v_sum[c][f] - v_sq_sum[c][f]);
            }
            scores[c] += inter;
        }

        unsigned char best = 0;
        if (scores[1] > scores[0]) best = 1;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```