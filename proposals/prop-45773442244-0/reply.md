```hypothesis
Family: label-fm (Factorization Machine). Adult income depends on feature interactions (e.g., education×occupation, age×hours/week) which linear models miss. A low-rank FM captures all pairwise interactions with O(k·d) extra parameters (k=8 factors, d=14 features). It should beat label-linear (0.8524) significantly while adding only ~60k params and ~200 FLOPs/item. If dev accuracy ≤ 0.8524, the interaction hypothesis fails for this encoding.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights: magic 0x464D4C31, d, k, C; float32 w0[C], w[d][256][C], v[d][256][k][C]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 300000, 50


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
    rng = np.random.default_rng(SEED)

    w0 = np.zeros(C, np.float32)
    w = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K, C))).astype(np.float32)

    params = [w0, w, v]
    state = [ (np.zeros_like(p), np.zeros_like(p)) for p in params ]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)

            scores = np.tile(w0, (B, 1))
            sum_v = np.zeros((B, K, C), np.float32)
            sum_v2 = np.zeros((B, K, C), np.float32)

            for j in range(d):
                vals = xb[:, j]
                scores += w[j, vals]
                vj = v[j, vals]
                sum_v += vj
                sum_v2 += vj * vj

            scores += 0.5 * (sum_v * sum_v - sum_v2).sum(axis=1)

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dscores = (probs / B).astype(np.float32)

            dw0 = dscores.sum(axis=0)
            dw = np.zeros_like(w)
            dv = np.zeros_like(v)

            for j in range(d):
                vals = xb[:, j]
                np.add.at(dw[j], vals, dscores)
                vj = v[j, vals]
                for f in range(K):
                    for c in range(C):
                        dv[j, vals, f, c] += dscores[:, c] * (sum_v[:, f, c] - vj[:, f, c])

            grads = [dw0, dw + L2 * w, dv + L2 * v]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v_adam = state[i]
                m *= 0.9; m += 0.1 * gr
                v_adam *= 0.999; v_adam += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v_adam / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (argmax). */
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
    if (C < 1 || C > 256 || K < 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)d * 256 * K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *w = p; p += (long)d * 256 * C;
    const float *v = p;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *sum_v = malloc(sizeof(float) * K * C);
    float *sum_v2 = malloc(sizeof(float) * K * C);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, w0, C * sizeof(float));
        memset(sum_v, 0, K * C * sizeof(float));
        memset(sum_v2, 0, K * C * sizeof(float));
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            const float *wj = w + ((long)j * 256 + val) * C;
            for (int c = 0; c < C; c++) scores[c] += wj[c];
            const float *vj = v + ((long)j * 256 + val) * K * C;
            for (int f = 0; f < K; f++) {
                for (int c = 0; c < C; c++) {
                    float vjfc = vj[(long)f * C + c];
                    sum_v[(long)f * C + c] += vjfc;
                    sum_v2[(long)f * C + c] += vjfc * vjfc;
                }
            }
        }
        for (int f = 0; f < K; f++) {
            for (int c = 0; c < C; c++) {
                float sv = sum_v[(long)f * C + c];
                scores[c] += 0.5f * (sv * sv - sum_v2[(long)f * C + c]);
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```