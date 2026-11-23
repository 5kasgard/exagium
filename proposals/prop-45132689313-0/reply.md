```hypothesis
Family: label-fm (Factorization Machine on 14 raw byte fields, rank-4).
Why: Adult has strong pairwise interactions (education×occupation, marital×sex, etc.). FM models all pairs with O(d·k) params/compute (d=14, k=4), avoiding one-hot explosion of linear and heavy MLP layers. Raw byte indices (0–255) as embedding inputs preserve ordinal order and categorical similarity. Expects >0.855 dev accuracy at <1000 Ir/item.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 1500, the factorization does not justify its complexity.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, K, C; float32 w0[C], w[C,d,256], V[C,d,256,K]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 256, 600000, 100


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
    assert d == 14 and C == 2
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # params: w0[C], w[C,d,256], V[C,d,256,K]
    w0 = np.zeros(C, np.float32)
    w = np.zeros((C, d, 256), np.float32)
    V = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)
    params = [w0, w, V]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # forward: scores[B, C]
            scores = np.zeros((B, C), np.float32)
            for c in range(C):
                # linear term
                lin = w0[c].copy()
                for j in range(d):
                    lin += w[c, j, xb[:, j]]
                scores[:, c] = lin
                # FM term
                for f in range(K):
                    sum_v = np.zeros(B, np.float32)
                    sum_v_sq = np.zeros(B, np.float32)
                    for j in range(d):
                        v = V[c, j, xb[:, j], f]
                        sum_v += v
                        sum_v_sq += v * v
                    scores[:, c] += 0.5 * (sum_v * sum_v - sum_v_sq)

            # softmax + cross-entropy grad
            scores -= scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grads_c = probs / B  # (B, C)

            # gradients
            gw0 = np.zeros(C, np.float32)
            gw = np.zeros((C, d, 256), np.float32)
            gV = np.zeros((C, d, 256, K), np.float32)

            for c in range(C):
                gc = grads_c[:, c]  # (B,)
                gw0[c] = gc.sum()
                for j in range(d):
                    # linear grad
                    np.add.at(gw[c, j], xb[:, j], gc)
                    # FM grad
                    for f in range(K):
                        v_vals = V[c, j, xb[:, j], f]
                        sum_v = np.zeros(B, np.float32)
                        for jj in range(d):
                            sum_v += V[c, jj, xb[:, jj], f]
                        np.add.at(gV[c, j, :, f], xb[:, j], gc * (sum_v - v_vals))

            grads = [gw0, gw, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps) + LR * L2 * p

    # pack weights: magic, d, K, C; then w0, w, V (all float32 little-endian)
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record (d=14). Output: 1 byte class (argmax of 2 scores). */
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
    if (d != 14 || C != 2 || K <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * (C + C * d * 256 + C * d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *w = p; p += C * d * 256;
    const float *V = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score[2];
        for (int c = 0; c < C; c++) {
            float s = w0[c];
            for (int j = 0; j < d; j++) s += w[((c * d + j) * 256 + xi[j])];
            for (int f = 0; f < K; f++) {
                float sum_v = 0.0f, sum_v_sq = 0.0f;
                for (int j = 0; j < d; j++) {
                    float v = V[((c * d + j) * 256 + xi[j]) * K + f];
                    sum_v += v;
                    sum_v_sq += v * v;
                }
                s += 0.5f * (sum_v * sum_v - sum_v_sq);
            }
            score[c] = s;
        }
        unsigned char best = (score[1] > score[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```