```hypothesis
Family: label-fm (Factorization Machine with rank-8 latent vectors).  
Why: Adult has strong pairwise feature interactions (e.g., education×occupation, marital-status×relationship) that a linear model misses. FM captures all pairwise interactions via low-rank factorization, adding only O(d*K) parameters and inference cost. With d=14, K=8, it should beat label-linear (0.8524) while staying far cheaper than GBDT (7570 Ir/item).  
Failure signal: dev accuracy ≤ 0.8524 or Ir/item ≥ 2000 (cost approaching label-mlp16).
```
```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    b = np.zeros(C, np.float32)
    w = (rng.normal(0, 0.01, (C, d, 256)) * math.sqrt(1.0 / 256)).astype(np.float32)
    v = (rng.normal(0, 0.01, (C, d, 256, K)) * math.sqrt(1.0 / (256 * K))).astype(np.float32)

    params = [b, w, v]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            scores = b[None, :] + np.sum(w[:, xb.T], axis=1).T
            for k in range(K):
                v_k = v[:, :, :, k]
                sum_v = np.zeros((bs, C), np.float32)
                sum_v_sq = np.zeros((bs, C), np.float32)
                for j in range(d):
                    val = v_k[:, j, xb[:, j]].T
                    sum_v += val
                    sum_v_sq += val * val
                scores += 0.5 * (sum_v * sum_v - sum_v_sq)

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grads_scores = probs / bs

            grad_b = grads_scores.sum(axis=0)
            grad_w = np.zeros_like(w)
            grad_v = np.zeros_like(v)

            for j in range(d):
                for c in range(C):
                    np.add.at(grad_w[c, j], xb[:, j], grads_scores[:, c])

            for k in range(K):
                v_k = v[:, :, :, k]
                sum_v = np.zeros((bs, C), np.float32)
                for j in range(d):
                    sum_v += v_k[:, j, xb[:, j]].T
                for j in range(d):
                    val = v_k[:, j, xb[:, j]].T
                    diff = grads_scores * (sum_v - val)
                    for c in range(C):
                        np.add.at(grad_v[c, j, :, k], xb[:, j], diff[:, c])

            grads = [grad_b, grad_w, grad_v]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                gr += L2 * p
                m, v_state = state[i]
                m *= 0.9; m += 0.1 * gr
                v_state *= 0.999; v_state += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v_state / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x464D4C31
    out = struct.pack("<4i", magic, d, K, C)
    out += b.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 32) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * (C + (long)C * d * 256 + (long)C * d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *b = p; p += C;
    const float *w = p; p += (long)C * d * 256;
    const float *v = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        scores[0] = b[0]; scores[1] = b[1];

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            scores[0] += w[((0 * d + j) * 256 + val)];
            scores[1] += w[((1 * d + j) * 256 + val)];
        }

        for (int kk = 0; kk < K; kk++) {
            float sum_v[2] = {0, 0};
            float sum_v_sq[2] = {0, 0};
            for (int j = 0; j < d; j++) {
                unsigned char val = xi[j];
                for (int c = 0; c < 2; c++) {
                    float vval = v[((long)c * d * 256 * K + (long)j * 256 * K + val * K + kk)];
                    sum_v[c] += vval;
                    sum_v_sq[c] += vval * vval;
                }
            }
            for (int c = 0; c < 2; c++) {
                scores[c] += 0.5f * (sum_v[c] * sum_v[c] - sum_v_sq[c]);
            }
        }

        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```