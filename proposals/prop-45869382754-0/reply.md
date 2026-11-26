```hypothesis
Family: label-fm (Factorization Machine with per-byte lookup tables). 
Why: Adult is a tabular dataset where pairwise feature interactions (education×occupation, marital-status×relationship, etc.) are known to be predictive. A rank-4 FM captures these interactions with only O(d·K) extra inference work over linear (K=4 latent dims), keeping CPU cost near label-linear (~800 Ir/item) while gaining accuracy. 
Failure signal: dev accuracy ≤ 0.8524 (label-linear) would show interactions don't help or K=4 is insufficient.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: per-byte lookup tables for 1st-order (T1) and 2nd-order (T2) embeddings.
FM: logit_c = b0_c + sum_j T1[j][x_j]_c + 0.5 * w2_c^T ( (sum_j v_j)^2 - sum_j v_j^2 )
where v_j = T2[j][x_j] in R^K. Trained with Adam on softmax cross-entropy.
Exported: magic 0x464D3100, d, K, C; float32 T1[d][256][C], T2[d][256][K], b0[C], w2[C][K]."""
import json, math, struct, sys

import numpy as np

K = 4
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
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    T1 = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    T2 = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    b0 = np.zeros(C, np.float32)
    w2 = (rng.normal(0, 0.01, (C, K))).astype(np.float32)

    params = [T1, T2, b0, w2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            linear = np.zeros((B, C), np.float32)
            sum_v = np.zeros((B, K), np.float32)
            sum_v2 = np.zeros((B, K), np.float32)

            for j in range(d):
                v = xb[:, j]
                linear += T1[j, v]
                t2 = T2[j, v]
                sum_v += t2
                sum_v2 += t2 * t2

            linear += b0
            interaction = 0.5 * (sum_v * sum_v - sum_v2)
            logits = linear + interaction @ w2.T

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B

            grad_w2 = grad_logits.T @ interaction
            grad_interaction = grad_logits @ w2
            grad_sum_v = grad_interaction * sum_v
            grad_sum_v2 = -0.5 * grad_interaction
            grad_linear = grad_logits
            grad_b0 = grad_linear.sum(axis=0)

            grad_T1 = np.zeros_like(T1)
            grad_T2 = np.zeros_like(T2)
            for j in range(d):
                v = xb[:, j]
                np.add.at(grad_T1[j], v, grad_linear)
                t2 = T2[j, v]
                grad_t2 = grad_sum_v + 2 * t2 * grad_sum_v2
                np.add.at(grad_T2[j], v, grad_t2)

            grad_T1 += L2 * T1
            grad_T2 += L2 * T2
            grad_b0 += L2 * b0
            grad_w2 += L2 * w2

            grads = [grad_T1, grad_T2, grad_b0, grad_w2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    out = struct.pack("<4i", 0x464D3100, d, K, C)
    out += T1.astype("<f4").tobytes()
    out += T2.astype("<f4").tobytes()
    out += b0.astype("<f4").tobytes()
    out += w2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (rank K=4) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights: magic 0x464D3100, d, K, C; float32 T1[d][256][C], T2[d][256][K], b0[C], w2[C][K]. */
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
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16;
    expected += 4L * d * 256 * C;
    expected += 4L * d * 256 * K;
    expected += 4L * C;
    expected += 4L * C * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T1 = (const float *)(hd + 4);
    const float *T2 = T1 + (long)d * 256 * C;
    const float *b0 = T2 + (long)d * 256 * K;
    const float *w2 = b0 + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);
    float *interaction = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) acc[c] = b0[c];
        for (int kk = 0; kk < K; kk++) { sum_v[kk] = 0.0f; sum_v2[kk] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *t1 = T1 + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) acc[c] += t1[c];

            const float *t2 = T2 + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = t2[kk];
                sum_v[kk] += val;
                sum_v2[kk] += val * val;
            }
        }

        for (int kk = 0; kk < K; kk++) {
            interaction[kk] = 0.5f * (sum_v[kk] * sum_v[kk] - sum_v2[kk]);
        }

        for (int c = 0; c < C; c++) {
            const float *w2c = w2 + (long)c * K;
            float dot = 0.0f;
            for (int kk = 0; kk < K; kk++) dot += w2c[kk] * interaction[kk];
            acc[c] += dot;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    return 0;
}
```