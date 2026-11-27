```hypothesis
Factorization Machine (label-fm) with 8-dimensional embeddings per feature value per class.
Adds pairwise feature interactions to the linear model, critical for tabular data where
combinations (e.g., education×occupation) determine income. With k=8, adds ~57k params
vs linear's ~7k; inference O(d·k·C) ≈ 224 ops. If dev accuracy ≤ 0.8524 (linear) or
Ir/item > 1500, the model fails.
```
```python
"""Factorization Machine for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Learns linear weights + k-dim embeddings
per feature value per class for pairwise interactions. Adam lr=1e-2, L2=1e-4, batch=256,
epochs=min(100, ceil(600000/n)). Exports: magic 0x464D0001, d,k,C; float32 T_lin[d][256][C],
T_emb[d][256][k][C], b[C]."""
import json, math, struct, sys

import numpy as np

K = 8
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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # params: T_lin[d,256,C], T_emb[d,256,K,C], b[C]
    scale = 0.01
    T_lin = (rng.uniform(-1, 1, (d, 256, C)) * scale).astype(np.float32)
    T_emb = (rng.uniform(-1, 1, (d, 256, K, C)) * scale).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [T_lin, T_emb, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            m = len(yb)

            # forward
            lin = b + np.sum(T_lin[np.arange(d)[:, None], xb.T], axis=0)  # (m,C)
            # interaction: 0.5 * sum_f [(sum_j v_jf)^2 - sum_j v_jf^2]
            v = T_emb[np.arange(d)[:, None], xb.T]  # (d,m,K,C)
            sum_v = v.sum(axis=0)                  # (m,K,C)
            sum_v2 = (v * v).sum(axis=0)           # (m,K,C)
            inter = 0.5 * ((sum_v * sum_v - sum_v2).sum(axis=1))  # (m,C)
            logits = lin + inter

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(m), yb] -= 1.0
            grad_out = (probs / m).astype(np.float32)  # (m,C)

            # gradients
            # b
            g_b = grad_out.sum(axis=0)
            # T_lin: for each (j,val,c) add grad_out[i,c] where xb[i,j]==val
            g_T_lin = np.zeros_like(T_lin)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_T_lin[j], (vals, slice(None)), grad_out)
            g_T_lin += L2 * T_lin

            # T_emb: grad_inter wrt v_jf = (sum_v_f - v_jf) * grad_out_c
            g_T_emb = np.zeros_like(T_emb)
            # sum_v: (m,K,C), v: (d,m,K,C), grad_out: (m,C)
            # for each j: g_v_jf = (sum_v_f - v_jf) * grad_out_c
            diff = sum_v[:, None, :, :] - v  # (m,d,K,C) via broadcasting
            # multiply by grad_out (m,1,1,C)
            g_v = diff * grad_out[:, None, None, :]  # (m,d,K,C)
            # accumulate over m
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_T_emb[j], (vals, slice(None), slice(None)), g_v[:, j])
            g_T_emb += L2 * T_emb

            grads = [g_T_lin, g_T_emb, g_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_, v_ = state[i]
                m_ *= 0.9; m_ += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                p -= LR * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)

    # pack weights: magic, d, K, C, T_lin, T_emb, b
    out = struct.pack("<4i", 0x464D0001, d, K, C)
    out += T_lin.astype("<f4").tobytes()
    out += T_emb.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L
        + 4L * d * 256 * C
        + 4L * d * 256 * K * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *T_emb = T_lin + (long)d * 256 * C;
    const float *b = T_emb + (long)d * 256 * K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K * C);
    float *sum_v2 = malloc(sizeof(float) * K * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // linear term
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *w = T_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        // interaction term
        memset(sum_v, 0, sizeof(float) * K * C);
        memset(sum_v2, 0, sizeof(float) * K * C);
        for (int j = 0; j < d; j++) {
            const float *v = T_emb + ((long)j * 256 + xi[j]) * K * C;
            for (int kk = 0; kk < K; kk++) {
                for (int c = 0; c < C; c++) {
                    float val = v[(long)kk * C + c];
                    sum_v[(long)kk * C + c] += val;
                    sum_v2[(long)kk * C + c] += val * val;
                }
            }
        }
        for (int kk = 0; kk < K; kk++) {
            for (int c = 0; c < C; c++) {
                float sv = sum_v[(long)kk * C + c];
                float sv2 = sum_v2[(long)kk * C + c];
                scores[c] += 0.5f * (sv * sv - sv2);
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```