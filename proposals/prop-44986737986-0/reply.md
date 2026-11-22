```hypothesis
Family: label-fm (factorization machine with shared embeddings for pairwise interactions).
Why: The adult dataset has 14 mixed ordinal/categorical features pre-binned to 256 values each. A factorization machine captures second-order feature interactions with O(d*K) inference cost (K=4), adding only ~86 KiB weights. It should beat label-linear (0.8524) by modeling interactions while staying faster than label-mlp16.
Failure sign: Dev accuracy ≤ 0.8524 or inference cost > label-linear.
```

```python
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 10


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

    # embeddings: (d, 256, K), linear: (d, 256, C), bias: (C,)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    lin = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [emb, lin, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward
            sum_e = np.zeros((bsz, K), np.float32)
            sum_e_sq = np.zeros((bsz, K), np.float32)
            lin_sum = bias.astype(np.float32) + np.zeros((bsz, C), np.float32)
            for j in range(d):
                v = xb[:, j]
                e = emb[j, v]                 # (bsz, K)
                sum_e += e
                sum_e_sq += e * e
                lin_sum += lin[j, v]

            inter = 0.5 * np.sum(sum_e * sum_e - sum_e_sq, axis=1, keepdims=True)  # (bsz,1)
            logits = lin_sum + inter                                        # (bsz,C)

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            prob = exp / exp.sum(axis=1, keepdims=True)
            prob[np.arange(bsz), yb] -= 1.0
            grad_logits = prob / bsz  # (bsz,C)

            # gradients
            g_bias = grad_logits.sum(axis=0)
            g_lin = np.zeros_like(lin)
            g_emb = np.zeros_like(emb)

            # interaction gradient: dL/d(inter) = sum_c grad_logits[:,c] -> (bsz,)
            g_inter = grad_logits.sum(axis=1)  # (bsz,)
            # d(inter)/d(sum_e) = sum_e, d(inter)/d(sum_e_sq) = -0.5
            g_sum_e = g_inter[:, None] * sum_e          # (bsz,K)
            g_sum_e_sq = -0.5 * g_inter[:, None]        # (bsz,K)

            for j in range(d):
                v = xb[:, j]
                # linear gradient
                np.add.at(g_lin[j], v, grad_logits)
                # embedding gradient
                # each example contributes g_sum_e[i] + 2 * e * g_sum_e_sq[i]
                g_e = g_sum_e + 2.0 * emb[j, v] * g_sum_e_sq
                np.add.at(g_emb[j], v, g_e)

            # L2 regularization
            g_emb += L2 * emb
            g_lin += L2 * lin

            grads = [g_emb, g_lin, g_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights: magic, d, K, C; then emb, lin, bias (all float32 LE)
    magic = 0x464D5F31  # "FM_1"
    out = struct.pack("<4i", magic, d, K, C)
    out += emb.astype("<f4").tobytes()
    out += lin.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || d < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 16
        + (size_t)d * 256 * K * sizeof(float)
        + (size_t)d * 256 * C * sizeof(float)
        + (size_t)C * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *lin = emb + (size_t)d * 256 * K;
    const float *bias = lin + (size_t)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_e[16], sum_e_sq[16], logits[2];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int kk = 0; kk < K; kk++) { sum_e[kk] = 0.0f; sum_e_sq[kk] = 0.0f; }
        logits[0] = bias[0]; logits[1] = bias[1];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *e = emb + ((size_t)j * 256 + v) * K;
            const float *l = lin + ((size_t)j * 256 + v) * C;
            for (int kk = 0; kk < K; kk++) {
                float ev = e[kk];
                sum_e[kk] += ev;
                sum_e_sq[kk] += ev * ev;
            }
            logits[0] += l[0];
            logits[1] += l[1];
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += sum_e[kk] * sum_e[kk] - sum_e_sq[kk];
        inter *= 0.5f;
        logits[0] += inter;
        logits[1] += inter;
        unsigned char best = logits[1] > logits[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```