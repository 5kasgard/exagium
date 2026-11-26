```hypothesis
Family: label-fm (Factorization Machine with binary output).
Why: Adult has 14 mixed-type features; a 2nd-order FM captures pairwise interactions cheaply (k=4 embeddings → 91 dot products per item) and often beats linear models on tabular data while staying far faster than MLP/GBDT.
Failure signal: dev accuracy ≤ label-linear (0.8524) or inference cost > 2× label-linear.
```
```python
"""label-fm: Factorization Machine for binary label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Learns bias, per-feature-value linear weights (14×256), and embeddings (14×256×k).
k=4, Adam lr=1e-2, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format (little-endian):
  magic=0x464D424C, d=14, k=4, C=2 (int32 each)
  bias (float32)
  linear_weights[d*256] (float32)
  embeddings[d*256*k] (float32)"""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)  # 0 or 1
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # parameters
    bias = np.float32(0.0)
    lin = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward
            scores = np.full(bsz, bias, dtype=np.float32)
            for j in range(d):
                scores += lin[j, xb[:, j]]
                emb_j = emb[j, xb[:, j]]  # (bsz, K)
            # pairwise dot products
            for i in range(d):
                ei = emb[i, xb[:, i]]
                for j in range(i + 1, d):
                    scores += (ei * emb[j, xb[:, j]]).sum(axis=1)

            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-scores))
            grad = p - yb  # (bsz,)

            # backward
            g_bias = grad.mean()
            g_lin = np.zeros_like(lin)
            g_emb = np.zeros_like(emb)

            for j in range(d):
                # linear gradient
                for v in range(256):
                    mask = (xb[:, j] == v)
                    if mask.any():
                        g_lin[j, v] = grad[mask].mean() + L2 * lin[j, v]
                # embedding gradient
                ej = emb[j, xb[:, j]]  # (bsz, K)
                # sum over other features' embeddings weighted by grad
                sum_other = np.zeros((bsz, K), dtype=np.float32)
                for i in range(d):
                    if i == j:
                        continue
                    sum_other += emb[i, xb[:, i]]
                for v in range(256):
                    mask = (xb[:, j] == v)
                    if mask.any():
                        g_emb[j, v] = (grad[mask, None] * sum_other[mask]).mean(axis=0) + L2 * emb[j, v]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            lin -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    # write weights
    magic = 0x464D424C  # 'LBMF' little-endian
    hdr = struct.pack("<4i", magic, d, K, 2)
    out = hdr + struct.pack("<f", bias) + lin.astype("<f4").tobytes() + emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K != 4 || C != 2) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = *p++;                     // bias
    const float *lin = p; p += d * 256;    // linear weights [d][256]
    const float *emb = p;                  // embeddings [d][256][K]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias;
        // linear part
        for (int j = 0; j < d; j++) score += lin[j * 256 + xi[j]];
        // pairwise embeddings
        for (int i1 = 0; i1 < d; i1++) {
            const float *e1 = emb + (i1 * 256 + xi[i1]) * K;
            for (int i2 = i1 + 1; i2 < d; i2++) {
                const float *e2 = emb + (i2 * 256 + xi[i2]) * K;
                for (int kk = 0; kk < K; kk++) score += e1[kk] * e2[kk];
            }
        }
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```