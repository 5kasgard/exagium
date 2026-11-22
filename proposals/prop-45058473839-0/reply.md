```hypothesis
Factorization Machine (FM) with binary cross-entropy: learns a bias, per-feature-value linear weights, and low-rank (K=8) embeddings for pairwise interactions. Captures feature crosses crucial for tabular data while keeping inference extremely fast (O(d·K) per item). Should beat label-linear (0.8524) on accuracy; if dev accuracy ≤0.852 or Ir/item >1000, the model does not justify its complexity.
```
```python
"""Factorization Machine for tabular binary classification.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D3130, d, K; float32 bias, linear[d*256], emb[d*256*K]."""
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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias = v_bias = np.float32(0.0)
    m_lin = np.zeros_like(linear); v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb); v_emb = np.zeros_like(emb)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            # linear term: bias + sum_j linear[j, xb[:,j]]
            lin_term = bias + np.sum(linear[np.arange(d)[:, None], xb.T], axis=0)  # (B,)
            # embeddings: (d, B, K)
            emb_batch = emb[np.arange(d)[:, None], xb.T]
            sum_v = emb_batch.sum(axis=0)          # (B, K)
            sum_sq_v = (emb_batch ** 2).sum(axis=0)  # (B, K)
            inter = 0.5 * ((sum_v ** 2 - sum_sq_v).sum(axis=1))  # (B,)
            logit = lin_term + inter

            # BCE gradient
            prob = 1.0 / (1.0 + np.exp(-logit))
            dlogit = (prob - yb) / B  # (B,)

            # Backward
            dbias = dlogit.sum()

            dlin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(dlin[j], xb[:, j], dlogit)

            demb = np.zeros_like(emb)
            for j in range(d):
                grad = dlogit[:, None] * (sum_v - emb_batch[j])  # (B, K)
                np.add.at(demb[j], xb[:, j], grad)

            # L2
            dlin += L2 * linear
            demb += L2 * emb

            # Adam updates
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * dbias
            v_bias = 0.999 * v_bias + 0.001 * dbias * dbias
            bias -= LR * (m_bias / (1 - 0.9 ** t)) / (np.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)
            # linear
            m_lin = 0.9 * m_lin + 0.1 * dlin
            v_lin = 0.999 * v_lin + 0.001 * dlin * dlin
            linear -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)
            # emb
            m_emb = 0.9 * m_emb + 0.1 * demb
            v_emb = 0.999 * v_emb + 0.001 * demb * demb
            emb -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    # Write weights
    magic = 0x464D3130  # "FM10"
    out = struct.pack("<3i", magic, d, K)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1). */
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
    if (nw < 12 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2];
    if (d <= 0 || d > 256 || K <= 0 || K > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 12 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 3);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[64], sum_sq_v[64];
    unsigned char pred;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int kk = 0; kk < K; kk++) { sum_v[kk] = 0.0f; sum_sq_v[kk] = 0.0f; }
        float logit = bias;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = e[kk];
                sum_v[kk] += val;
                sum_sq_v[kk] += val * val;
            }
        }

        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) inter += sum_v[kk] * sum_v[kk] - sum_sq_v[kk];
        logit += 0.5f * inter;
        pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```