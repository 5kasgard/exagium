```hypothesis
Factorization Machine (FM) with low-rank embeddings captures pairwise feature interactions effectively for tabular data. It should outperform label-linear by modeling interactions while remaining faster and smaller than GBDT/MLP. If dev accuracy ≤ label-linear (0.8524) or inference cost (ns/item) ≥ label-linear (60.3), the family does not meet its goal.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D3132, d, k, C; float32 w0, w[d][256], v[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8                 # embedding rank
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
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "FM binary only"
    rng = np.random.default_rng(SEED)

    # Parameters
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)

            # Linear term: w0 + sum_j w[j, xb[:,j]]
            linear = w0 + w[np.arange(d), xb.T].T.sum(axis=1)   # (B,)

            # Interaction term: 0.5 * sum_f ( (sum_j v[j, xb[:,j], f])^2 - sum_j v[j, xb[:,j], f]^2 )
            v_gather = v[np.arange(d), xb.T].transpose(1, 0, 2)   # (B, d, K)
            sum_v = v_gather.sum(axis=1)                          # (B, K)
            sum_v_sq = (v_gather ** 2).sum(axis=1)                # (B, K)
            interaction = 0.5 * ((sum_v ** 2 - sum_v_sq).sum(axis=1))

            logits = linear + interaction                         # (B,)
            p = 1.0 / (1.0 + np.exp(-logits))
            err = p - yb                                          # (B,)  derivative of logistic loss

            # Gradients
            g_w0 = err.mean()
            # dw[j, val] = mean_{i: xb[i,j]==val} err[i]
            g_w = np.zeros_like(w)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_w[j], vals, err)
            g_w /= len(err)

            # dv[j, val, f] = mean_{i: xb[i,j]==val} err[i] * (sum_v[i,f] - v[j, val, f])
            g_v = np.zeros_like(v)
            for j in range(d):
                vals = xb[:, j]
                for f in range(K):
                    tmp = err * (sum_v[:, f] - v_gather[:, j, f])
                    np.add.at(g_v[j, :, f], vals, tmp)
            g_v /= len(err)

            # L2 regularization
            g_w0 += L2 * w0
            g_w += L2 * w
            g_v += L2 * v

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # v
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)
            v -= LR * m_v_hat / (np.sqrt(v_v_hat) + eps)

    # Write weights.bin
    magic = 0x464D3132
    header = struct.pack("<4i", magic, d, K, C)
    body = struct.pack("<f", w0) + w.astype("<f4").tobytes() + v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output record: 1 byte class index (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0 || nw != 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            linear += w[(long)j * 256 + val];
            const float *vj = v + ((long)j * 256 + val) * K;
            for (int f = 0; f < K; f++) {
                float vjf = vj[f];
                sum_v[f] += vjf;
                sum_v_sq[f] += vjf * vjf;
            }
        }

        float interaction = 0.0f;
        for (int f = 0; f < K; f++) {
            interaction += sum_v[f] * sum_v[f] - sum_v_sq[f];
        }
        interaction *= 0.5f;

        float logit = linear + interaction;
        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```