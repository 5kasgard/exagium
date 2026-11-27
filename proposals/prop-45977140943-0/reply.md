```hypothesis
Family: label-fm (Factorization Machine with rank-4 embeddings).
Why: Adult has known feature interactions (e.g., education×occupation, capital-gain×hours). A rank-4 FM adds only ~14×256×4≈14k parameters and ~60 FLOPs/item, capturing pairwise interactions while staying faster than label-mlp16. It should beat label-linear (0.8524) and approach label-gbdt (0.8664) at near-linear cost.
Failure signal: Dev accuracy ≤ 0.8524 or instructions/item > 1000.
```

```python
"""Factorization Machine (rank k) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x464D4C31, d, k, C; float32 bias, linear[d][256], embed[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 30
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


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
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    bias = np.float32(0.0)
    lin = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward
            lin_sum = bias + lin[np.arange(d)[:, None], xb.T].sum(axis=0)  # (bsz,)
            emb_sum = emb[np.arange(d)[:, None], xb.T].sum(axis=0)         # (bsz, K)
            emb_sq_sum = (emb[np.arange(d)[:, None], xb.T] ** 2).sum(axis=0)  # (bsz, K)
            inter = 0.5 * ((emb_sum ** 2 - emb_sq_sum).sum(axis=1))        # (bsz,)
            score = lin_sum + inter                                        # (bsz,)

            # Logistic loss gradient
            sig = 1.0 / (1.0 + np.exp(-score))
            grad_score = (sig - yb).astype(np.float32)                     # (bsz,)

            # Backward
            dbias = grad_score.mean()
            dlin = np.zeros_like(lin)
            demb = np.zeros_like(emb)

            # Linear gradients
            for j in range(d):
                vals = xb[:, j]
                np.add.at(dlin[j], vals, grad_score)
            dlin /= bsz

            # Embedding gradients
            # d(inter)/demb = grad_score * (emb_sum - emb_val)
            for j in range(d):
                vals = xb[:, j]
                for f in range(K):
                    # emb_sum[:, f] is (bsz,), emb_val is (bsz,)
                    emb_val = emb[j, vals, f]
                    diff = emb_sum[:, f] - emb_val
                    g = grad_score * diff
                    np.add.at(demb[j, :, f], vals, g)
            demb /= bsz

            # L2 regularization
            dbias += L2 * bias
            dlin += L2 * lin
            demb += L2 * emb

            # Adam update
            t += 1
            # bias
            m_bias = ADAM_B1 * m_bias + (1 - ADAM_B1) * dbias
            v_bias = ADAM_B2 * v_bias + (1 - ADAM_B2) * dbias * dbias
            m_hat = m_bias / (1 - ADAM_B1 ** t)
            v_hat = v_bias / (1 - ADAM_B2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

            # linear
            m_lin = ADAM_B1 * m_lin + (1 - ADAM_B1) * dlin
            v_lin = ADAM_B2 * v_lin + (1 - ADAM_B2) * dlin * dlin
            m_hat = m_lin / (1 - ADAM_B1 ** t)
            v_hat = v_lin / (1 - ADAM_B2 ** t)
            lin -= LR * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

            # embedding
            m_emb = ADAM_B1 * m_emb + (1 - ADAM_B1) * demb
            v_emb = ADAM_B2 * v_emb + (1 - ADAM_B2) * demb * demb
            m_hat = m_emb / (1 - ADAM_B1 ** t)
            v_hat = v_emb / (1 - ADAM_B2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += lin.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output record: 1 byte class (0/1). */
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
    if (C != 2 || d <= 0 || K <= 0 || K > 64) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *lin = p + 1;
    const float *emb = lin + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = calloc(K, sizeof(float));
    float *emb_sq_sum = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = bias;
        memset(emb_sum, 0, K * sizeof(float));
        memset(emb_sq_sum, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin_sum += lin[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                emb_sum[f] += val;
                emb_sq_sum[f] += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            inter += emb_sum[f] * emb_sum[f] - emb_sq_sum[f];
        }
        inter *= 0.5f;

        float score = lin_sum + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(emb_sum); free(emb_sq_sum);
    return 0;
}
```