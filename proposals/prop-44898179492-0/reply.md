```hypothesis
Family: label-fm (Factorization Machine with 8-dimensional embeddings).
Why: Adult has 14 mixed-type features; pairwise interactions capture non-linearities missed by linear models while staying far cheaper than trees or MLPs. FMs are known to excel on sparse/tabular data with categorical features.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear by >2×, the family does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D3130, d, k, C, bias, linear[d*256], embed[d*256*k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

K = 8                 # embedding dimension
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 20

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
    assert C == 2

    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    idx = X + (np.arange(d, dtype=np.uint16) * 256)  # (n, d) global indices 0..d*256-1

    # Parameters
    bias = np.float32(0.0)
    linear = rng.uniform(-0.01, 0.01, d * 256).astype(np.float32)
    embed  = rng.uniform(-0.01, 0.01, (d * 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed),  np.zeros_like(embed)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s+BATCH]
            xi = idx[batch_idx]          # (B, d)
            yi = y[batch_idx]            # (B,)

            # Forward
            lin_part = linear[xi].sum(axis=1)                       # (B,)
            emb_sum  = embed[xi].sum(axis=1)                        # (B, K)
            emb_sq   = (embed[xi] ** 2).sum(axis=1)                 # (B, K)
            inter    = 0.5 * ((emb_sum ** 2 - emb_sq).sum(axis=1))  # (B,)
            logit    = bias + lin_part + inter
            p1 = 1.0 / (1.0 + np.exp(-logit))
            grad = p1 - yi                                          # (B,)

            # Gradients
            g_bias = grad.mean()
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(embed)

            np.add.at(g_lin, xi.ravel(), grad.repeat(d))
            for f in range(K):
                sum_f = emb_sum[:, f]                               # (B,)
                np.add.at(g_emb[:, f], xi.ravel(),
                          (grad * (sum_f[:, None] - embed[xi, f])).ravel())

            # L2
            g_lin += L2 * linear
            g_emb += L2 * embed

            # Adam update
            t += 1
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= LR * (m_bias / (1 - beta1**t)) / (np.sqrt(v_bias / (1 - beta2**t)) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1**t)) / (np.sqrt(v_lin / (1 - beta2**t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            embed -= LR * (m_emb / (1 - beta1**t)) / (np.sqrt(v_emb / (1 - beta2**t)) + eps)

    # Pack weights
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes. Output: one byte class index (0 or 1). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint8_t *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4 + (long)d*256*4 + (long)d*256*K*4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *embed = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = *bias;
        for (int f = 0; f < K; f++) emb_sum[f] = 0.0f;

        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            logit += linear[idx];
            const float *e = embed + (long)idx * K;
            for (int f = 0; f < K; f++) emb_sum[f] += e[f];
        }
        for (int f = 0; f < K; f++) {
            float sq = 0.0f;
            for (int j = 0; j < d; j++) {
                int idx = j * 256 + xi[j];
                float v = embed[(long)idx * K + f];
                sq += v * v;
            }
            logit += 0.5f * (emb_sum[f] * emb_sum[f] - sq);
        }

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```