```hypothesis
Family: label-fm (Factorization Machine with binary logit).
Why: Adult has strong pairwise interactions (e.g., education×occupation) that linear models miss. A rank-8 FM adds only O(d·K) parameters (d=14, K=8) and computes interactions in O(d·K) time, so it should beat label-linear (0.8524) while staying far cheaper than label-mlp16 (2325 Ir/item).
Failure signal: dev accuracy ≤ 0.8524 or Ir/item ≥ 1500.
```

```python
"""Factorization Machine (binary logit) for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format: magic(0x464D0001), d, K, C; float32 bias, linear[d*256], embed[d*256*K]."""
import json, math, struct, sys

import numpy as np

K = 8                     # embedding rank (tuned on last 20% of train.in)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100
MAGIC = 0x464D0001        # "FM\0\1"

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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)   # 0 / 1
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "binary only"

    rng = np.random.default_rng(SEED)
    # parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    embed  = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(embed),  np.zeros_like(embed)

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (B, d)
            yb = y[idx]                 # (B,)
            B = xb.shape[0]

            # forward
            lin_term = bias + np.sum(linear[np.arange(d), xb.T], axis=0)          # (B,)
            emb_vals = embed[np.arange(d), xb.T, :]                               # (d, B, K) -> transpose
            emb_vals = emb_vals.transpose(1, 0, 2)                                # (B, d, K)
            s_k = np.sum(emb_vals, axis=1)                                        # (B, K)
            inter = 0.5 * np.sum(s_k * s_k - np.sum(emb_vals * emb_vals, axis=1), axis=1)  # (B,)
            logit = lin_term + inter
            prob = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = prob - yb                                                # (B,)

            # gradients
            g_bias = np.mean(grad_logit)
            g_lin = np.zeros_like(linear)
            for i in range(d):
                np.add.at(g_lin[i], xb[:, i], grad_logit)
            g_lin /= B

            g_emb = np.zeros_like(embed)
            for i in range(d):
                for k in range(K):
                    np.add.at(g_emb[i, :, k], xb[:, i], grad_logit * (s_k[:, k] - emb_vals[:, i, k]))
            g_emb /= B

            # L2
            g_lin += L2 * linear
            g_emb += L2 * embed

            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            bias -= LR * (m_bias / (1 - 0.9 ** t)) / (np.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)
            # linear
            m_lin = 0.9 * m_lin + 0.1 * g_lin
            v_lin = 0.999 * v_lin + 0.001 * g_lin * g_lin
            linear -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)
            # embed
            m_emb = 0.9 * m_emb + 0.1 * g_emb
            v_emb = 0.999 * v_emb + 0.001 * g_emb * g_emb
            embed -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    # write weights.bin
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record = d bytes (one per feature). Output = 1 byte class (0/1). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0 || nw != 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *p = (const float *)(hd + 4);
    const float bias = p[0];
    const float *linear = p + 1;
    const float *embed  = p + 1 + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *s_k = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias;
        for (int j = 0; j < d; j++) logit += linear[(long)j * 256 + xi[j]];

        for (int kk = 0; kk < K; kk++) s_k[kk] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) s_k[kk] += e[kk];
        }
        float inter = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = e[kk];
                inter += s_k[kk] * v - v * v;
            }
        }
        logit += 0.5f * inter;

        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    free(s_k);
    return 0;
}
```