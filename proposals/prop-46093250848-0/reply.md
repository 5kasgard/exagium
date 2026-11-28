```hypothesis
Family: label-fm (Factorization Machine with 8‑dim embeddings).
Reason: FMs capture pairwise feature interactions crucial for tabular data like Adult, are more expressive than linear models, and with k=8 remain lightweight (≈32k params) for fast inference. The ordered quantile bins and categorical indices are naturally handled as categorical variables.
Failure signal: Dev accuracy ≤ 0.8524 (label‑linear) or inference cost > 2× label‑linear.
```

```python
"""Factorization Machine for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports: magic 0x464D3131, d, k, C, bias, linear[d][256], emb[d][256][k] (all float32 LE)."""
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
    C = card["classes"]
    assert C == 2
    d = X.shape[1]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters
    bias = np.float32(0.0)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            bsz = len(idx)

            # Forward
            lin_part = linear[np.arange(d), xb.T].sum(axis=0)          # (b,)
            emb_selected = emb[np.arange(d), xb.T]                     # (d, b, K)
            sum_emb = emb_selected.sum(axis=0)                         # (b, K)
            sum_sq_emb = (emb_selected ** 2).sum(axis=0)               # (b, K)
            inter_part = 0.5 * ((sum_emb ** 2).sum(axis=1) - (sum_sq_emb).sum(axis=1))
            score = bias + lin_part + inter_part                       # (b,)

            # Loss gradient (binary cross‑entropy with y in {0,1})
            sig = 1.0 / (1.0 + np.exp(-score))
            g = (sig - yb).astype(np.float32)                          # (b,)

            # Gradients
            dbias = g.mean()
            # linear gradients: for each feature j and value v, average g over batch where xb[:,j]==v
            dlin = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(dlin[j], vals, g)
            dlin /= bsz
            dlin += L2 * linear

            # embedding gradients: dI/demb_{j,v} = g * (sum_emb - emb_{j,v})
            demb = np.zeros_like(emb)
            for j in range(d):
                vals = xb[:, j]
                # g[:,None] * (sum_emb - emb[j, vals])
                diff = sum_emb - emb[j, vals]                          # (b, K)
                # accumulate per value
                for v in range(256):
                    mask = (vals == v)
                    if mask.any():
                        demb[j, v] = (g[mask, None] * diff[mask]).sum(axis=0)
            demb /= bsz
            demb += L2 * emb

            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * dbias
            v_bias = 0.999 * v_bias + 0.001 * dbias * dbias
            m_hat = m_bias / (1 - 0.9 ** t)
            v_hat = v_bias / (1 - 0.999 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            # linear
            m_lin = 0.9 * m_lin + 0.1 * dlin
            v_lin = 0.999 * v_lin + 0.001 * dlin * dlin
            m_hat = m_lin / (1 - 0.9 ** t)
            v_hat = v_lin / (1 - 0.999 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            # emb
            m_emb = 0.9 * m_emb + 0.1 * demb
            v_emb = 0.999 * v_emb + 0.001 * demb * demb
            m_hat = m_emb / (1 - 0.9 ** t)
            v_hat = v_emb / (1 - 0.999 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic, d, K, C
    out = struct.pack("<4i", 0x464D3131, d, K, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1 || d > 256 || K > 256) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_sq_emb = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = bias[0];
        memset(sum_emb, 0, sizeof(float) * K);
        memset(sum_sq_emb, 0, sizeof(float) * K);

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = e[u];
                sum_emb[u] += val;
                sum_sq_emb[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += sum_emb[u] * sum_emb[u] - sum_sq_emb[u];
        score += 0.5f * inter;

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```