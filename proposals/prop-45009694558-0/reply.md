```hypothesis
Family: label-fm (factorization machine with per-value embeddings).
Why: Adult has 14 byte-valued features (ordinal bins + categorical indices). A factorization machine captures pairwise interactions via low-rank embeddings, which logistic regression misses, while staying far cheaper than GBDT. With embedding dim k=8 the model has ~32k parameters; inference is ~14*k FLOPs/item.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or CPU cost ≥ 3000 Ir/item, the trade-off is not justified.
```

```python
"""Factorization Machine for tabular byte features (FM with k=8).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format: magic 0x464D4C31, d, k, C, bias, linear[d][256], emb[d][256][k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


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
    C = card["classes"]
    assert C == 2, "FM binary only"
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = X[idx]          # (B, d) uint8
            yy = y[idx]         # (B,) int64
            B = len(idx)

            # Forward
            lin_part = bias + linear[np.arange(d), z.T].sum(axis=0)  # (B,)
            emb_vals = emb[np.arange(d), z.T]                        # (d, B, K)
            emb_sum = emb_vals.sum(axis=0)                           # (B, K)
            sq_sum = (emb_vals ** 2).sum(axis=0)                     # (B, K)
            inter = 0.5 * ((emb_sum ** 2 - sq_sum).sum(axis=1))      # (B,)
            logit = lin_part + inter                                 # (B,)
            prob = 1.0 / (1.0 + np.exp(-logit))
            dlogit = (prob - yy).astype(np.float32)                  # (B,)

            # Gradients
            g_bias = dlogit.mean()
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(g_lin[j], z[:, j], dlogit) / B
                emb_j = emb_vals[j]  # (B, K)
                grad_emb_j = dlogit[:, None] * (emb_sum - emb_j)     # (B, K)
                np.add.at(g_emb[j], z[:, j], grad_emb_j) / B

            # L2
            g_lin += L2 * linear
            g_emb += L2 * emb

            # Adam updates
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)

            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    total_floats = 1 + d * 256 + d * 256 * K
    expected_bytes = 16 + 4 * total_floats
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    assert len(out) == expected_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output one byte class (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || K < 1 || d > 256 || K > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = calloc(K, sizeof(float));
    float *sq_sum = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias;
        memset(emb_sum, 0, K * sizeof(float));
        memset(sq_sum, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int l = 0; l < K; l++) {
                float ev = e[l];
                emb_sum[l] += ev;
                sq_sum[l] += ev * ev;
            }
        }
        for (int l = 0; l < K; l++) logit += 0.5f * (emb_sum[l] * emb_sum[l] - sq_sum[l]);

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(emb_sum); free(sq_sum);
    return 0;
}
```