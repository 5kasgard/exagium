```hypothesis
Family: label-fm (factorization machine with per-field embeddings).
Why: Adult has strong pairwise interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. A factorization machine captures all pairwise interactions via low-rank embeddings (k=8), adding only ~130k parameters. It should beat label-linear (0.8524) while staying faster than label-mlp16 (2325 Ir/item).
Failure signal: dev accuracy ≤ 0.8524 or instructions/item ≥ 719 (label-linear cost).
```

```python
"""Factorization Machine for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Encodes each of the d=14 fields as 256 categories. Learns bias, linear weights per field-value,
and k-dimensional embeddings per field-value. Optimizes binary cross-entropy with Adam.
Exports: magic 0x464D4C31, d, k, C=2; float32 bias, linear[d*256], emb[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 8
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 20
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # parameters
    bias = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            b = xb.shape[0]

            # forward
            # linear term: (b,)
            lin_term = np.zeros(b, np.float32)
            for f in range(d):
                lin_term += linear[f, xb[:, f]]
            # interaction term: (b,)
            inter_term = np.zeros(b, np.float32)
            for f in range(d):
                emb_f = emb[f, xb[:, f]]          # (b, K)
                for g in range(f + 1, d):
                    emb_g = emb[g, xb[:, g]]      # (b, K)
                    inter_term += np.sum(emb_f * emb_g, axis=1)
            logits = bias + lin_term + inter_term
            probs = 1.0 / (1.0 + np.exp(-logits))
            # gradient of BCE w.r.t logits
            grad_logits = (probs - yb).astype(np.float32)  # (b,)

            # gradients
            g_bias = grad_logits.mean()
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            for f in range(d):
                # linear gradient
                for val in range(256):
                    mask = (xb[:, f] == val)
                    if np.any(mask):
                        g_lin[f, val] = grad_logits[mask].mean()
                # embedding gradient
                emb_f = emb[f, xb[:, f]]  # (b, K)
                for g in range(f + 1, d):
                    emb_g = emb[g, xb[:, g]]
                    # dL/demb_f = grad_logits * emb_g
                    # dL/demb_g = grad_logits * emb_f
                    for k in range(K):
                        g_emb[f, xb[:, f], k] += grad_logits * emb_g[:, k]
                        g_emb[g, xb[:, g], k] += grad_logits * emb_f[:, k]

            # average over batch
            g_lin /= b
            g_emb /= b

            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * emb

            # Adam update
            t += 1
            # bias
            m_bias = BETA1 * m_bias + (1 - BETA1) * g_bias
            v_bias = BETA2 * v_bias + (1 - BETA2) * g_bias * g_bias
            m_hat = m_bias / (1 - BETA1 ** t)
            v_hat = v_bias / (1 - BETA2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + EPS)
            # linear
            m_lin = BETA1 * m_lin + (1 - BETA1) * g_lin
            v_lin = BETA2 * v_lin + (1 - BETA2) * g_lin * g_lin
            linear -= LR * (m_lin / (1 - BETA1 ** t)) / (np.sqrt(v_lin / (1 - BETA2 ** t)) + EPS)
            # emb
            m_emb = BETA1 * m_emb + (1 - BETA1) * g_emb
            v_emb = BETA2 * v_emb + (1 - BETA2) * g_emb * g_emb
            emb -= LR * (m_emb / (1 - BETA1 ** t)) / (np.sqrt(v_emb / (1 - BETA2 ** t)) + EPS)

    # export
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<3i", magic, d, K)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0/1).
 * Weights format: magic(0x464D4C31), int32 d, int32 k, float bias,
 *   float linear[d*256], float emb[d*256*k]. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 12 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2];
    if (d != 14 || k != 8) { fprintf(stderr, "unexpected d/k\n"); return 3; }
    long expected = 12 + 4 + 4L * d * 256 * (1 + k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias = (const float *)(hd + 3);
    const float *linear = bias + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score = *bias;
        // linear term
        for (int f = 0; f < d; f++) {
            score += linear[f * 256 + xi[f]];
        }
        // pairwise interactions
        for (int f = 0; f < d; f++) {
            const float *emb_f = emb + (long)(f * 256 + xi[f]) * k;
            for (int g = f + 1; g < d; g++) {
                const float *emb_g = emb + (long)(g * 256 + xi[g]) * k;
                for (int kk = 0; kk < k; kk++) {
                    score += emb_f[kk] * emb_g[kk];
                }
            }
        }
        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char pred = prob > 0.5f ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```