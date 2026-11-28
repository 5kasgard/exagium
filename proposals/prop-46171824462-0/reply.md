```hypothesis
Family: label-fm (Factorization Machine with 8-dimensional embeddings).
Why: Adult has strong feature interactions (e.g., education×occupation, marital×sex). FM captures pairwise interactions with O(d*K) parameters and inference cost, much cheaper than GBDT or MLP. With K=8 it adds ~14*256*8 ≈ 28k params vs linear's ~3.6k, still tiny.
Failure: If dev accuracy ≤ label-linear (0.8524) or instructions/item ≥ 2× label-linear (~1400 Ir/item), the model is not worthwhile.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: each of the 14 columns is a byte 0-255 (ordinal quantile bins, categorical indices, 255=unseen).
Model: bias + linear[j][v] + sum_{f=1..K} 0.5*( (sum_j emb[j][v][f])^2 - sum_j emb[j][v][f]^2 ).
Trained with Adam (lr=0.01, L2=1e-4), batch=256, epochs=20.
Weights format: magic 0x464D424C, int32 d, K, C; float32 bias, linear[d][256], emb[d][256][K] (all little-endian)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 20


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
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            bsize = xb.shape[0]

            # forward
            lin_part = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (b,)
            emb_gathered = emb[np.arange(d), xb.T]  # (d, b, K)
            sum_e = np.sum(emb_gathered, axis=0)    # (b, K)
            sum_e2 = np.sum(emb_gathered ** 2, axis=0)
            inter = 0.5 * np.sum(sum_e ** 2 - sum_e2, axis=1)
            logits = lin_part + inter

            # logistic loss gradient
            sig = 1.0 / (1.0 + np.exp(-logits))
            grad_logits = (sig - yb).astype(np.float32)  # (b,)

            # gradients
            g_bias = np.mean(grad_logits)

            g_lin = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        g_lin[j, v] = np.mean(grad_logits[mask]) + L2 * linear[j, v]

            g_emb = np.zeros_like(emb)
            for j in range(d):
                vals = xb[:, j]
                for v in range(256):
                    mask = (vals == v)
                    if np.any(mask):
                        # for each factor f: grad = grad_logits * (sum_e[:,f] - emb[j,v,f])
                        diff = sum_e[mask] - emb[j, v]  # (m, K)
                        g_emb[j, v] = np.mean(grad_logits[mask, None] * diff, axis=0) + L2 * emb[j, v]

            # Adam update
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
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights.bin
    magic = 0x464D424C  # "FMBL"
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0 or 1).
 * Weights format: magic 0x464D424C, int32 d, K, C; float32 bias, linear[d][256], emb[d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = *bias;
        for (int j = 0; j < d; j++) {
            logit += linear[(long)j * 256 + xi[j]];
        }
        for (int f = 0; f < K; f++) {
            float sum_e = 0.0f, sum_e2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float e = emb[((long)j * 256 + xi[j]) * K + f];
                sum_e += e;
                sum_e2 += e * e;
            }
            logit += 0.5f * (sum_e * sum_e - sum_e2);
        }
        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```