```hypothesis
Family: label-fm (Factorization Machine with per-field embeddings).
Why: Adult has 14 fields with categorical/ordinal values; FM captures pairwise field interactions via low-rank embeddings, which logistic regression misses, while staying lightweight (k=8 → ~32k params). It should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at similar or lower inference cost.
Failure: If dev accuracy ≤ 0.8524 or instructions/item exceed label-linear by >20%.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic(0x464D3130), d, k, reserved, bias, linear[d*256], emb[d*256*k] (all float32 little-endian)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
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
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    n = X_bytes.shape[0]

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

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_bytes[idx]
            yy = y[idx]
            bsz = len(idx)

            # Forward
            logit = bias + np.sum(linear[np.arange(d)[:, None], xb.T], axis=0)  # (bsz,)
            sum_emb = np.zeros((bsz, K), dtype=np.float32)
            sum_sq_emb = np.zeros((bsz, K), dtype=np.float32)
            for j in range(d):
                e = emb[j, xb[:, j]]  # (bsz, K)
                sum_emb += e
                sum_sq_emb += e * e
            inter = 0.5 * np.sum(sum_emb * sum_emb - sum_sq_emb, axis=1)
            logit += inter

            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yy).astype(np.float32)  # (bsz,)

            # Backward
            g_bias = np.mean(grad)
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            for j in range(d):
                vals = xb[:, j]
                g_lin[j, vals] += grad
                # embedding gradient: grad * (sum_emb - emb_j)
                for f in range(K):
                    g_emb[j, vals, f] += grad * (sum_emb[:, f] - emb[j, vals, f])

            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * emb

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

    # Write weights
    magic = 0x464D3130  # "FM10"
    reserved = 0
    out = struct.pack("<5i", magic, d, K, reserved, 0)  # last 0 placeholder for bias (float)
    out = out[:-4] + struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (field values 0..255). Output: one byte class (0 or 1). */
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
    if (nw < 20 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2];
    if (d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad dims\n"); return 3; }
    float bias = *(float *)(hd + 5);
    const float *linear = (const float *)(hd + 6);
    const float *emb = linear + (long)d * 256;
    long expected = 20 + 4L * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(K * sizeof(float));
    float *sum_sq_emb = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = bias;
        memset(sum_emb, 0, K * sizeof(float));
        memset(sum_sq_emb, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            int val = xi[j];
            logit += linear[(long)j * 256 + val];
            const float *e = emb + ((long)j * 256 + val) * K;
            for (int f = 0; f < K; f++) {
                float ev = e[f];
                sum_emb[f] += ev;
                sum_sq_emb[f] += ev * ev;
            }
        }
        for (int f = 0; f < K; f++) {
            logit += 0.5f * (sum_emb[f] * sum_emb[f] - sum_sq_emb[f]);
        }
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```