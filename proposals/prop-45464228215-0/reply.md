```hypothesis
Factorization Machine (label-fm) with rank-4 embeddings captures pairwise feature interactions (e.g., education×occupation, marital-status×relationship) critical for Adult income prediction. The binary FM computes a single logit: linear bias + per-byte linear weights + 0.5*(||Σemb||² - Σ||emb||²). This adds only ~57 KB weights and ~2× linear inference cost. If dev accuracy ≤ 0.8524 (label-linear), interactions aren't helping and the family should be discarded.
```
```python
"""Factorization Machine for binary label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports: magic 0x464D3130, d, k, C=2;
float32 linear_table[d][256], linear_bias, emb_table[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 4
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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert C == 2, "FM binary only"
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: linear[d, 256], bias, emb[d, 256, K]
    linear = np.zeros((d, 256), np.float32)
    bias = np.float32(0.0)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward
            lin_part = bias + np.sum(linear[xb, np.arange(d)], axis=1)  # [B]
            emb_sum = np.sum(emb[xb, np.arange(d)], axis=1)             # [B, K]
            emb_sq_sum = np.sum(emb[xb, np.arange(d)] ** 2, axis=1)     # [B, K]
            interaction = 0.5 * np.sum(emb_sum ** 2 - emb_sq_sum, axis=1)  # [B]
            logit = lin_part + interaction
            p = 1.0 / (1.0 + np.exp(-logit))
            # BCE gradient: dL/dlogit = p - y
            grad_logit = (p - yb).astype(np.float32)  # [B]

            # Gradients
            # linear: for each sample, each feature j, value v, add grad_logit[b]
            # We'll accumulate via scatter-add
            grad_linear = np.zeros_like(linear)
            for j in range(d):
                np.add.at(grad_linear[j], xb[:, j], grad_logit)
            grad_bias = np.float32(np.sum(grad_logit))

            # emb: grad = grad_logit[:, None] * (emb_sum - emb[xb])
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                v = xb[:, j]
                diff = emb_sum - emb[v, np.arange(bsz)]  # [B, K] - [B, K] via advanced indexing? emb[v] is [B, K]
                # emb[v] shape: [B, K], emb_sum: [B, K]
                g = grad_logit[:, None] * (emb_sum - emb[v])
                np.add.at(grad_emb[j], v, g)

            # L2 regularization
            grad_linear += L2 * linear
            grad_bias += L2 * bias
            grad_emb += L2 * emb

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * grad_linear
            v_lin = beta2 * v_lin + (1 - beta2) * grad_linear * grad_linear
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1 - beta2) * grad_bias * grad_bias
            bias -= LR * (m_bias / (1 - beta1 ** t)) / (np.sqrt(v_bias / (1 - beta2 ** t)) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * grad_emb * grad_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    # magic, d, K, C
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += struct.pack("<f", bias)
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (binary) for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 + 1 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float bias = linear[(long)d * 256];
    const float *emb = linear + (long)d * 256 + 1;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = (float *)calloc(K, sizeof(float));
    float *emb_sq_sum = (float *)calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin = bias;
        for (int u = 0; u < K; u++) { emb_sum[u] = 0.0f; emb_sq_sum[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = e[u];
                emb_sum[u] += val;
                emb_sq_sum[u] += val * val;
            }
        }

        float inter = 0.0f;
        for (int u = 0; u < K; u++) {
            float s = emb_sum[u];
            inter += s * s - emb_sq_sum[u];
        }
        inter *= 0.5f;
        float logit = lin + inter;
        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```