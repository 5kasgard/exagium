```hypothesis
Family: label-fm (factorization machine). It models pairwise feature interactions via low-rank embeddings, which are crucial for tabular data with mixed categorical/ordinal columns like Adult. With embedding size k=8, it should surpass label-linear (0.8524) by capturing interactions while remaining fast (O(d*k) inference). If dev accuracy ≤ 0.8524 or inference cost exceeds label-linear significantly, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Exports: magic 0x464D31, d, k, C; float32 w0, linear[d*256], emb[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 10


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
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    w0 = np.float32(0.0)
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)

    # Adam state
    m_w0 = v_w0 = 0.0
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    n = X.shape[0]
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = xb.shape[0]

            # forward
            lin_scores = w0 + np.sum(linear[np.arange(d), xb.T], axis=0)
            emb_sum = np.sum(emb[np.arange(d), xb.T], axis=0)
            emb_sq_sum = np.sum(emb[np.arange(d), xb.T] ** 2, axis=0)
            interaction = 0.5 * np.sum(emb_sum ** 2 - emb_sq_sum, axis=1)
            scores = lin_scores + interaction

            probs = 1.0 / (1.0 + np.exp(-scores))
            grad_scores = (probs - yb).astype(np.float32)

            # backward
            grad_linear = np.zeros_like(linear)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(grad_linear[j], vals, grad_scores)
            grad_linear += L2 * linear

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                vals = xb[:, j]
                emb_j_vals = emb[j, vals]
                diff = emb_sum - emb_j_vals
                grad_emb_j = grad_scores[:, None] * diff
                for b in range(B):
                    grad_emb[j, vals[b]] += grad_emb_j[b]
            grad_emb += L2 * emb

            grad_w0 = np.sum(grad_scores) + L2 * w0

            # Adam update
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * (grad_w0 * grad_w0)
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_lin = beta1 * m_lin + (1 - beta1) * grad_linear
            v_lin = beta2 * v_lin + (1 - beta2) * (grad_linear * grad_linear)
            linear -= LR * (m_lin / (1 - beta1 ** t)) / (np.sqrt(v_lin / (1 - beta2 ** t)) + eps)

            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * (grad_emb * grad_emb)
            emb -= LR * (m_emb / (1 - beta1 ** t)) / (np.sqrt(v_emb / (1 - beta2 ** t)) + eps)

    # write weights.bin
    magic = 0x464D31  # 'FM1'
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *linear = w0 + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_emb[32], sum_emb_sq[32];  // K <= 32

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = *w0;
        for (int u = 0; u < K; u++) { sum_emb[u] = 0.0f; sum_emb_sq[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            score += linear[(long)j * 256 + val];
            const float *emb_j = emb + ((long)j * 256 + val) * K;
            for (int u = 0; u < K; u++) {
                float v = emb_j[u];
                sum_emb[u] += v;
                sum_emb_sq[u] += v * v;
            }
        }
        for (int u = 0; u < K; u++) {
            score += 0.5f * (sum_emb[u] * sum_emb[u] - sum_emb_sq[u]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```