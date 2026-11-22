```hypothesis
Family: label-fm (factorization machine for binary classification). It adds pairwise feature interactions via low-rank embeddings (k=4) to a linear model, capturing non-linearities in the Adult dataset (e.g., education × occupation) while keeping inference fast. The model is trained with binary cross-entropy and Adam, deterministic seed. If dev accuracy does not exceed label-linear (0.8524) or cost exceeds ~1000 Ir/item, the approach fails.
```
```python
"""Factorization Machine for binary label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights format: magic 0x464D0001, int32 d, k, C; float32 bias, w[d*256], v[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100


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
    assert C == 2, "FM binary expects 2 classes"

    rng = np.random.default_rng(SEED)
    # parameters
    bias = np.float32(0.0)
    w = (rng.normal(0, 0.01, (d, 256)) * math.sqrt(6.0 / 256)).astype(np.float32)
    v = (rng.normal(0, 0.01, (d, 256, K)) * math.sqrt(6.0 / 256)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,) int64
            B = xb.shape[0]

            # Forward: compute logit = bias + sum_i w[i, xi] + sum_{i<j} <v_i, v_j>
            linear = np.zeros(B, np.float32)
            for i in range(d):
                linear += w[i, xb[:, i]]
            inter = np.zeros(B, np.float32)
            for i in range(d):
                vi = v[i, xb[:, i]]      # (B, K)
                for j in range(i + 1, d):
                    vj = v[j, xb[:, j]]
                    inter += np.sum(vi * vj, axis=1)
            logit = bias + linear + inter

            # Binary cross-entropy gradient: dL/dlogit = sigmoid(logit) - y
            sig = 1.0 / (1.0 + np.exp(-logit))
            grad_logit = (sig - yb).astype(np.float32) / B

            # Backward
            grad_bias = np.sum(grad_logit)
            grad_w = np.zeros_like(w)
            grad_v = np.zeros_like(v)
            for i in range(d):
                # linear part
                np.add.at(grad_w[i], xb[:, i], grad_logit)
                # interaction part
                vi = v[i, xb[:, i]]      # (B, K)
                for j in range(i + 1, d):
                    vj = v[j, xb[:, j]]
                    g = grad_logit[:, None] * vj   # (B, K)
                    np.add.at(grad_v[i], xb[:, i], g)
                    g2 = grad_logit[:, None] * vi
                    np.add.at(grad_v[j], xb[:, j], g2)

            # L2 regularization
            grad_w += L2 * w
            grad_v += L2 * v

            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * grad_bias
            v_bias = 0.999 * v_bias + 0.001 * grad_bias * grad_bias
            bias -= LR * (m_bias / (1 - 0.9 ** t)) / (np.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)
            # w
            m_w = 0.9 * m_w + 0.1 * grad_w
            v_w = 0.999 * v_w + 0.001 * grad_w * grad_w
            w -= LR * (m_w / (1 - 0.9 ** t)) / (np.sqrt(v_w / (1 - 0.999 ** t)) + 1e-8)
            # v
            m_v = 0.9 * m_v + 0.1 * grad_v
            v_v = 0.999 * v_v + 0.001 * grad_v * grad_v
            v -= LR * (m_v / (1 - 0.9 ** t)) / (np.sqrt(v_v / (1 - 0.999 ** t)) + 1e-8)

    # Write weights.bin
    # magic, d, K, C
    out = struct.pack("<4i", 0x464D0001, d, K, C)
    out += np.array([bias], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine binary inference: ./infer weights.bin in.rec N
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K != 4) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *w = p + 1;
    const float *v = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = 0.0f;
        for (int j = 0; j < d; j++) linear += w[(long)j * 256 + xi[j]];

        float inter = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *vj = v + ((long)j * 256 + xi[j]) * K;
            for (int l = j + 1; l < d; l++) {
                const float *vl = v + ((long)l * 256 + xi[l]) * K;
                for (int f = 0; f < K; f++) inter += vj[f] * vl[f];
            }
        }
        float logit = bias + linear + inter;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```