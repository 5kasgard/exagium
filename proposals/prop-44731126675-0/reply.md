```hypothesis
Family: label-fm (Factorization Machine).
Why: Adult income prediction is known to benefit from feature interactions (e.g., education×occupation, marital-status×relationship). A factorization machine captures all pairwise interactions via low-rank embeddings (K=4) with only ~72KB weights—far fewer than an MLP. It should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at lower inference cost (no hidden layer, just embedding lookups and dot products).
Failure: If dev accuracy ≤ 0.8524 or inference cost exceeds label-linear significantly.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin (deterministic: seed 1, single thread)
Treats all 14 input bytes as categorical (256 values each). Learns w0, w[d][256], V[d][256][K].
Adam lr=0.01, L2=1e-4, batch=64, epochs=50. Exports: magic 0x464D4C31, d, K, C; float32 w0, w, V."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 50


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

    w0 = np.float32(0.0)
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yi = X[idx], y[idx]
            B = len(yi)

            # forward
            linear = w0 + np.sum(w[np.arange(d), xi.T], axis=0)
            V_sum = np.zeros((B, K), np.float32)
            V_sq_sum = np.zeros((B, K), np.float32)
            for j in range(d):
                v_j = V[j, xi[:, j]]
                V_sum += v_j
                V_sq_sum += v_j * v_j
            pairwise = 0.5 * np.sum(V_sum * V_sum - V_sq_sum, axis=1)
            logits = linear + pairwise

            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logits = (p - yi).astype(np.float32)

            # gradients
            gw0 = np.mean(grad_logits)
            gw = np.zeros_like(w)
            gV = np.zeros_like(V)
            for j in range(d):
                xj = xi[:, j]
                np.add.at(gw[j], xj, grad_logits)
                Vj_xj = V[j, xj]
                V_sum_minus_j = V_sum - Vj_xj
                grad_V = grad_logits[:, None] * V_sum_minus_j
                np.add.at(gV[j], xj, grad_V)
            gw /= B; gV /= B
            gw += L2 * w; gV += L2 * V

            # Adam update
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1**t)) / (np.sqrt(v_w0 / (1 - beta2**t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= LR * (m_w / (1 - beta1**t)) / (np.sqrt(v_w / (1 - beta2**t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)

    # save: magic, d, K, C, w0, w[d][256], V[d][256][K]
    magic = 0x464D4C31
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index 0/1).
 * Weights: magic 0x464D4C31, d, K, C, float32 w0, w[d][256], V[d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(wb + 16);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *V_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        float sq_sum = 0.0f;
        for (int j = 0; j < K; j++) V_sum[j] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += w[(long)j * 256 + v];
            const float *Vjv = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) {
                float val = Vjv[kk];
                V_sum[kk] += val;
                sq_sum += val * val;
            }
        }

        float pairwise = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float s = V_sum[kk];
            pairwise += s * s;
        }
        pairwise = 0.5f * (pairwise - sq_sum);

        float score = linear + pairwise;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```