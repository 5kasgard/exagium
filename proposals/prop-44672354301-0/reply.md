```hypothesis
Family: label-fm (Factorization Machine). Adult has 14 mixed ordinal/categorical features with many pairwise interactions (e.g., education×occupation, marital×sex). FMs capture all pairwise interactions via low-rank embeddings in O(d·k) time, stronger than linear (label-linear) and faster than MLP/GBDT. With k=8 embeddings per feature-value, ~130 KB weights, it should exceed 0.8524 accuracy at ~1000 Ir/item. Failure: dev accuracy ≤ 0.8524 or Ir/item > 2000.
```
```python
"""Factorization Machine for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Embedding dim k=8, Adam lr=0.01, L2=1e-4, batch=256."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 200000, 50


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

    rng = np.random.default_rng(SEED)
    # Parameters: w0, w[d,256], V[d,256,k]
    w0 = np.float32(0.0)
    w = (rng.normal(0, 0.01, (d, 256)) * 0).astype(np.float32)  # zero init for linear
    V = (rng.normal(0, 0.01, (d, 256, K)) * np.sqrt(1.0 / K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]          # [B, d]
            yi = y[idx]          # [B]
            B = len(yi)

            # Forward: score = w0 + sum_j w[j,xi] + 0.5*(||sum V||^2 - sum ||V||^2)
            # Linear part
            lin = w0 + np.sum(w[np.arange(d)[:, None], xi.T], axis=0)  # [B]
            # Interaction part via FM trick
            V_xi = V[np.arange(d)[:, None], xi.T]  # [d, B, K]
            sum_V = np.sum(V_xi, axis=0)           # [B, K]
            sum_sq_V = np.sum(V_xi * V_xi, axis=(0, 2))  # [B]
            inter = 0.5 * (np.sum(sum_V * sum_V, axis=1) - sum_sq_V)  # [B]
            score = lin + inter  # [B]

            # Binary cross-entropy gradient: sigmoid(score) - y
            p = 1.0 / (1.0 + np.exp(-score))
            g = (p - yi).astype(np.float32)  # [B]

            # Gradients
            # w0
            gw0 = np.mean(g)
            # w: [d, 256]
            gw = np.zeros_like(w)
            for j in range(d):
                np.add.at(gw[j], xi[:, j], g)
            gw /= B
            gw += L2 * w
            # V: [d, 256, k]
            gV = np.zeros_like(V)
            for j in range(d):
                vj = V[j, xi[:, j]]          # [B, K]
                sum_V_ = sum_V - vj          # [B, K]
                # d inter / d V_j = g * sum_{l!=j} V_l = g * (sum_V - V_j)
                gj = g[:, None] * sum_V_     # [B, K]
                np.add.at(gV[j], xi[:, j], gj)
            gV /= B
            gV += L2 * V

            t += 1
            # Adam updates
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (np.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV * gV
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Write weights: magic, d, k, C, w0, w[d*256], V[d*256*k]
    magic = 0x464D31  # "FM1"
    out = struct.pack("<4if", magic, d, K, C, w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (0/1). FM with k-dim embeddings. */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x464D31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t d = *(int32_t *)(wb + 4);
    int32_t K = *(int32_t *)(wb + 8);
    int32_t C = *(int32_t *)(wb + 12);
    float w0 = *(float *)(wb + 16);
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + (long)d * 256 * 4 + (long)d * 256 * K * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *w = (const float *)(wb + 20);
    const float *V = (const float *)(wb + 20 + (long)d * 256 * 4);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        memset(sum_V, 0, sizeof(float) * K);
        float sum_sq_V = 0.0f;

        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            score += w[(long)j * 256 + v];
            const float *vj = V + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = vj[u];
                sum_V[u] += val;
                sum_sq_V += val * val;
            }
        }
        float sum_V_sq = 0.0f;
        for (int u = 0; u < K; u++) sum_V_sq += sum_V[u] * sum_V[u];
        score += 0.5f * (sum_V_sq - sum_sq_V);

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```