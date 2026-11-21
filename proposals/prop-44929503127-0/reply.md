```hypothesis
Family: label-fm (Factorization Machine with k=4 embeddings). Treats each of the 14 byte features as a 256-category categorical variable, learning a linear weight and a 4-dim embedding per (feature,value) pair. Captures pairwise feature interactions at O(d*k) inference cost (d=14, k=4), far cheaper than the label-linear's O(d*256) table lookups. Expects to beat label-linear's 0.8524 accuracy on Adult (where feature interactions matter) while cutting instructions/item by ~10x. Failure mode: if interactions are negligible or embeddings overfit, accuracy drops below linear and we fall back to linear baseline.
```
```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Each of the d input bytes (0-255) gets a linear weight and a k-dim embedding. Logit = bias + sum linear + 0.5*sum_f((sum emb_f)^2 - sum emb_f^2).
Adam lr=1e-2, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)), seed=1. Deterministic, single thread.
Weights: magic 0x464D5F31, int32 d,k,C; float32 bias, linear[d*256], emb[d*256*k]. C=1 (binary logit)."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


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
    d = card["input"]["bytes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Model: bias (1), linear (d*256), emb (d*256, K)
    linear = rng.normal(0, 0.01, size=(d * 256,)).astype(np.float32)
    emb = rng.normal(0, 0.01, size=(d * 256, K)).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_b, v_b = np.float32(0.0), np.float32(0.0)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx].astype(np.float32)  # (B,) 0/1
            B = len(idx)

            # Forward
            logit = np.full(B, bias, dtype=np.float32)
            sum_emb = np.zeros((B, K), dtype=np.float32)
            sum_emb_sq = np.zeros((B, K), dtype=np.float32)
            for j in range(d):
                v = xb[:, j]                     # (B,) uint8
                base = j * 256
                lin_w = linear[base:base + 256]  # (256,)
                emb_w = emb[base:base + 256]     # (256, K)
                logit += lin_w[v]
                sum_emb += emb_w[v]
                sum_emb_sq += emb_w[v] * emb_w[v]
            logit += 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)

            # Logistic loss gradient: sigmoid(logit) - y
            sig = 1.0 / (1.0 + np.exp(-logit))
            grad = sig - yb                     # (B,)

            # Backward
            db = grad.mean()
            dlinear = np.zeros_like(linear)
            demb = np.zeros_like(emb)

            for j in range(d):
                v = xb[:, j]
                base = j * 256
                # linear gradient
                np.add.at(dlinear, base + v, grad)
                # embedding gradient: for each f, d/d emb = (sum_emb_f - emb_f) * grad
                # sum_emb is (B, K), emb_w[v] is (B, K)
                g = grad[:, None] * (sum_emb - emb_w[v])  # (B, K)
                np.add.at(demb, base + v, g)

            dlinear /= B
            demb /= B
            db /= B

            # L2 regularization
            dlinear += L2 * linear
            demb += L2 * emb
            db += L2 * bias

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            m_lin = beta1 * m_lin + (1 - beta1) * dlinear
            v_lin = beta2 * v_lin + (1 - beta2) * dlinear * dlinear
            m_emb = beta1 * m_emb + (1 - beta1) * demb
            v_emb = beta2 * v_emb + (1 - beta2) * demb * demb
            m_b = beta1 * m_b + (1 - beta1) * db
            v_b = beta2 * v_b + (1 - beta2) * db * db

            m_lin_hat = m_lin / (1 - beta1 ** t)
            v_lin_hat = v_lin / (1 - beta2 ** t)
            m_emb_hat = m_emb / (1 - beta1 ** t)
            v_emb_hat = v_emb / (1 - beta2 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)

            linear -= LR * m_lin_hat / (np.sqrt(v_lin_hat) + eps)
            emb -= LR * m_emb_hat / (np.sqrt(v_emb_hat) + eps)
            bias -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)

    # Pack weights: magic, d, K, C=1; bias; linear; emb
    out = struct.pack("<4i", 0x464D5F31, d, K, 1)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += linear.astype(np.float32).tobytes()
    out += emb.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0/1).
 * Weights: magic 0x464D5F31, int32 d,k,C; float32 bias, linear[d*256], emb[d*256*k]. */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D5F31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K != 4 || C != 1) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_emb[4], sum_emb_sq[4];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias[0];
        for (int f = 0; f < K; f++) { sum_emb[f] = 0.0f; sum_emb_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            long idx = (long)j * 256 + v;
            logit += linear[idx];
            const float *e = emb + (long)idx * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                sum_emb[f] += val;
                sum_emb_sq[f] += val * val;
            }
        }
        for (int f = 0; f < K; f++) logit += 0.5f * (sum_emb[f] * sum_emb[f] - sum_emb_sq[f]);

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```