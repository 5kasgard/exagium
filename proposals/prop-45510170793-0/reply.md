```hypothesis
Factorization Machine (FM) with binary logistic loss: per-feature-value linear weights plus low-rank (k=4) interaction embeddings. Captures pairwise feature interactions (e.g., education×occupation) at inference cost O(d·k) ≈ 56 FMA/item, only ~2× label-linear. Adult is known to benefit from categorical interactions. If dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ 2325 (label-mlp16), the model fails.
```

```python
"""Factorization Machine for tabular label tasks (binary).
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x464D3130, d, k, C=1; float32 bias, linear[d*256], emb[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.float32(0.0)
    linear = rng.normal(0, 0.01, size=(d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            lin_part = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (B,)
            emb_sel = emb[np.arange(d), xb.T]  # (d, B, K)
            sum_emb = np.sum(emb_sel, axis=0)          # (B, K)
            sum_emb_sq = np.sum(emb_sel ** 2, axis=0)  # (B, K)
            inter = 0.5 * np.sum(sum_emb ** 2 - sum_emb_sq, axis=1)  # (B,)
            logits = lin_part + inter
            p = sigmoid(logits)

            # Gradient wrt logits
            grad_logits = (p - yb).astype(np.float32)  # (B,)

            # Gradients
            g_bias = np.mean(grad_logits)
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            # Linear gradient
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_lin[j], vals, grad_logits)
            g_lin /= B
            g_lin += L2 * linear

            # Embedding gradient
            # d(inter)/d(emb_{j,v,f}) = grad_logits * (sum_emb[f] - emb_{j,v,f})
            for j in range(d):
                vals = xb[:, j]
                for f in range(K):
                    # sum_emb[:, f] is (B,)
                    # emb_sel[j, :, f] is (B,)
                    diff = sum_emb[:, f] - emb_sel[j, :, f]
                    np.add.at(g_emb[j, :, f], vals, grad_logits * diff)
            g_emb /= B
            g_emb += L2 * emb

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

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

    # Write weights.bin
    magic = 0x464D3130  # 'FM10'
    header = struct.pack("<4i", magic, d, K, 1)
    body = struct.pack("<f", bias) + linear.astype("<f4").tobytes() + emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 1) { fprintf(stderr, "expected binary model (C=1)\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = bias;
        for (int f = 0; f < K; f++) { sum_emb[f] = 0.0f; sum_emb_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            logit += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = e[f];
                sum_emb[f] += val;
                sum_emb_sq[f] += val * val;
            }
        }
        for (int f = 0; f < K; f++) {
            logit += 0.5f * (sum_emb[f] * sum_emb[f] - sum_emb_sq[f]);
        }
        unsigned char pred = (logit >= 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```