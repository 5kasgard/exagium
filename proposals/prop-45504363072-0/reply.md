```hypothesis
Factorization Machine (FM) with 8-dimensional latent factors captures pairwise feature interactions (e.g., education×occupation, marital-status×relationship) known to be predictive in Adult, while adding only ~32k parameters. It should beat label-linear (0.8524) and approach label-gbdt (0.8664) at near-linear inference cost (~1.5×). Failure: dev accuracy ≤ 0.853 or inference cost > 2× label-linear.
```
```python
"""Factorization Machine (degree-2) for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encoding: ordinal z-scored, categorical one-hot over training values (255=unseen).
FM: w0 + w·x + 0.5*sum_f((sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2).
Adam lr=1e-2, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports per-byte lookup tables: magic 0x464D3231, d, f, C; float32 lin[d][256], fact[d][256][f], w0."""
import json, math, struct, sys

import numpy as np

F = 8
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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Build encoders and feature matrix Z (n x k)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])  # (256, 1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)  # (256, n_cat)

    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    # FM parameters
    w0 = np.float32(0.0)
    w = np.zeros(k, np.float32)
    V = (rng.uniform(-1, 1, (k, F)) * math.sqrt(6.0 / k)).astype(np.float32)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_w0, v_w0 = 0.0, 0.0
    t = 0

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            bsz = len(yy)

            # FM forward
            linear = z @ w + w0
            zV = z @ V  # (bsz, F)
            inter = 0.5 * np.sum(zV * zV - (z * z) @ (V * V), axis=1)
            logit = linear + inter
            p = sigmoid(logit)

            # Gradients (logistic loss)
            grad = (p - yy).astype(np.float32)  # (bsz,)
            g_w0 = grad.mean()
            g_w = (z.T @ grad) / bsz + L2 * w
            # d/dV_f = sum_i grad_i * (z_i,f * zV_i - z_i^2 * V_f)
            g_V = (z.T @ (grad[:, None] * zV) - (z * z).T @ (grad[:, None] * V)) / bsz + L2 * V

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            # Update w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (math.sqrt(v_w0 / (1 - beta2 ** t)) + eps)

            # Update w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            # Update V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Build per-byte lookup tables
    lin_table = np.zeros((X.shape[1], 256), np.float32)
    fact_table = np.zeros((X.shape[1], 256, F), np.float32)
    off = 0
    for j, E in enumerate(enc):
        k_j = E.shape[1]
        w_j = w[off:off + k_j]
        V_j = V[off:off + k_j]
        lin_table[j] = E @ w_j
        fact_table[j] = E @ V_j
        off += k_j

    # Export
    magic, d, f, C = 0x464D3231, X.shape[1], F, 2
    header = struct.pack("<4i", magic, d, f, C)
    body = lin_table.astype("<f4").tobytes() + fact_table.astype("<f4").tobytes() + np.array([w0], "<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree-2) inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index (0/1). */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], f = hd[2], C = hd[3];
    if (C != 2 || f < 1 || f > 64 || d < 1 || d > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * (1 + f) + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *lin = (const float *)(hd + 4);
    const float *fact = lin + (long)d * 256;
    const float w0 = fact[(long)d * 256 * f];

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * f);
    float *sum_v2 = malloc(sizeof(float) * f);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int u = 0; u < f; u++) { sum_v[u] = 0.0f; sum_v2[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_row = lin + ((long)j * 256 + v);
            const float *fact_row = fact + ((long)j * 256 * f + (long)v * f);
            score += *lin_row;
            for (int u = 0; u < f; u++) {
                float val = fact_row[u];
                sum_v[u] += val;
                sum_v2[u] += val * val;
            }
        }
        for (int u = 0; u < f; u++) {
            score += 0.5f * (sum_v[u] * sum_v[u] - sum_v2[u]);
        }
        unsigned char best = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```