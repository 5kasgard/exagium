```hypothesis
Family: label-fm (Factorization Machine with per-column embeddings).
Why: Adult has mixed categorical/ordinal features; FM captures pairwise interactions cheaply via low-rank embeddings (k=4).
It should beat linear (0.8524) by modeling interactions, while staying faster than MLP16 (2325 Ir/item) due to O(d*k) inference.
Failure: If validation accuracy on tuning split does not exceed linear by >0.002, the family is not worthwhile.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: each of the d input bytes directly indexes per-column lookup tables.
Model: linear term + FM interaction term with embedding dim k=4.
Exported as float32: magic 0x464D4C31, d, k, C; bias[C]; linear[d][256][C]; emb[d][256][k]; V[k][C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, no tuning)
K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

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
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    bias = np.zeros(C, np.float32)
    linear = (rng.normal(0, 0.01, (d, 256, C)) * math.sqrt(2.0 / (256 * C))).astype(np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K)) * math.sqrt(2.0 / (256 * K))).astype(np.float32)
    V = (rng.normal(0, 0.01, (K, C)) * math.sqrt(2.0 / (K * C))).astype(np.float32)

    params = [bias, linear, emb, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

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
            lin_scores = bias + np.sum(linear[np.arange(d), xb.T], axis=0)  # (B, C)
            e = emb[np.arange(d), xb.T]  # (d, B, K)
            e = np.transpose(e, (1, 0, 2))  # (B, d, K)
            sum_v = e.sum(axis=1)           # (B, K)
            sum_sq = (e * e).sum(axis=1)    # (B, K)
            inter_vec = 0.5 * (sum_v * sum_v - sum_sq)  # (B, K)
            inter_scores = inter_vec @ V    # (B, C)
            logits = lin_scores + inter_scores

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)  # (B, C)

            # Gradients
            g_bias = grad_logits.sum(axis=0)
            g_linear = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)
            g_V = np.zeros_like(V)

            # Linear gradient
            for j in range(d):
                np.add.at(g_linear[j], xb[:, j], grad_logits)

            # V gradient
            g_V = inter_vec.T @ grad_logits

            # Interaction vector gradient
            g_inter_vec = grad_logits @ V.T  # (B, K)

            # Backprop through interaction vector
            g_sum_v = g_inter_vec * sum_v
            g_sum_sq = g_inter_vec * (-0.5)

            # Backprop through sum_v and sum_sq to embeddings
            g_e = g_sum_v[:, None, :] + 2.0 * e * g_sum_sq[:, None, :]  # (B, d, K)

            # Accumulate embedding gradients
            for j in range(d):
                np.add.at(g_emb[j], xb[:, j], g_e[:, j, :])

            # L2 regularization
            g_bias += 0  # no L2 on bias
            g_linear += 2 * L2 * linear
            g_emb += 2 * L2 * emb
            g_V += 2 * L2 * V

            grads = [g_bias, g_linear, g_emb, g_V]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index.
 * Weights format: magic 0x464D4C31, int32 d, k, C; float32 bias[C]; linear[d][256][C]; emb[d][256][k]; V[k][C]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    size_t expected = 16
        + (size_t)C * 4
        + (size_t)d * 256 * C * 4
        + (size_t)d * 256 * k * 4
        + (size_t)k * C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + (size_t)d * 256 * C;
    const float *V = emb + (size_t)d * 256 * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_sq = malloc(sizeof(float) * k);
    float *inter_vec = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        // Linear term
        memcpy(acc, bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *w = linear + ((size_t)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc[c] += w[c];
        }

        // Embeddings
        memset(sum_v, 0, k * sizeof(float));
        memset(sum_sq, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * k;
            for (int u = 0; u < k; u++) {
                float val = e[u];
                sum_v[u] += val;
                sum_sq[u] += val * val;
            }
        }

        // Interaction vector
        for (int u = 0; u < k; u++) {
            inter_vec[u] = 0.5f * (sum_v[u] * sum_v[u] - sum_sq[u]);
        }

        // Interaction scores
        for (int c = 0; c < C; c++) {
            float s = 0.0f;
            for (int u = 0; u < k; u++) s += inter_vec[u] * V[(size_t)u * C + c];
            acc[c] += s;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```