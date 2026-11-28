```hypothesis
Family: label-fm (factorization machine with 8-d embeddings).  
Why: Adult has strong pairwise interactions (e.g., education × occupation, marital-status × sex) that a linear model misses. A factorization machine captures all pairwise interactions via low-rank embeddings, adding only 14*256*8 ≈ 29k parameters. Training with Adam on the full data is fast and deterministic.  
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2000 Ir/item, the family does not meet its goal.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: seed 1, single thread, numpy only.
Encoding: each of the d=14 bytes is a field with 256 possible values.
Model: logit = w0 + Σ_j w[j, x_j] + Σ_{j1<j2} <V[j1, x_j1], V[j2, x_j2]>
Second order computed via 0.5*( (Σ V_j)^2 - Σ V_j^2 ).
Binary classification: output class 1 if logit>0 else 0.
Exported weights: int32 magic 0x464D4C31, d, k, C; float32 w0, w[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

K = 8                 # embedding dimension
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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)   # 0 or 1
    d = X.shape[1]                          # 14
    C = card["classes"]                     # 2
    assert C == 2, "label-fm expects binary classification"
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    w0 = np.float32(0.0)
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (B, d)
            yb = y[idx]                 # (B,)
            B = xb.shape[0]

            # --- Forward ---
            # first order
            first = w0 + np.sum(w[np.arange(d), xb.T], axis=0)   # (B,)

            # second order: 0.5 * ( (sum_j V_j)^2 - sum_j V_j^2 )
            V_sel = V[np.arange(d), xb.T]       # (d, B, K)
            sum_V = np.sum(V_sel, axis=0)       # (B, K)
            sum_sq_V = np.sum(V_sel * V_sel, axis=0)  # (B, K)
            second = 0.5 * np.sum(sum_V * sum_V - sum_sq_V, axis=1)   # (B,)

            logit = first + second              # (B,)

            # logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logit))    # (B,)
            grad_logit = (p - yb) / B           # (B,)

            # --- Backward ---
            # w0
            g_w0 = np.sum(grad_logit)
            # w
            g_w = np.zeros_like(w)
            np.add.at(g_w, (np.arange(d)[:, None], xb.T), grad_logit)   # (d, 256)
            g_w += L2 * w
            # V
            g_V = np.zeros_like(V)
            # For each field j: contribution to V[j, xb[:, j]] is grad_logit * (sum_V - V_sel[j])
            for j in range(d):
                # sum_V - V_sel[j] has shape (B, K)
                diff = sum_V - V_sel[j]
                # grad_logit[:, None] * diff  -> (B, K)
                np.add.at(g_V[j], xb[:, j], grad_logit[:, None] * diff)
            g_V += L2 * V

            # --- Adam update ---
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_hat = m_w0 / (1 - beta1 ** t)
            v_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_hat = m_w / (1 - beta1 ** t)
            v_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_hat = m_V / (1 - beta1 ** t)
            v_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # --- Export weights ---
    # Format: magic, d, K, C; w0; w[d*256]; V[d*256*K]
    magic = 0x464D4C31  # 'FML1'
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([w0], dtype="<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=14 for adult); output is one byte class (0 or 1).
 * Weights format: int32 magic(0x464D4C31), d, K, C; float32 w0, w[d*256], V[d*256*K].
 * Prediction: logit = w0 + sum_j w[j, x_j] + 0.5*( (sum_j V_j)^2 - sum_j V_j^2 ). Class 1 if logit>0 else 0. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = p + 1 + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);
    float *sq_V  = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float logit = w0;
        // first order
        for (int j = 0; j < d; j++) logit += w[(long)j * 256 + xi[j]];

        // second order
        memset(sum_V, 0, K * sizeof(float));
        memset(sq_V, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * K;
            for (int u = 0; u < K; u++) {
                float val = v[u];
                sum_V[u] += val;
                sq_V[u]  += val * val;
            }
        }
        float second = 0.0f;
        for (int u = 0; u < K; u++) {
            float s = sum_V[u];
            second += s * s - sq_V[u];
        }
        logit += 0.5f * second;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_V); free(sq_V);
    return 0;
}
```