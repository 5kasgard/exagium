```hypothesis
Family: label-fm (factorization machine with class-specific pairwise interactions).
Why: Adult has 14 mixed categorical/ordinal features; FM captures pairwise interactions via low-rank embeddings (k=8) which often beats linear models on tabular data while staying fast (O(C·d²·k) ≈ 2·14²·8 ops/item).
Failure: If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2× label-linear.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x464D5F31, d, C, k; float32 w0[C], w[d][256][C], V[C][d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 20
K = 8  # embedding dimension


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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # tuning split (last 20%) - not used for hyperparameters here (fixed K)
    rng = np.random.default_rng(SEED)

    # parameters
    w0 = np.zeros(C, dtype=np.float32)
    w = np.zeros((d, 256, C), dtype=np.float32)
    V = rng.normal(0, 0.01, (C, d, 256, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.zeros_like(w0), np.zeros_like(w0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # --- forward ---
            # linear part per class: (B, C)
            linear = np.zeros((B, C), dtype=np.float32)
            for c in range(C):
                for i in range(d):
                    linear[:, c] += w[i, xb[:, i], c]

            # interaction part per class: (B, C)
            inter = np.zeros((B, C), dtype=np.float32)
            for c in range(C):
                S = np.zeros((B, K), dtype=np.float32)
                Q = np.zeros((B, K), dtype=np.float32)
                for i in range(d):
                    v = V[c, i, xb[:, i], :]   # (B, K)
                    S += v
                    Q += v * v
                inter[:, c] = 0.5 * np.sum(S * S - Q, axis=1)

            scores = w0 + linear + inter        # (B, C)
            # softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            # --- gradients ---
            grad_scores = probs.copy()
            grad_scores[np.arange(B), yb] -= 1.0   # (B, C)

            # w0 grad
            g_w0 = grad_scores.sum(axis=0) / B

            # w grad
            g_w = np.zeros_like(w)
            for c in range(C):
                for i in range(d):
                    # accumulate gradient for each value
                    vals = xb[:, i]
                    np.add.at(g_w[i, :, c], vals, grad_scores[:, c])
            g_w /= B

            # V grad
            g_V = np.zeros_like(V)
            for c in range(C):
                g_sc = grad_scores[:, c]   # (B,)
                # precompute S for this class
                S = np.zeros((B, K), dtype=np.float32)
                for i in range(d):
                    S += V[c, i, xb[:, i], :]
                for i in range(d):
                    v = V[c, i, xb[:, i], :]   # (B, K)
                    # dI/dv = S - v
                    dv = (S - v) * g_sc[:, None]   # (B, K)
                    vals = xb[:, i]
                    for f in range(K):
                        np.add.at(g_V[c, i, :, f], vals, dv[:, f])
            g_V /= B

            # L2 regularization
            g_w += L2 * w
            g_V += L2 * V

            # Adam update
            t += 1
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # --- write weights ---
    # format: magic, d, C, K (int32), then w0[C], w[d*256*C], V[C*d*256*K] as float32 LE
    magic = 0x464D5F31  # 'FM_1'
    header = struct.pack("<4i", magic, d, C, K)
    body = w0.astype("<f4").tobytes()
    body += w.astype("<f4").tobytes()
    body += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (argmax). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (d <= 0 || C <= 0 || K <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * (C + (long)d * 256 * C + (long)C * d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *w  = w0 + C;
    const float *V  = w + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_score = -1e30f;
        best = 0;
        for (int c = 0; c < C; c++) {
            float score = w0[c];
            float sum_vec[16], sqsum[16];  // K <= 16 in practice
            for (int f = 0; f < K; f++) { sum_vec[f] = 0.0f; sqsum[f] = 0.0f; }

            for (int j = 0; j < d; j++) {
                uint8_t val = xi[j];
                // linear term
                score += w[((long)j * 256 + val) * C + c];
                // interaction term
                const float *v_ptr = V + (((long)c * d + j) * 256 + val) * K;
                for (int f = 0; f < K; f++) {
                    float v = v_ptr[f];
                    sum_vec[f] += v;
                    sqsum[f] += v * v;
                }
            }
            for (int f = 0; f < K; f++) {
                score += 0.5f * (sum_vec[f] * sum_vec[f] - sqsum[f]);
            }
            if (score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```