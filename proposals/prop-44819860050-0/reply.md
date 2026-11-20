```hypothesis
Family: label-fm (Factorization Machine with per-feature embeddings).
Why: Adult data has strong feature interactions (e.g., education × occupation). A second-order FM captures pairwise interactions with low-rank embeddings (k=8), adding only ~260 KB weights. It should surpass linear (0.8524) and approach MLP16 (0.8591) at lower inference cost (≈1500 Ir/item). 
Failure signal: dev accuracy ≤ 0.8524 or inference cost > 2500 Ir/item.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, k, C; float32 w0[C], w[C][d][256], v[C][d][256][k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 8          # embedding dimension

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
    # Parameters per class: w0 (scalar), w[d][256], v[d][256][K]
    w0 = np.zeros(C, dtype=np.float32)
    w = (rng.normal(0, 0.01, (C, d, 256)).astype(np.float32))
    v = (rng.normal(0, 0.01, (C, d, 256, K)).astype(np.float32) * math.sqrt(2.0 / (256 + K)))

    # Adam state
    m_w0, v_w0 = np.zeros_like(w0), np.zeros_like(w0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward pass for all classes
            # Linear part: w0 + sum_j w[c, j, xb[:,j]]
            lin = w0[:, None] + np.sum(w[:, np.arange(d), xb.T], axis=2)   # (C, B)

            # Interaction part: 0.5 * ( (sum_j v)^2 - sum_j v^2 )
            # v_batch: (C, B, d, K)
            v_batch = v[:, np.arange(d), xb.T]  # (C, d, B, K) -> transpose to (C, B, d, K)
            v_batch = np.transpose(v_batch, (0, 2, 1, 3))
            sum_v = np.sum(v_batch, axis=2)          # (C, B, K)
            sum_v2 = np.sum(v_batch * v_batch, axis=2)  # (C, B, K)
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=2)  # (C, B)

            scores = lin + inter                     # (C, B)
            # Softmax
            scores -= scores.max(axis=0, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=0, keepdims=True)  # (C, B)

            # Gradient wrt scores
            grad_scores = probs.copy()
            grad_scores[yb, np.arange(B)] -= 1.0
            grad_scores /= B

            # Gradients for w0, w, v
            g_w0 = grad_scores.sum(axis=1)                           # (C,)
            g_w = np.zeros_like(w)                                   # (C, d, 256)
            g_v = np.zeros_like(v)                                   # (C, d, 256, K)

            # For each class and feature, accumulate gradients
            for c in range(C):
                g_w[c, np.arange(d), xb.T] += grad_scores[c][:, None]   # (d, B) -> sum over batch
                # v gradient: (sum_v - v_ij) * grad_scores
                # sum_v[c] is (B, K), v_batch[c] is (B, d, K)
                diff = sum_v[c][:, None, :] - v_batch[c]               # (B, d, K)
                g_v[c, np.arange(d), xb.T] += (diff * grad_scores[c][:, None, None]).sum(axis=0)

            # Add L2 regularization (on w and v, not w0)
            g_w += L2 * w
            g_v += L2 * v

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
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

            # v
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v
            m_hat = m_v / (1 - beta1 ** t)
            v_hat = v_v / (1 - beta2 ** t)
            v -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights.bin
    # Layout: int32 magic, d, K, C; float32 w0[C]; float32 w[C][d][256]; float32 v[C][d][256][K]
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    out = bytearray(header)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (d=14). Output: one byte class index (0 or 1). */
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

    // Expected weight size: 4 ints + C floats (w0) + C*d*256 floats (w) + C*d*256*K floats (v)
    long expected = 16
        + (long)C * 4
        + (long)C * d * 256 * 4
        + (long)C * d * 256 * K * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *w = p; p += (long)C * d * 256;
    const float *v = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float scores[2] = {w0[0], w0[1]};
        float sum_v[2][8];       // K <= 8, hardcoded max for stack allocation
        float sum_v2[2][8];
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                sum_v[c][kk] = 0.0f;
                sum_v2[c][kk] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            for (int c = 0; c < C; c++) {
                // linear weight
                scores[c] += w[((long)c * d + j) * 256 + val];
                // embedding vector
                const float *v_ptr = v + (((long)c * d + j) * 256 + val) * K;
                for (int kk = 0; kk < K; kk++) {
                    float vv = v_ptr[kk];
                    sum_v[c][kk] += vv;
                    sum_v2[c][kk] += vv * vv;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                inter += sum_v[c][kk] * sum_v[c][kk] - sum_v2[c][kk];
            }
            scores[c] += 0.5f * inter;
        }

        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```