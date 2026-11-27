```hypothesis
Family: label-fm (Factorization Machine with per-class embeddings).
Reason: The Adult dataset is known to have strong pairwise feature interactions (e.g., education × occupation, marital-status × occupation). A factorization machine captures all second-order interactions via low-rank embeddings, adding only O(d·K) parameters and inference cost. With K=8 it should surpass the linear baseline (0.8524) while remaining far faster than GBDT or KNN.
Failure signal: If dev accuracy does not exceed label-linear by at least 0.5 percentage points (i.e., stays below 0.8574), the hypothesis is invalid.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, K, C; float32 bias[C], linear[d][256][C], emb[d][256][C][K]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2_LIN, L2_EMB, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 1e-4, 64, 600000, 100


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
    d, C = card["input"]["bytes"], card["classes"]
    n = X.shape[0]
    V = 256

    rng = np.random.default_rng(SEED)

    bias = np.zeros(C, dtype=np.float32)
    linear = rng.normal(0, 0.01, (d, V, C)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, V, C, K)).astype(np.float32)

    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(idx)

            # Forward
            lin_scores = np.zeros((B, C), dtype=np.float32)
            for j in range(d):
                lin_scores += linear[j, xb[:, j], :]

            inter_scores = np.zeros((B, C), dtype=np.float32)
            for c in range(C):
                s_sum = np.zeros((B, K), dtype=np.float32)
                s_sq = np.zeros((B, K), dtype=np.float32)
                for j in range(d):
                    e = emb[j, xb[:, j], c, :]
                    s_sum += e
                    s_sq += e * e
                inter_scores[:, c] = 0.5 * np.sum(s_sum * s_sum - s_sq, axis=1)

            scores = bias + lin_scores + inter_scores
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B

            # Backward
            grad_bias = grad_scores.sum(axis=0)

            grad_lin = np.zeros_like(linear)
            for j in range(d):
                np.add.at(grad_lin[j], xb[:, j], grad_scores)

            grad_emb = np.zeros_like(emb)
            for c in range(C):
                g = grad_scores[:, c]
                s_sum = np.zeros((B, K), dtype=np.float32)
                for j in range(d):
                    s_sum += emb[j, xb[:, j], c, :]
                for j in range(d):
                    e = emb[j, xb[:, j], c, :]
                    grad = g[:, None] * (s_sum - e)
                    np.add.at(grad_emb[j, :, c, :], xb[:, j], grad)

            grad_lin += L2_LIN * linear
            grad_emb += L2_EMB * emb

            t += 1
            # Adam updates
            m_bias = 0.9 * m_bias + 0.1 * grad_bias
            v_bias = 0.999 * v_bias + 0.001 * grad_bias * grad_bias
            bias -= LR * (m_bias / (1 - 0.9 ** t)) / (np.sqrt(v_bias / (1 - 0.999 ** t)) + 1e-8)

            m_lin = 0.9 * m_lin + 0.1 * grad_lin
            v_lin = 0.999 * v_lin + 0.001 * grad_lin * grad_lin
            linear -= LR * (m_lin / (1 - 0.9 ** t)) / (np.sqrt(v_lin / (1 - 0.999 ** t)) + 1e-8)

            m_emb = 0.9 * m_emb + 0.1 * grad_emb
            v_emb = 0.999 * v_emb + 0.001 * grad_emb * grad_emb
            emb -= LR * (m_emb / (1 - 0.9 ** t)) / (np.sqrt(v_emb / (1 - 0.999 ** t)) + 1e-8)

    # Write weights
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    const int V = 256;
    long expected = 16L + 4L * (C + (long)d * V * C + (long)d * V * C * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + (long)d * V * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *s = calloc(C * K, sizeof(float));
    float *sq = calloc(C * K, sizeof(float));
    float scores[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        scores[0] = bias[0]; scores[1] = bias[1];
        memset(s, 0, C * K * sizeof(float));
        memset(sq, 0, C * K * sizeof(float));

        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            long base_lin = ((long)j * V + v) * C;
            for (int c = 0; c < C; c++) scores[c] += linear[base_lin + c];

            long base_emb = ((long)j * V + v) * C * K;
            for (int c = 0; c < C; c++) {
                for (int kk = 0; kk < K; kk++) {
                    float e = emb[base_emb + c * K + kk];
                    s[c * K + kk] += e;
                    sq[c * K + kk] += e * e;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float sk = s[c * K + kk];
                inter += sk * sk - sq[c * K + kk];
            }
            scores[c] += 0.5f * inter;
        }

        unsigned char best = scores[1] > scores[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }

    free(s); free(sq);
    return 0;
}
```