```hypothesis
Family: label-fm (factorization machine with per-column per-value embeddings).
Why: Adult has strong feature interactions (e.g., education×occupation, marital×sex). A degree-2 FM captures all pairwise interactions via low-rank embeddings (K=4) with only ~36k parameters, adding minimal compute over linear. The per-value embedding tables fold into the same lookup pattern as label-linear, so inference stays fast.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 1000 (more than label-mlp16).
```

```python
"""Factorization machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Exports: magic 0x464D4C31, d, K, C; float32 w0[C], w[C][d][256], v[C][d][256][K]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 256, 400000, 50


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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # parameters per class
    w0 = np.zeros(C, np.float32)
    w = (rng.normal(0, 0.01, (C, d, 256)) * 0.0).astype(np.float32)  # zero init for linear
    v = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.zeros_like(w0), np.zeros_like(w0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yy = X[idx], y[idx]
            bsz = len(yy)

            # forward
            linear = w0 + w[:, xi, np.arange(d)].sum(axis=2).T  # [bsz, C]
            v_xi = v[:, xi, np.arange(d)]  # [C, bsz, d, K] -> transpose
            v_xi = np.transpose(v_xi, (1, 0, 2, 3))  # [bsz, C, d, K]
            sum_v = v_xi.sum(axis=2)          # [bsz, C, K]
            sum_v2 = (v_xi * v_xi).sum(axis=2)  # [bsz, C, K]
            inter = 0.5 * ((sum_v * sum_v - sum_v2).sum(axis=2))  # [bsz, C]
            scores = linear + inter  # [bsz, C]

            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            probs /= bsz  # [bsz, C]

            # gradients
            g_w0 = probs.sum(axis=0) + L2 * w0
            g_w = np.zeros_like(w)
            g_v = np.zeros_like(v)

            # vectorized gradient accumulation
            for c in range(C):
                pc = probs[:, c]  # [bsz]
                # linear weights gradient
                for j in range(d):
                    vals = xi[:, j]
                    np.add.at(g_w[c, j], vals, pc)
                g_w[c] += L2 * w[c]

                # embedding gradients
                # g_v[c, j, val, k] += pc[i] * (sum_v[i,c,k] - v_xi[i,c,j,k])
                sum_v_c = sum_v[:, c, :]  # [bsz, K]
                for j in range(d):
                    vals = xi[:, j]
                    v_xi_c_j = v_xi[:, c, j, :]  # [bsz, K]
                    diff = sum_v_c - v_xi_c_j  # [bsz, K]
                    for k in range(K):
                        np.add.at(g_v[c, j, :, k], vals, pc * diff[:, k])
                g_v[c] += L2 * v[c]

            t += 1
            # Adam updates
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
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

    # pack weights.bin
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class index. */
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
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 16) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 16
        + (size_t)C * sizeof(float)
        + (size_t)C * d * 256 * sizeof(float)
        + (size_t)C * d * 256 * K * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    float *p = (float *)(wb + 16);
    float *w0 = p; p += C;
    float *w = p; p += (size_t)C * d * 256;
    float *v = p; p += (size_t)C * d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * C * K);
    float *sum_v2 = malloc(sizeof(float) * C * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            scores[c] = w0[c];
            for (int kk = 0; kk < K; kk++) {
                sum_v[c * K + kk] = 0.0f;
                sum_v2[c * K + kk] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            for (int c = 0; c < C; c++) {
                scores[c] += w[((size_t)c * d + j) * 256 + val];
                float *v_ptr = &v[((size_t)c * d + j) * 256 * K + val * K];
                for (int kk = 0; kk < K; kk++) {
                    float vv = v_ptr[kk];
                    sum_v[c * K + kk] += vv;
                    sum_v2[c * K + kk] += vv * vv;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float inter = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float sv = sum_v[c * K + kk];
                inter += sv * sv - sum_v2[c * K + kk];
            }
            scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```