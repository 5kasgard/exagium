```hypothesis
Factorization Machine (label-fm) with 8-dim embeddings per feature-value. Adult has strong pairwise interactions (education×occupation, marital×relationship) that linear models miss; FM captures them via dot-product of summed embeddings. Inference stays O(d·K) with K=8 (≈1000 ops/item), similar to label-linear's O(d). If dev accuracy ≤0.8524 or latency >2× label-linear, the hypothesis fails.
```
```python
import json, math, struct, sys

import numpy as np

K = 8
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
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    L = rng.normal(0, 0.01, (C, d, 256)).astype(np.float32)
    E = rng.normal(0, 0.01, (C, d, 256, K)).astype(np.float32)

    m_L, v_L = np.zeros_like(L), np.zeros_like(L)
    m_E, v_E = np.zeros_like(E), np.zeros_like(E)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            B = len(yy)

            scores = np.zeros((B, C), np.float32)
            sum_emb = np.zeros((B, C, K), np.float32)
            sum_emb_sq = np.zeros((B, C, K), np.float32)
            linear = np.zeros((B, C), np.float32)

            for c in range(C):
                for j in range(d):
                    linear[:, c] += L[c, j, x[:, j]]
                for j in range(d):
                    emb = E[c, j, x[:, j], :]
                    sum_emb[:, c, :] += emb
                    sum_emb_sq[:, c, :] += emb * emb
                fm = 0.5 * np.sum(sum_emb[:, c, :]**2 - sum_emb_sq[:, c, :], axis=1)
                scores[:, c] = linear[:, c] + fm

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            grad_scores = probs.copy()
            grad_scores[np.arange(B), yy] -= 1.0
            grad_scores /= B

            grad_L = np.zeros_like(L)
            grad_E = np.zeros_like(E)

            for c in range(C):
                g = grad_scores[:, c]
                for j in range(d):
                    np.add.at(grad_L[c, j], x[:, j], g)
                for j in range(d):
                    emb = E[c, j, x[:, j], :]
                    diff = sum_emb[:, c, :] - emb
                    g_diff = g[:, None] * diff
                    for kk in range(K):
                        np.add.at(grad_E[c, j, :, kk], x[:, j], g_diff[:, kk])

            grad_L += L2 * L
            grad_E += L2 * E

            t += 1
            b1, b2 = 0.9, 0.999
            eps = 1e-8
            m_L = b1 * m_L + (1 - b1) * grad_L
            v_L = b2 * v_L + (1 - b2) * grad_L * grad_L
            L -= LR * (m_L / (1 - b1**t)) / (np.sqrt(v_L / (1 - b2**t)) + eps)
            m_E = b1 * m_E + (1 - b1) * grad_E
            v_E = b2 * v_E + (1 - b2) * grad_E * grad_E
            E -= LR * (m_E / (1 - b1**t)) / (np.sqrt(v_E / (1 - b2**t)) + eps)

    magic = 0x464D4C31
    out = struct.pack("<4i", magic, d, K, C)
    out += L.astype("<f4").tobytes()
    out += E.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * C * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *L = (const float *)(hd + 4);
    const float *E = L + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * C * K);
    float *sum_emb_sq = malloc(sizeof(float) * C * K);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            score[c] = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                sum_emb[c*K + kk] = 0.0f;
                sum_emb_sq[c*K + kk] = 0.0f;
            }
        }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                score[c] += L[((long)c * d + j) * 256 + v];
                const float *e = E + (((long)c * d + j) * 256 + v) * K;
                for (int kk = 0; kk < K; kk++) {
                    float val = e[kk];
                    sum_emb[c*K + kk] += val;
                    sum_emb_sq[c*K + kk] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float fm = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float s = sum_emb[c*K + kk];
                fm += s * s - sum_emb_sq[c*K + kk];
            }
            score[c] += 0.5f * fm;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```