```hypothesis
Family: label-fm (factorization machine with FM-trick for fast inference).
Why: Adult income depends on feature interactions (e.g., education×occupation, capital-gain×hours). FM captures all pairwise interactions via low-rank embeddings (K=4) while the FM trick reduces inference to O(dK) ≈ linear cost. More expressive than label-linear, faster than label-mlp16.
Failure: If interactions are weak or high-rank, or K=4 is too small, accuracy may not exceed label-linear (0.8524).
```
```python
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 256, 600000, 100


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
    n, d = X.shape
    C = card["classes"]
    assert d == 14 and C == 2

    rng = np.random.default_rng(SEED)

    w = np.zeros((C, d, 256), dtype=np.float32)
    v = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_v = np.zeros_like(v); v_v = np.zeros_like(v)
    m_b = np.zeros_like(bias); v_b = np.zeros_like(bias)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]
            yb = y[idx]
            B = len(idx)

            scores = np.zeros((B, C), dtype=np.float32)
            sum_v_store = np.zeros((B, C, K), dtype=np.float32)

            for c in range(C):
                w_c = w[c]
                v_c = v[c]
                w_sum = np.zeros(B, dtype=np.float32)
                sum_v = np.zeros((B, K), dtype=np.float32)
                sum_sq = np.zeros(B, dtype=np.float32)
                for i in range(d):
                    vals = Xb[:, i]
                    w_sum += w_c[i, vals]
                    v_feat = v_c[i, vals]
                    sum_v += v_feat
                    sum_sq += np.sum(v_feat * v_feat, axis=1)
                sum_v_store[:, c] = sum_v
                interaction = 0.5 * (np.sum(sum_v * sum_v, axis=1) - sum_sq)
                scores[:, c] = bias[c] + w_sum + interaction

            scores_max = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            grad_scores = probs.copy()
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B

            dbias = np.sum(grad_scores, axis=0)
            dw = np.zeros_like(w)
            dv = np.zeros_like(v)

            for c in range(C):
                g = grad_scores[:, c]
                sum_v = sum_v_store[:, c]
                for i in range(d):
                    vals = Xb[:, i]
                    np.add.at(dw[c, i], vals, g)
                    v_feat = v[c, i, vals]
                    factor = g[:, None] * (sum_v - v_feat)
                    np.add.at(dv[c, i], vals, factor)

            dw += L2 * w
            dv += L2 * v

            t += 1
            m_w = 0.9 * m_w + 0.1 * dw
            v_w = 0.999 * v_w + 0.001 * dw * dw
            w -= LR * (m_w / (1 - 0.9 ** t)) / (np.sqrt(v_w / (1 - 0.999 ** t)) + 1e-8)

            m_v = 0.9 * m_v + 0.1 * dv
            v_v = 0.999 * v_v + 0.001 * dv * dv
            v -= LR * (m_v / (1 - 0.9 ** t)) / (np.sqrt(v_v / (1 - 0.999 ** t)) + 1e-8)

            m_b = 0.9 * m_b + 0.1 * dbias
            v_b = 0.999 * v_b + 0.001 * dbias * dbias
            bias -= LR * (m_b / (1 - 0.9 ** t)) / (np.sqrt(v_b / (1 - 0.999 ** t)) + 1e-8)

    magic = 0x464D4C31
    out = struct.pack("<4i", magic, d, K, C)
    out += bias.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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
    if (d != 14 || C != 2 || K < 1 || K > 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (C + C * d * 256 + C * d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *w = bias + C;
    const float *v = w + C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_score = -1e30f;
        unsigned char best_class = 0;

        for (int c = 0; c < C; c++) {
            float score = bias[c];
            float sum_v[16];
            float sum_sq = 0.0f;
            for (int f = 0; f < K; f++) sum_v[f] = 0.0f;

            const float *w_c = w + c * d * 256;
            const float *v_c = v + c * d * 256 * K;

            for (int j = 0; j < d; j++) {
                int val = xi[j];
                score += w_c[j * 256 + val];
                const float *v_feat = v_c + (j * 256 + val) * K;
                for (int f = 0; f < K; f++) {
                    float vf = v_feat[f];
                    sum_v[f] += vf;
                    sum_sq += vf * vf;
                }
            }

            float dot_sum_v = 0.0f;
            for (int f = 0; f < K; f++) dot_sum_v += sum_v[f] * sum_v[f];
            float interaction = 0.5f * (dot_sum_v - sum_sq);
            score += interaction;

            if (c == 0 || score > best_score) {
                best_score = score;
                best_class = (unsigned char)c;
            }
        }

        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```