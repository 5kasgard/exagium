```hypothesis
Family: label-fm (Factorization Machine). The Adult dataset has strong pairwise interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. A factorization machine with K=4 learns 4-dimensional embeddings per feature-value, capturing all pairwise interactions in O(d·K) time. It should beat label-linear's 0.8524 dev accuracy at ~2× inference cost (still <1500 Ir/item). If dev accuracy ≤0.8524 or Ir/item >1500, the hypothesis fails.
```
```python
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-5, 256, 300000, 50


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
    rng = np.random.default_rng(SEED)

    linear = rng.normal(0, 0.01, (C, d, 256)).astype(np.float32)
    embed = rng.normal(0, 0.01, (C, d, 256, K)).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [linear, embed, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(yb)

            # Forward: scores[C, B]
            scores = bias[:, None].copy()
            for c in range(C):
                for j in range(d):
                    scores[c] += linear[c, j, xb[:, j]]

            for c in range(C):
                for f in range(K):
                    sum_f = np.zeros(B, np.float32)
                    for j in range(d):
                        sum_f += embed[c, j, xb[:, j], f]
                    sum_sq_f = np.zeros(B, np.float32)
                    for j in range(d):
                        e = embed[c, j, xb[:, j], f]
                        sum_sq_f += e * e
                    scores[c] += 0.5 * (sum_f * sum_f - sum_sq_f)

            # Softmax gradient
            scores -= scores.max(axis=0, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=0, keepdims=True)
            probs[yb, np.arange(B)] -= 1.0
            grad_scores = probs / B  # [C, B]

            # Gradients
            grad_linear = np.zeros_like(linear)
            grad_embed = np.zeros_like(embed)
            grad_bias = grad_scores.sum(axis=1)

            for c in range(C):
                for j in range(d):
                    grad_linear[c, j] = np.bincount(xb[:, j], weights=grad_scores[c], minlength=256)

            for c in range(C):
                for f in range(K):
                    sum_f = np.zeros(B, np.float32)
                    for j in range(d):
                        sum_f += embed[c, j, xb[:, j], f]
                    for j in range(d):
                        e = embed[c, j, xb[:, j], f]
                        contrib = grad_scores[c] * (sum_f - e)
                        grad_embed[c, j, :, f] = np.bincount(xb[:, j], weights=contrib, minlength=256)

            grad_linear += L2 * linear
            grad_embed += L2 * embed

            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad_linear, grad_embed, grad_bias])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save: magic, d, K, C, linear[C,d,256], embed[C,d,256,K], bias[C]
    out = struct.pack("<4i", 0x464D3100, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 16 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)C * d * 256 + (long)C * d * 256 * K + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *embed = linear + (long)C * d * 256;
    const float *bias = embed + (long)C * d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        for (int c = 0; c < C; c++) {
            for (int j = 0; j < d; j++) {
                scores[c] += linear[((long)c * d + j) * 256 + xi[j]];
            }
        }

        for (int c = 0; c < C; c++) {
            for (int f = 0; f < K; f++) {
                float sum_f = 0.0f, sum_sq_f = 0.0f;
                for (int j = 0; j < d; j++) {
                    float e = embed[(((long)c * d + j) * 256 + xi[j]) * K + f];
                    sum_f += e;
                    sum_sq_f += e * e;
                }
                scores[c] += 0.5f * (sum_f * sum_f - sum_sq_f);
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```