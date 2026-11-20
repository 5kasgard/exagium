```hypothesis
Family: label-fm2 (Factorization Machine with adjacent-position interactions).
Why: Character prediction depends heavily on local bigram statistics (byte i and i+1). label-linear only uses unigram features per position. label-fm2 adds a learned 8-dim embedding per (position, byte) and weights the dot product of adjacent embeddings per class, capturing position-specific bigram transitions with only ~1.2 MB weights and ~2.3K FLOPs/item.
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost > 10K Ir/item, the interaction terms do not justify their complexity.
```
```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 15
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


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

    rng = np.random.default_rng(SEED)

    bias = np.zeros(C, np.float32)
    linear = (rng.normal(0, 0.01, (d, 256, C)) * math.sqrt(2.0 / (256 + C))).astype(np.float32)
    embed = (rng.normal(0, 0.01, (d, 256, K)) * math.sqrt(2.0 / (256 + K))).astype(np.float32)
    interact = np.zeros((d - 1, C), np.float32)

    params = [bias, linear, embed, interact]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    idx = np.arange(n)
    for _ in range(EPOCHS):
        rng.shuffle(idx)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s + BATCH]
            B = len(batch_idx)
            xb = X[batch_idx]
            yb = y[batch_idx]

            scores = np.tile(bias, (B, 1))
            for j in range(d):
                scores += linear[j, xb[:, j], :]

            E = np.zeros((B, d, K), np.float32)
            for j in range(d):
                E[:, j, :] = embed[j, xb[:, j], :]

            inter_scores = np.zeros((B, d - 1), np.float32)
            for j in range(d - 1):
                inter_scores[:, j] = np.sum(E[:, j, :] * E[:, j + 1, :], axis=1)

            scores += inter_scores @ interact

            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)

            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B

            grad_interact = inter_scores.T @ grad_scores
            grad_inter_scores = grad_scores @ interact.T

            grad_bias = grad_scores.sum(axis=0)
            grad_linear = np.zeros_like(linear)
            grad_embed = np.zeros_like(embed)

            for j in range(d):
                np.add.at(grad_linear[j], xb[:, j], grad_scores)

            for j in range(d - 1):
                d_dot = grad_inter_scores[:, j]
                np.add.at(grad_embed[j], xb[:, j], d_dot[:, None] * E[:, j + 1, :])
                np.add.at(grad_embed[j + 1], xb[:, j + 1], d_dot[:, None] * E[:, j, :])

            grads = [grad_bias, grad_linear, grad_embed, grad_interact]
            grads[1] += L2 * linear
            grads[2] += L2 * embed
            grads[3] += L2 * interact

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * gr
                v[i] = BETA2 * v[i] + (1 - BETA2) * (gr * gr)
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    magic, num_interact = 0x464D3231, d - 1
    header = struct.pack("<5i", magic, d, C, K, num_interact)
    out = header
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    out += interact.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3], num_interact = hd[4];
    if (d != 16 || C != 65 || K != 8 || num_interact != 15) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20 + (long)C * 4 + (long)d * 256 * C * 4 + (long)d * 256 * K * 4 + (long)num_interact * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *ptr = (float *)(wb + 20);
    float *bias = ptr; ptr += C;
    float *linear = ptr; ptr += (long)d * 256 * C;
    float *embed = ptr; ptr += (long)d * 256 * K;
    float *interact = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));
    float *E = malloc(d * K * sizeof(float));
    float *inter_scores = malloc(num_interact * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, bias, C * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_ptr = linear + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += lin_ptr[c];
            const float *emb_ptr = embed + ((long)j * 256 + v) * K;
            memcpy(E + j * K, emb_ptr, K * sizeof(float));
        }

        for (int j = 0; j < num_interact; j++) {
            float dot = 0.0f;
            for (int kk = 0; kk < K; kk++) dot += E[j * K + kk] * E[(j + 1) * K + kk];
            inter_scores[j] = dot;
        }

        for (int j = 0; j < num_interact; j++) {
            float w = inter_scores[j];
            const float *ip = interact + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += w * ip[c];
        }

        unsigned char best = 0;
        float best_score = scores[0];
        for (int c = 1; c < C; c++) if (scores[c] > best_score) { best_score = scores[c]; best = (unsigned char)c; }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```