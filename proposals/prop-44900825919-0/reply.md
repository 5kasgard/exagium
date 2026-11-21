```hypothesis
Family: label-fm (Factorization Machine with 4-dim embeddings). Captures pairwise feature interactions (e.g., education×occupation) crucial for tabular income prediction, while staying lightweight: 14×256×4 embeddings + linear terms ≈ 86 KB weights. Inference: 91 4-dot products + 28 linear adds ≈ 400 FLOPs/item, targeting <1000 Ir/item. Fails if dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 2325 (label-mlp16).
```
```python
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 50


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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    linear = np.zeros((C, d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            B = len(idx)

            lin_scores = np.zeros((B, C), dtype=np.float32)
            for c in range(C):
                for j in range(d):
                    lin_scores[:, c] += linear[c, j, x[:, j]]

            inter_scores = np.zeros(B, dtype=np.float32)
            for j1 in range(d):
                for j2 in range(j1 + 1, d):
                    e1 = emb[j1, x[:, j1]]
                    e2 = emb[j2, x[:, j2]]
                    inter_scores += np.sum(e1 * e2, axis=1)

            scores = lin_scores + inter_scores[:, None]
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            grad_scores = probs.astype(np.float32)
            grad_scores[np.arange(B), yy] -= 1.0
            grad_scores /= B

            grad_linear = np.zeros_like(linear)
            for c in range(C):
                for j in range(d):
                    np.add.at(grad_linear[c, j], x[:, j], grad_scores[:, c])
            grad_linear += L2 * linear

            grad_emb = np.zeros_like(emb)
            grad_inter = grad_scores.sum(axis=1)
            for j1 in range(d):
                for j2 in range(j1 + 1, d):
                    e1 = emb[j1, x[:, j1]]
                    e2 = emb[j2, x[:, j2]]
                    g1 = grad_inter[:, None] * e2
                    g2 = grad_inter[:, None] * e1
                    np.add.at(grad_emb[j1], x[:, j1], g1)
                    np.add.at(grad_emb[j2], x[:, j2], g2)
            grad_emb += L2 * emb

            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for param, grad, m, v in [(linear, grad_linear, m_lin, v_lin),
                                       (emb, grad_emb, m_emb, v_emb)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    magic = 0x464D31
    out = struct.pack("<4i", magic, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (C * d * 256 + d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            scores[c] = 0.0f;
            for (int j = 0; j < d; j++)
                scores[c] += linear[(c * d + j) * 256 + xi[j]];
        }

        float inter = 0.0f;
        for (int j1 = 0; j1 < d; j1++) {
            const float *e1 = emb + (j1 * 256 + xi[j1]) * K;
            for (int j2 = j1 + 1; j2 < d; j2++) {
                const float *e2 = emb + (j2 * 256 + xi[j2]) * K;
                float dot = 0.0f;
                for (int f = 0; f < K; f++) dot += e1[f] * e2[f];
                inter += dot;
            }
        }
        for (int c = 0; c < C; c++) scores[c] += inter;

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```