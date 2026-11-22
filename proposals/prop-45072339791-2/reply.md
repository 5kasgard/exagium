```hypothesis
The label-hashbigram family extends the linear unigram model with hashed bigram features (adjacent byte pairs). 
This captures local sequential structure critical for character-level language modeling while retaining a linear model's 
convex training and fast inference. The hash reduces bigram space (256²) to 1024 buckets, adding only 15 table lookups 
per example. We expect dev accuracy >0.3375 (label-linear) with ~2× the instruction count. If accuracy does not improve, 
it indicates that bigram statistics are either already captured by unigrams or that hashing collisions destroy signal.
```
```python
"""Hash bigram family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Adds hashed bigram features to the linear unigram model. Deterministic (seed 1), single thread."""
import json, math, struct, sys

import numpy as np

B = 1024                 # bigram hash buckets
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x48424731       # 'HBG1'

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
    assert d == 16 and C == 65

    # feature layout: 16 unigram positions (256 each) + 15 bigram positions (B each)
    unigram_total = d * 256
    bigram_total = (d - 1) * B
    total_k = unigram_total + bigram_total

    # precompute feature indices for every example (31 features each)
    idx_arr = np.empty((n, 31), dtype=np.int32)
    for i in range(n):
        # unigrams
        for j in range(d):
            idx_arr[i, j] = j * 256 + X[i, j]
        # bigrams
        for j in range(d - 1):
            h = (X[i, j] * 257 + X[i, j + 1]) % B
            idx_arr[i, 16 + j] = unigram_total + j * B + h

    rng = np.random.default_rng(SEED)
    W = np.zeros((total_k, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            bs = len(batch_idx)
            # build dense batch matrix (bs, total_k)
            Z = np.zeros((bs, total_k), dtype=np.float32)
            rows = np.arange(bs)[:, None]
            Z[rows, idx_arr[batch_idx]] = 1.0

            scores = Z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            yy = y[batch_idx]
            probs[np.arange(bs), yy] -= 1.0
            grad_scores = probs / bs

            grad_W = Z.T @ grad_scores + L2 * W
            grad_b = grad_scores.sum(axis=0)

            t += 1
            for param, grad, m, v in [(W, grad_W, mW, vW), (b, grad_b, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights: magic, d, B, C, then unigram weights (d*256*C), bigram weights ((d-1)*B*C), bias (C)
    header = struct.pack("<4i", MAGIC, d, B, C)
    body = W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* Hash bigram linear model for label tasks: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x48424731) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], B = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + (long)(d - 1) * B * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *unigram = W;
    const float *bigram = unigram + (long)d * 256 * C;
    const float *bias = bigram + (long)(d - 1) * B * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *w = unigram + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        for (int j = 0; j < d - 1; j++) {
            int h = (xi[j] * 257 + xi[j + 1]) % B;
            const float *w = bigram + ((long)j * B + h) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        out_byte = best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```