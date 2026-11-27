```hypothesis
Family: label-poly2 (linear on 64 pixels + 112 adjacent pairwise products).
Why: Optdigits 8x8 images have spatial structure; products of horizontal/vertical neighbors capture stroke continuity missed by pure linear models. 176 features keep inference fast (single dot product).
Failure if: dev accuracy ≤ 0.9613 (label-linear) or Ir/item > 8000.
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def expand_features(X):  # X: (n, 64) uint8
    n = X.shape[0]
    f1 = X.astype(np.float32) / 16.0
    f2 = np.zeros((n, 112), dtype=np.float32)
    idx = 0
    for r in range(8):
        for c in range(7):
            f2[:, idx] = (X[:, r * 8 + c] * X[:, r * 8 + c + 1]).astype(np.float32) / 256.0
            idx += 1
    for r in range(7):
        for c in range(8):
            f2[:, idx] = (X[:, r * 8 + c] * X[:, (r + 1) * 8 + c]).astype(np.float32) / 256.0
            idx += 1
    return np.concatenate([f1, f2], axis=1)


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]

    Z = expand_features(X_bytes)
    n, d_exp = Z.shape

    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Z - mu) / sd

    rng = np.random.default_rng(SEED)
    W = np.zeros((d_exp, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    out = struct.pack("<4i", 0x504F4C59, 64, d_exp, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x504F4C59) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d_orig = hd[1], d_exp = hd[2], C = hd[3];
    if (d_orig != 64 || d_exp != 176 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * d_exp * 2 + 4L * d_exp * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *mu = (float *)(hd + 4);
    float *sd = mu + d_exp;
    float *W = sd + d_exp;
    float *b = W + (long)d_exp * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *features = malloc(sizeof(float) * d_exp);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d_orig) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d_orig); return 3; }

        for (int j = 0; j < 64; j++) features[j] = xi[j] / 16.0f;
        int idx = 64;
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 7; c++)
                features[idx++] = (xi[r * 8 + c] * xi[r * 8 + c + 1]) / 256.0f;
        for (int r = 0; r < 7; r++)
            for (int c = 0; c < 8; c++)
                features[idx++] = (xi[r * 8 + c] * xi[(r + 1) * 8 + c]) / 256.0f;

        for (int j = 0; j < d_exp; j++) features[j] = (features[j] - mu[j]) / sd[j];

        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int j = 0; j < d_exp; j++) sum += features[j] * W[(long)j * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```