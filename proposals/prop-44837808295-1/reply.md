```hypothesis
Family: label-poly2 (quadratic logistic regression on z-scored pixels).
Why: Optdigits pixels have spatial structure; pairwise products capture strokes/corners that linear models miss.
Expected: ~0.970 dev accuracy (beats label-linear 0.961) at ~15x inference cost (~6000 ns/item).
Failure: If accuracy < 0.965 or cost > 10000 ns/item, the feature expansion isn't worth it.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 1e-2
L2 = 1e-3
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000


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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Z-score per pixel (ordinal encoding)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)  # n x d

    # Polynomial features: linear + quadratic (upper triangular including diagonal)
    n = Z.shape[0]
    n_quad = d * (d + 1) // 2
    n_feat = d + n_quad
    Phi = np.empty((n, n_feat), dtype=np.float32)
    Phi[:, :d] = Z
    col = d
    for i in range(d):
        for j in range(i, d):
            Phi[:, col] = Z[:, i] * Z[:, j]
            col += 1

    # Initialize weights (Xavier for linear, smaller for quadratic)
    W = np.zeros((n_feat, C), dtype=np.float32)
    W[:d] = rng.normal(0, math.sqrt(2.0 / d), (d, C)).astype(np.float32)
    W[d:] = rng.normal(0, math.sqrt(1.0 / n_quad), (n_quad, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            phi = Phi[idx]
            yy = y[idx]

            logits = phi @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)  # n x C

            gW = phi.T @ grad + L2 * W
            gb = grad.sum(axis=0)

            t += 1
            mW = 0.9 * mW + 0.1 * gW
            vW = 0.999 * vW + 0.001 * gW * gW
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)

            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb * gb
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    W_lin = W[:d]
    W_quad = W[d:]

    # Write weights: magic, d, C, degree, mean[d], std[d], W_lin[d*C], W_quad[n_quad*C], bias[C]
    magic = 0x504F4C32  # 'POL2'
    out = struct.pack("<4i", magic, d, C, 2)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += W_quad.astype("<f4").tobytes()
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

static inline int quad_idx(int i, int j, int d) {
    return i * (2 * d - i + 1) / 2 + (j - i);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], degree = hd[3];
    if (d != 64 || C != 10 || degree != 2) { fprintf(stderr, "bad weights dims\n"); return 3; }

    int n_quad = d * (d + 1) / 2;
    int n_feat = d + n_quad;
    long expected = 16 + (long)d * 4 * 2 + (long)n_feat * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *mean = (float *)(hd + 4);
    float *std = mean + d;
    float *W_lin = std + d;
    float *W_quad = W_lin + (long)d * C;
    float *bias = W_quad + (long)n_quad * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) z[j] = (xi[j] - mean[j]) / std[j];

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        for (int j = 0; j < d; j++) {
            float zj = z[j];
            for (int c = 0; c < C; c++) scores[c] += zj * W_lin[(long)j * C + c];
        }

        for (int i = 0; i < d; i++) {
            float zi = z[i];
            for (int j = i; j < d; j++) {
                float zij = zi * z[j];
                int idx = quad_idx(i, j, d);
                for (int c = 0; c < C; c++) scores[c] += zij * W_quad[(long)idx * C + c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```