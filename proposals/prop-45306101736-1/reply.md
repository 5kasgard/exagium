```hypothesis
Family: label-poly2 (quadratic polynomial logistic regression).
Why: Optdigits pixels have strong pairwise interactions (strokes, loops); degree-2 features capture these non-linearities while adding only ~2k features (vs 64 linear).
Expected: ~0.97 dev accuracy (beating label-linear's 0.961) at ~15k Ir/item (3x linear cost, still far below GBDT/MLP). Failure if accuracy <= 0.962 or Ir/item > 50k.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
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

    # Standardize per pixel (ordinal)
    Xf = X.astype(np.float64)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    X_std = ((Xf - mean) / std).astype(np.float32)

    # Polynomial features: linear + quadratic (i <= j)
    quad_dim = d * (d + 1) // 2
    feat_dim = d + quad_dim
    Z = np.empty((n, feat_dim), dtype=np.float32)
    Z[:, :d] = X_std
    col = d
    for i in range(d):
        for j in range(i, d):
            Z[:, col] = X_std[:, i] * X_std[:, j]
            col += 1

    rng = np.random.default_rng(SEED)
    limit = math.sqrt(6.0 / feat_dim)
    W = rng.uniform(-limit, limit, (feat_dim, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]

            logits = z @ W + b
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)

            gW = z.T @ grad_logits + L2 * W
            gb = grad_logits.sum(axis=0)

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

    # Export: magic, d, C, feat_dim, mean[d], std[d], W[feat_dim*C], b[C]
    magic = 0x504F4C32  # "POL2"
    out = struct.pack("<4i", magic, d, C, feat_dim)
    out += mean.astype(np.float32).tobytes()
    out += std.astype(np.float32).tobytes()
    out += W.astype(np.float32).tobytes()
    out += b.astype(np.float32).tobytes()
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
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }

    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }

    int32_t *hd = (int32_t *)wb;
    int d = hd[1], C = hd[2], feat_dim = hd[3];
    if (d != 64 || C != 10 || feat_dim != 64 + 64*65/2) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16 + 8L*d + 4L*feat_dim*C + 4L*C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    float *mean = (float *)(wb + 16);
    float *std = mean + d;
    float *W = (float *)(std + d);
    float *b = W + (long)feat_dim * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_std = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            x_std[j] = (xi[j] - mean[j]) / std[j];
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];

        for (int j = 0; j < d; j++) {
            float xj = x_std[j];
            for (int c = 0; c < C; c++) {
                scores[c] += xj * W[j * C + c];
            }
        }

        int idx = d;
        for (int i2 = 0; i2 < d; i2++) {
            float xi_val = x_std[i2];
            for (int j2 = i2; j2 < d; j2++) {
                float prod = xi_val * x_std[j2];
                for (int c = 0; c < C; c++) {
                    scores[c] += prod * W[idx * C + c];
                }
                idx++;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(x_std); free(scores);
    if (x) free(x);
    free(wb);
    return 0;
}
```