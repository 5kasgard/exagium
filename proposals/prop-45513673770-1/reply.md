```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression).
Why: Optdigits pixels have strong local correlations; quadratic terms capture stroke curvature and intersections that linear models miss, while staying far cheaper than MLPs or trees.
Failure signal: If dev accuracy ≤ 0.965 (only ~0.004 above label-linear) the extra features do not pay off.
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

    # Per-feature z-scoring (all features are ordinal)
    means = X.mean(axis=0).astype(np.float64)
    stds = X.std(axis=0).astype(np.float64)
    stds[stds == 0] = 1.0
    X_norm = (X - means) / stds  # (n, d)

    # Polynomial features: linear + quadratic (i <= j)
    num_quad = d * (d + 1) // 2
    num_features = d + num_quad
    Z = np.empty((n, num_features), dtype=np.float32)
    Z[:, :d] = X_norm
    idx = d
    for i in range(d):
        for j in range(i, d):
            Z[:, idx] = X_norm[:, i] * X_norm[:, j]
            idx += 1

    # Model parameters
    rng = np.random.default_rng(SEED)
    W = np.zeros((num_features, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            z = Z[idx_batch]
            yy = y[idx_batch]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)
            gW = z.T @ grad + L2 * W
            gb = grad.sum(axis=0)
            t += 1
            mW = 0.9 * mW + 0.1 * gW
            vW = 0.999 * vW + 0.001 * gW * gW
            W -= LR * (mW / (1 - 0.9 ** t)) / (np.sqrt(vW / (1 - 0.999 ** t)) + 1e-8)
            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb * gb
            b -= LR * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)

    # Write weights.bin: magic, d, C, means[d], stds[d], weights[C][num_features] (class-major), bias[C]
    magic = 0x504F4C32  # "POL2" little-endian
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<iii", magic, d, C))
        f.write(means.astype("<f4").tobytes())
        f.write(stds.astype("<f4").tobytes())
        f.write(W.T.astype("<f4").tobytes())  # class-major
        f.write(b.astype("<f4").tobytes())

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
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    int32_t magic = hdr[0];
    int d = hdr[1];
    int C = hdr[2];
    if (magic != 0x504F4C32 || d != 64 || C != 10) { fprintf(stderr, "bad weights header\n"); return 3; }
    int num_features = d + d * (d + 1) / 2;  // 2144
    long expected = 12L + (long)d * 4 * 2 + (long)C * num_features * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *means = (float *)(wb + 12);
    float *stds = means + d;
    float *weights = stds + d;          // class-major: weights[c * num_features + f]
    float *bias = weights + (long)C * num_features;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float x_norm[64];
        for (int j = 0; j < d; j++) {
            float std = stds[j];
            x_norm[j] = (std != 0.0f) ? ((float)xi[j] - means[j]) / std : 0.0f;
        }

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // Linear features
        for (int j = 0; j < d; j++) {
            float val = x_norm[j];
            for (int c = 0; c < C; c++) scores[c] += val * weights[c * num_features + j];
        }

        // Quadratic features (i <= j)
        int fidx = d;
        for (int ii = 0; ii < d; ii++) {
            for (int jj = ii; jj < d; jj++) {
                float val = x_norm[ii] * x_norm[jj];
                for (int c = 0; c < C; c++) scores[c] += val * weights[c * num_features + fidx];
                fidx++;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```