```hypothesis
Family: label-svm (multi-class linear SVM via one-vs-all Pegasos). Hinge loss often generalizes better than log-loss on small, clean datasets like optdigits. Inference is a single matrix-vector multiply per item (same cost as label-linear). If dev accuracy does not exceed label-linear's 0.9613, the hypothesis is falsified.
```
```python
"""label-svm: multi-class linear SVM (one-vs-all Pegasos) for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encoding: ordinal features z-scored per training statistics.
Weights format: magic 0x53564D4C, int32 d, int32 C, float32 mean[d], float32 std[d], float32 W[C*(d+1)] (row-major, last column is bias)."""
import json, struct, sys

import numpy as np

SEED = 1
LAMBDA = 1e-4
EPOCHS = 100


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

    # All features are ordinal per card
    rng = np.random.default_rng(SEED)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X_norm = (X.astype(np.float32) - mean) / std

    # Add bias column
    X_aug = np.hstack([X_norm, np.ones((X_norm.shape[0], 1), dtype=np.float32)])  # n x (d+1)
    n, d1 = X_aug.shape

    W = np.zeros((C, d1), dtype=np.float32)  # each row is weight vector for class c

    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        for c in range(C):
            w = W[c]
            for idx in perm:
                x = X_aug[idx]
                y_bin = 1.0 if y[idx] == c else -1.0
                t = epoch * n + idx + 1
                eta = 1.0 / (LAMBDA * t)
                margin = y_bin * np.dot(w, x)
                if margin < 1.0:
                    w = (1.0 - eta * LAMBDA) * w + eta * y_bin * x
                else:
                    w = (1.0 - eta * LAMBDA) * w
            W[c] = w

    # Pack weights
    out = struct.pack("<4i", 0x53564D4C, d, C, 0)  # last int unused placeholder
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-svm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (ordinal). Output record: 1 byte class index. */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x53564D4C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t *hd = (int32_t *)(wb + 4);
    int d = hd[0], C = hd[1];
    if (d != 64 || C != 10) { fprintf(stderr, "unexpected d/C\n"); return 3; }
    float *mean = (float *)(hd + 2);
    float *std = mean + d;
    float *W = std + d;  // C * (d+1)
    long expected = 16 + 4L * (2 * d + C * (d + 1));
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_aug = malloc((d + 1) * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            x_aug[j] = (xi[j] - mean[j]) / std[j];
        }
        x_aug[d] = 1.0f;  // bias
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            const float *wc = W + c * (d + 1);
            for (int j = 0; j <= d; j++) sum += x_aug[j] * wc[j];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```