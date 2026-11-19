```hypothesis
Family: label-lda (Linear Discriminant Analysis). Optdigits pixel counts are approximately Gaussian per class with shared covariance; LDA's generative assumption fits better than logistic regression's discriminative one, yielding higher accuracy at near-identical inference cost (one 64×64 matvec + 10 dot products). If dev accuracy ≤ 0.9613, the Gaussian/shared-covariance assumption is too violated for this ordinal pixel data.
```
```python
"""label-lda: Linear Discriminant Analysis for fixed-size byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Encodes ordinal bytes as z-scores (mean/std from train), fits Gaussian class conditionals
with shared covariance Σ, regularizes Σ += 1e-3*I, computes discriminant weights
W = Σ^{-1} M and biases b_c = -0.5 μ_c^T Σ^{-1} μ_c + log π_c.
Exports exact same weight format as label-linear (H=0): magic, d, 0, C; T[d][256][C]; b1[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
REG = 1e-3


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
    n, d = X.shape

    rng = np.random.default_rng(SEED)

    # per-feature z-score encoding (ordinal only in this task)
    mus = X.mean(axis=0).astype(np.float64)
    sds = X.std(axis=0).astype(np.float64)
    sds[sds == 0] = 1.0
    Z = ((X - mus) / sds).astype(np.float64)

    # class stats
    priors = np.zeros(C, np.float64)
    means = np.zeros((C, d), np.float64)
    for c in range(C):
        mask = (y == c)
        nc = mask.sum()
        priors[c] = nc / n
        means[c] = Z[mask].mean(axis=0)

    # shared covariance
    Sigma = np.zeros((d, d), np.float64)
    for c in range(C):
        diff = Z[y == c] - means[c]
        Sigma += diff.T @ diff
    Sigma /= (n - C)
    Sigma += REG * np.eye(d)

    # discriminant parameters
    Sigma_inv = np.linalg.inv(Sigma)
    W = (Sigma_inv @ means.T).astype(np.float32)          # d x C
    b = np.zeros(C, np.float32)
    for c in range(C):
        mc = means[c].astype(np.float32)
        b[c] = -0.5 * float(mc @ Sigma_inv @ mc) + math.log(priors[c])

    # build lookup tables T[j][v] = ((v - mu_j)/sd_j) * W[j, :]
    T = np.zeros((d, 256, C), np.float32)
    for j in range(d):
        scale = 1.0 / float(sds[j])
        offset = -float(mus[j]) * scale
        for v in range(256):
            z = (v + offset) * scale
            T[j, v, :] = z * W[j, :]

    b1 = b.astype("<f4")
    magic = struct.pack("<4i", 0x54424C31, d, 0, C)
    body = T.astype("<f4").tobytes() + b1.tobytes()
    open(sys.argv[4], "wb").write(magic + body)


if __name__ == "__main__":
    main()
```
```c
/* label-lda inference: ./infer weights.bin in.rec N
 * Weight format identical to label-linear (H=0): magic 0x54424C31, d, 0, C;
 * T[d][256][C] float32, b1[C] float32. Input record = d bytes. Output = 1 byte class. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad weights: H=%d C=%d\n", H, C); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(acc, b1, sizeof(float) * C);
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```