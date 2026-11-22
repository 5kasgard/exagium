```hypothesis
Family: label-lda (Linear Discriminant Analysis with ridge regularization).
Why: Optdigits pixels are spatially correlated; LDA models the shared within-class covariance, giving a linear decision boundary that adapts to feature correlations, unlike label-linear's per-pixel z-scoring. Inference is one dot-product per class (640 mul-adds), nearly identical cost to label-linear's table lookups.
Failure: If dev accuracy ≤ 0.9613 (label-linear's score), the covariance structure does not help enough.
```
```python
import json, struct, sys
import numpy as np

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

    # Standardize per feature on training data
    mu_feat = X.mean(axis=0).astype(np.float64)
    sigma_feat = X.std(axis=0).astype(np.float64)
    sigma_feat[sigma_feat == 0] = 1.0
    X_std = (X.astype(np.float64) - mu_feat) / sigma_feat

    # Class priors and standardized means
    priors = np.zeros(C, np.float64)
    mu_std = np.zeros((C, d), np.float64)
    for c in range(C):
        mask = (y == c)
        nc = mask.sum()
        priors[c] = nc / n
        mu_std[c] = X_std[mask].mean(axis=0)

    # Pooled within-class covariance
    S = np.zeros((d, d), np.float64)
    for c in range(C):
        mask = (y == c)
        Xc = X_std[mask] - mu_std[c]
        S += Xc.T @ Xc
    S /= (n - C)

    # Ridge regularization (deterministic, data-dependent)
    lam = 1e-3 * np.trace(S) / d
    S_reg = S + lam * np.eye(d)

    # Invert regularized covariance
    inv_S = np.linalg.inv(S_reg)

    # LDA coefficients in standardized space
    w_std = (inv_S @ mu_std.T).T  # C x d
    b_std = -0.5 * np.sum(mu_std * w_std, axis=1) + np.log(priors)

    # Transform back to raw pixel space: x_std = (x_raw - mu) / sigma
    # score = x_std @ w_std^T + b_std = x_raw @ (w_std/sigma)^T + (b_std - mu @ (w_std/sigma))
    w_raw = w_std / sigma_feat
    b_raw = b_std - np.sum(mu_feat * w_raw, axis=1)

    # Pack weights: magic, d, C, reserved; then W[C][d], b[C] as float32 LE
    magic = 0x4C444131  # "LDA1"
    out = struct.pack("<4i", magic, d, C, 0)
    out += w_raw.astype("<f4").tobytes()
    out += b_raw.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4C444131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d != 64 || nw != 16 + 4L * C * d + 4L * C) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)C * d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            float xv = (float)xi[j];
            for (int c = 0; c < C; c++) scores[c] += xv * W[(long)c * d + j];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```