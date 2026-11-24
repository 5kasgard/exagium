```hypothesis
Family: label-lda (Linear Discriminant Analysis with shrinkage regularization).
Why: Optdigits pixel counts (0–16) are approximately Gaussian per class with near-shared covariance; LDA is the Bayes-optimal linear classifier under this assumption. Shrinkage (λ=0.01) stabilizes the 64×64 covariance estimate on n=2867 samples. Inference is a single dense matrix-vector multiply (640 mul-adds), matching label-linear's CPU cost while using a stronger inductive bias for this data.
Failure mode: If class covariances differ significantly (requiring QDA) or pixel distributions are far from Gaussian, accuracy will not exceed label-linear's 0.9613.
```
```python
"""Linear Discriminant Analysis with shrinkage for label tasks (fixed-size byte items).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seed), single thread, stdlib + numpy only.
Exports: int32 magic 0x4C444101, d, C; float32 W[C*d] (row-major), b[C].
Inference: scores = x @ W.T + b, argmax."""
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
    C = card["classes"]
    d = X.shape[1]

    # Global z-score (all features are ordinal)
    mu = X.mean(axis=0, dtype=np.float64)
    sigma = X.std(axis=0, dtype=np.float64) + 1e-8
    Z = (X - mu) / sigma  # (n, d) float64

    # Class priors and means
    pi = np.zeros(C, np.float64)
    M = np.zeros((C, d), np.float64)
    for c in range(C):
        mask = (y == c)
        pi[c] = mask.mean()
        M[c] = Z[mask].mean(axis=0)

    # Within-class scatter matrix
    S_W = np.zeros((d, d), np.float64)
    for c in range(C):
        Zc = Z[y == c] - M[c]
        S_W += Zc.T @ Zc

    n = X.shape[0]
    # Shrinkage regularization
    lam = 0.01
    tr = np.trace(S_W)
    Sigma = (S_W + lam * (tr / d) * np.eye(d)) / (n - C)

    # Solve Sigma * A = I  =>  A = Sigma^{-1}
    A = np.linalg.solve(Sigma, np.eye(d))  # (d, d)

    # LDA weights on z-scored space: W_lda = M @ A  (C, d)
    W_lda = M @ A
    # Biases: b_lda[c] = -0.5 * M[c] @ W_lda[c] + log(pi[c])
    b_lda = -0.5 * np.sum(M * W_lda, axis=1) + np.log(pi)

    # Absorb z-scoring: W_final = W_lda / sigma, b_final = b_lda - mu @ W_final.T
    W_final = (W_lda / sigma).astype(np.float32)  # (C, d)
    b_final = (b_lda - mu @ W_final.T).astype(np.float32)  # (C,)

    # Pack weights
    magic = 0x4C444101  # "LDA\1"
    hdr = struct.pack("<3i", magic, d, C)
    body = W_final.astype("<f4").tobytes() + b_final.astype("<f4").tobytes()
    expected = 12 + 4 * (C * d + C)
    if len(hdr) + len(body) != expected:
        raise RuntimeError(f"weight size mismatch: {len(hdr)+len(body)} != {expected}")
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```
```c
/* Linear Discriminant Analysis inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output record: 1 byte (class 0..9).
 * Weights format: int32 magic(0x4C444101), d, C; float32 W[C*d] row-major, b[C]. */
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
    if (nw < 12 || hd[0] != 0x4C444101) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d < 1 || nw != 12 + 4L * ((long)C * d + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 3);
    const float *b = W + (long)C * d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            float xj = (float)xi[j];
            for (int c = 0; c < C; c++) scores[c] += xj * W[(long)c * d + j];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```