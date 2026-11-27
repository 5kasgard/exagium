```hypothesis
Family: label-gnb (Gaussian Naive Bayes with full 256-value lookup tables).
Why: Optdigits pixels are ordinal 0–16 and approximately conditionally independent given the digit; GNB models per-class per-pixel variance, which linear logistic regression cannot. The 256-entry lookup tables make inference a simple sum of precomputed log-probabilities (≈640 adds/item), so it should beat label-linear’s 0.9613 at similar or lower CPU cost.
Failure signal: dev accuracy ≤ 0.9613 or instructions/item ≥ label-linear’s 5059.
```
```python
"""Gaussian Naive Bayes with per-class per-feature 256-entry log-probability lookup tables.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (no RNG needed), single thread, stdlib + numpy only.
Weights format (little-endian):
  int32 magic=0x474E4142 ('BANG'), int32 d, int32 C,
  float32 log_prior[C],
  float32 log_prob[C][d][256]  (row-major: class, then feature, then value 0..255)."""
import json, struct, sys
import numpy as np

SMOOTH = 1e-3  # variance floor


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
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    # Class priors
    prior = np.bincount(y, minlength=C).astype(np.float64)
    log_prior = np.log(prior / n).astype(np.float32)

    # Per-class per-feature mean and variance
    mean = np.zeros((C, d), np.float64)
    var = np.zeros((C, d), np.float64)
    for c in range(C):
        Xc = X[y == c].astype(np.float64)
        if len(Xc) > 0:
            mean[c] = Xc.mean(axis=0)
            var[c] = Xc.var(axis=0) + SMOOTH
        else:
            mean[c] = 0.0
            var[c] = 1.0

    # Build 256-entry lookup tables: log N(x | mean, var) = -0.5*log(2π*var) - 0.5*(x-mean)^2/var
    vals = np.arange(256, dtype=np.float64)
    log_prob = np.zeros((C, d, 256), np.float32)
    for c in range(C):
        for j in range(d):
            m, v = mean[c, j], var[c, j]
            log_prob[c, j] = (-0.5 * np.log(2 * np.pi * v) - 0.5 * (vals - m) ** 2 / v).astype(np.float32)

    # Write weights
    magic = 0x474E4142  # 'BANG'
    out = struct.pack("<3i", magic, d, C)
    out += log_prior.tobytes()
    out += log_prob.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Gaussian Naive Bayes inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format (little-endian):
 *   int32 magic=0x474E4142, int32 d, int32 C,
 *   float32 log_prior[C],
 *   float32 log_prob[C][d][256]  (row-major). */
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
    if (nw < 12 || hd[0] != 0x474E4142) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 12 + (size_t)C * 4 + (size_t)C * d * 256 * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *log_prior = (const float *)(hd + 3);
    const float *log_prob = log_prior + C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_score = -INFINITY;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            float score = log_prior[c];
            const float *lp = log_prob + ((size_t)c * d * 256);
            for (int j = 0; j < d; j++) {
                score += lp[j * 256 + xi[j]];
            }
            if (score > best_score) {
                best_score = score;
                best_class = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```