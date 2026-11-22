```hypothesis
Family: label-nbayes (multinomial naive Bayes with per-pixel categorical distributions).
Why: optdigits pixels (0..16) have strong class-conditional differences; naive Bayes captures this with 10*64*17 log-probability tables. Inference is ~640 adds + table lookups per item, far cheaper than label-linear's 64*10=640 multiply-adds plus softmax. Fixed Laplace smoothing (alpha=1) requires no tuning.
Failure signal: dev accuracy below 0.95 (independence assumption too crude).
```
```python
"""Naive Bayes for ordinal pixel features (0..16) with Laplace smoothing.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed fixed), single thread, stdlib + numpy only."""
import json, struct, sys
import numpy as np

ALPHA = 1.0          # Laplace smoothing
SEED = 1             # fixed seed for any randomness (none used, but kept for consistency)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 64
    C = card["classes"]                 # 10
    max_val = 16                        # pixels are 0..16 per card description

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Count class frequencies
    class_counts = np.bincount(y, minlength=C).astype(np.float64)
    # Prior with smoothing
    log_prior = np.log((class_counts + ALPHA) / (n + C * ALPHA)).astype(np.float32)

    # Conditional counts: [C, d, max_val+1]
    cond_counts = np.zeros((C, d, max_val + 1), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if np.any(mask):
            Xc = X[mask]                # (n_c, d)
            # Accumulate counts per pixel per value
            for j in range(d):
                # values are 0..16
                vals = Xc[:, j]
                np.add.at(cond_counts[c, j], vals, 1)

    # Laplace smoothing and normalize to probabilities
    cond_counts += ALPHA
    cond_probs = cond_counts / cond_counts.sum(axis=2, keepdims=True)
    log_cond = np.log(cond_probs).astype(np.float32)   # [C, d, 17]

    # Write weights.bin
    # Magic: 0x4E424159 ('NBAY' as little-endian uint32)
    magic = 0x4E424159
    header = struct.pack("<4I", magic, d, C, max_val)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(log_prior.tobytes())           # C floats
        f.write(log_cond.tobytes())            # C * d * 17 floats

if __name__ == "__main__":
    main()
```
```c
/* Naive Bayes inference for optdigits: ./infer weights.bin in.rec N
 * Each input record: 64 bytes (pixels 0..16). Output: one byte class (0..9). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4E424159) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1], C = (int)hd[2], max_val = (int)hd[3];
    if (d != 64 || C != 10 || max_val != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)C + (long)C * d * (max_val + 1);
    if (nw != 16 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_cond = log_prior + C;   // [C][d][max_val+1]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[10];
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            if (v > max_val) { fprintf(stderr, "pixel value %u > %d\n", v, max_val); return 3; }
            for (int c = 0; c < C; c++) {
                scores[c] += log_cond[((long)c * d + j) * (max_val + 1) + v];
            }
        }
        best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```