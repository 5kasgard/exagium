```hypothesis
Family: label-nb (Multinomial Naive Bayes with Laplace smoothing).
Why: Optdigits features are 64 ordinal pixel counts (0..16). NB models each feature independently per class, which suits the small discrete domain and yields a tiny, fast model (10*64*17 log-probs). It often beats linear models on this dataset because the conditional independence assumption is less harmful than the linear decision boundary when pixel correlations are weak.
Failure signal: Dev accuracy ≤ 0.9613 (label-linear) or inference cost ≥ label-linear.
```
```python
"""Multinomial Naive Bayes for fixed-size ordinal byte features (0..16).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-threaded, stdlib+numpy only.
Weights format: magic 0x4E424159 ('NBAY'), int32 d, C, V; float32 log_prior[C]; float32 log_likelihood[C][d][V]."""
import json, struct, sys
import numpy as np

ALPHA = 1.0  # Laplace smoothing

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
    d = card["input"]["bytes"]          # 64
    C = card["classes"]                 # 10
    V = 17                              # values 0..16 inclusive

    # Count feature values per class
    # shape (C, d, V)
    counts = np.zeros((C, d, V), dtype=np.int64)
    class_counts = np.zeros(C, dtype=np.int64)
    for xi, yi in zip(X, y):
        class_counts[yi] += 1
        counts[yi, np.arange(d), xi] += 1

    # Laplace smoothing
    # P(x_j=v | c) = (counts[c,j,v] + ALPHA) / (class_counts[c] + V * ALPHA)
    denom = class_counts[:, None, None] + V * ALPHA
    probs = (counts + ALPHA) / denom
    log_likelihood = np.log(probs, dtype=np.float32)  # (C, d, V)

    # Class priors with smoothing
    prior = (class_counts + ALPHA) / (class_counts.sum() + C * ALPHA)
    log_prior = np.log(prior, dtype=np.float32)       # (C,)

    # Write weights
    magic = 0x4E424159  # 'NBAY'
    out = struct.pack("<4i", magic, d, C, V)
    out += log_prior.astype("<f4").tobytes()
    out += log_likelihood.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Multinomial Naive Bayes inference for ordinal byte features.
 * ./infer weights.bin in.rec N
 * Each input record: d bytes (pixels 0..16). Output: one byte class (0..9). */
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
    if (nw < 16 || hd[0] != 0x4E424159) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], V = hd[3];
    if (d != 64 || C != 10 || V != 17) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16 + 4L * (C + (long)C * d * V);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_likelihood = log_prior + C;  // [C][d][V]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[10];
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v >= V) v = V - 1;  // clamp, though data should be 0..16
            const float *ll = log_likelihood + ((long)c * d + j) * V;  // wait, c not defined here
            // We need to add for each class. Better loop classes inside.
        }
        // Recompute with class loop outside for cache? d=64, C=10, V=17 small.
        for (int c = 0; c < C; c++) {
            float sum = log_prior[c];
            for (int j = 0; j < d; j++) {
                int v = xi[j];
                sum += log_likelihood[((long)c * d + j) * V + v];
            }
            scores[c] = sum;
        }

        best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```