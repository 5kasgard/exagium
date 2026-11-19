```hypothesis
Family: label-nb (Naive Bayes with per-feature 256-bin log-likelihood tables).
Why: The input is already quantized to 256 bins per column. Naive Bayes makes no monotonicity assumption (unlike linear on ordinal z-scores), captures arbitrary per-feature non-linearity, and is extremely fast at inference (14 table lookups + adds per item). It should beat label-linear's 0.8524 on Adult where relationships like age/hours-per-week vs income are non-monotonic.
Fail signal: Dev accuracy ≤ 0.8524 or Ir/item ≥ 719 (label-linear's cost).
```
```python
"""Naive Bayes for tabular label tasks with 256-bin features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 0), single thread, stdlib + numpy only.
Weights: magic 0x4E424159, d, C; float32 log_prior[C], log_prob[C][d][256]."""
import json, struct, sys
import numpy as np

SEED = 0
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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Count occurrences: count[c, j, v]
    count = np.zeros((C, d, 256), dtype=np.int64)
    for c in range(C):
        mask = (y == c)
        if not mask.any():
            continue
        Xc = X[mask]
        for j in range(d):
            np.add.at(count[c, j], Xc[:, j], 1)

    # Log probabilities with Laplace smoothing
    # log P(x_j=v | y=c) = log(count + alpha) - log(total_c + alpha*256)
    total_c = count.sum(axis=(1, 2), keepdims=True)  # (C,1,1)
    log_prob = np.log(count.astype(np.float64) + ALPHA) - np.log(total_c + ALPHA * 256.0)

    # Log prior: log P(y=c)
    prior = np.bincount(y, minlength=C).astype(np.float64)
    log_prior = np.log(prior + ALPHA) - np.log(n + ALPHA * C)

    # Export: magic, d, C; log_prior[C]; log_prob[C][d][256] (all float32 LE)
    out = struct.pack("<3i", 0x4E424159, d, C)
    out += log_prior.astype("<f4").tobytes()
    out += log_prob.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Naive Bayes inference for 256-bin features: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: 1 byte class (argmax log-posterior). */
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
    if (nw < 12 || hd[0] != 0x4E424159) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || nw != 12 + 4L * (C + (long)C * d * 256)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *log_prior = (const float *)(hd + 3);
    const float *log_prob = log_prior + C;  // [C][d][256]
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_score = -1e30f; unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            float score = log_prior[c];
            for (int j = 0; j < d; j++) {
                score += log_prob[((long)c * d + j) * 256 + xi[j]];
            }
            if (score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```