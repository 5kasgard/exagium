```hypothesis
Family: label-nb (Naive Bayes with per-position per-class Laplace smoothing).
Why: Character-level language modeling benefits from generative modeling of P(context|next_char); NB captures strong positional dependencies (e.g., 'q'→'u') with zero hyperparameters, one-pass counting, and inference cost identical to linear (16×65 additions). It often outperforms discriminative linear models on small-text tasks.
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or inference instructions/item ≥ label-linear.
```
```python
"""Naive Bayes for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed not needed), single thread, stdlib+numpy only.
Exports: int32 magic 0x4E424159, d, C, V; float32 log_prior[C]; float32 log_prob[d][C][V]."""
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
    d = X.shape[1]
    C = card["classes"]
    V = 256

    # Prior with Laplace smoothing
    prior = np.bincount(y, minlength=C).astype(np.float64) + 1.0
    log_prior = np.log(prior / prior.sum()).astype(np.float32)

    # Counts per position, class, value with Laplace smoothing
    counts = np.ones((d, C, V), dtype=np.float64)
    for j in range(d):
        # Vectorized counting using np.add.at
        np.add.at(counts[j], (y, X[:, j]), 1)

    # Normalize and log
    counts /= counts.sum(axis=2, keepdims=True)
    log_prob = np.log(counts).astype(np.float32)  # shape (d, C, V)

    # Pack weights
    magic = 0x4E424159  # 'NBAY'
    hdr = struct.pack("<4i", magic, d, C, V)
    body = log_prior.tobytes() + log_prob.tobytes()
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```
```c
/* Naive Bayes for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax log posterior). */
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
    if (d != 16 || C != 65 || V != 256) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L + 4L * C + 4L * d * C * V;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob = log_prior + C;  // [d][C][V]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(score, log_prior, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *lp = log_prob + ((long)j * C + 0) * V + xi[j];
            for (int c = 0; c < C; c++) score[c] += lp[c * V];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```