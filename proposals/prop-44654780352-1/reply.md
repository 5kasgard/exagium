```hypothesis
Family: label-nbayes (categorical Naive Bayes with Laplace smoothing).
Why: 64 ordinal pixel features (0–16) are nearly conditionally independent given the digit class; NB models P(x_j|y) directly with tiny lookup tables, yielding fast inference (640 adds/item) and often strong accuracy on this classic dataset.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or inference cost ≥ 5000 Ir/item.
```
```python
"""Categorical Naive Bayes for ordinal byte features (0..V-1).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-thread, stdlib+numpy only.
Exports: magic, d, C, V; float32 log_prior[C]; float32 log_prob[d][V][C] (row-major)."""
import json, struct, sys
import numpy as np

V = 17          # pixel values 0..16
ALPHA = 1.0     # Laplace smoothing
SEED = 1

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
    assert d == 64 and C == 10 and V == 17

    # Class counts
    class_count = np.bincount(y, minlength=C).astype(np.float64)
    log_prior = np.log(class_count / class_count.sum()).astype(np.float32)

    # Feature-value counts per class: shape (d, V, C)
    counts = np.zeros((d, V, C), dtype=np.float64)
    for j in range(d):
        # X[:,j] in 0..16, y in 0..9
        np.add.at(counts[j], (X[:, j], y), 1.0)

    # Laplace smoothing and log
    # P(x_j=v | y=c) = (counts[j,v,c] + ALPHA) / (class_count[c] + ALPHA * V)
    denom = class_count + ALPHA * V          # (C,)
    prob = (counts + ALPHA) / denom[None, None, :]  # (d, V, C)
    log_prob = np.log(prob).astype(np.float32)      # (d, V, C)

    # Write weights.bin
    magic = 0x4E424159  # 'NBAY'
    hdr = struct.pack("<4i", magic, d, C, V)
    body = log_prior.tobytes() + log_prob.ravel(order='C').tobytes()
    open(sys.argv[4], "wb").write(hdr + body)

if __name__ == "__main__":
    main()
```
```c
/* Categorical Naive Bayes inference: ./infer weights.bin in.rec N
 * Input record: d bytes (0..V-1). Output: one byte class (argmax log-posterior). */
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
    if (nw < 16 || hd[0] != 0x4E424159) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], V = hd[3];
    if (C < 1 || C > 256 || V < 1 || V > 256 || d < 1 ||
        nw != 16 + 4L * (C + (long)d * V * C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob = log_prior + C;  // [d][V][C] row-major
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *score = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) score[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v >= V) { fprintf(stderr, "feature %d value %d >= V=%d\n", j, v, V); return 3; }
            const float *lp = log_prob + ((long)j * V + v) * C;
            for (int c = 0; c < C; c++) score[c] += lp[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```