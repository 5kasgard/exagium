```hypothesis
label-nb: Naive Bayes with per-pixel categorical distributions (17 bins) and Laplace smoothing.
Optdigits pixels are 0..16 counts; NB models each pixel's class-conditional distribution independently.
This captures non-linear per-pixel evidence that linear models miss, with tiny weights and fast inference.
If dev accuracy does not exceed label-linear (0.9613), the independence assumption is too strong for this data.
```

```python
import json, struct, sys

import numpy as np


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 64
    C = card["classes"]         # 10
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    num_bins = 17  # pixel values 0..16

    class_counts = np.bincount(y, minlength=C).astype(np.float64)
    log_prior = np.log(class_counts / n).astype("<f4")

    # conditional counts with Laplace smoothing
    cond_counts = np.ones((C, d, num_bins), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if np.any(mask):
            Xc = X[mask]
            for j in range(d):
                vals = Xc[:, j]
                bc = np.bincount(vals, minlength=num_bins)
                cond_counts[c, j] += bc

    cond_probs = cond_counts / cond_counts.sum(axis=2, keepdims=True)
    log_prob = np.log(cond_probs).astype("<f4")  # shape (C, d, num_bins)

    magic = 0x4E424C31  # "NLB1"
    out = struct.pack("<4i", magic, d, C, num_bins)
    out += log_prior.tobytes()
    out += log_prob.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0;
    long N = atol(argv[3]);
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4E424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], num_bins = hd[3];
    if (d != 64 || C != 10 || num_bins != 17) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L*C + 4L*C*d*num_bins;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob = (const float *)(log_prior + C);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;
    float *scores = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            if (v >= num_bins) { fprintf(stderr, "pixel value %d out of range\n", v); return 3; }
            for (int c = 0; c < C; c++) {
                scores[c] += log_prob[((long)c * d + j) * num_bins + v];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```