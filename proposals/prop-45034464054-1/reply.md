```hypothesis
Family: label-nb (Naive Bayes with per-pixel histograms). Optdigits pixels are ordinal 0..16 and conditionally near-independent given the digit class, so a generative model with 64×17×10 log-probability tables should match or exceed linear accuracy. Inference is 640 table lookups + adds (same pattern as label-linear) but weights are 43 KiB vs 655 KiB, so better cache behaviour and lower instruction count. Failure: dev accuracy ≤ label-linear (0.9613) or Ir/item ≥ label-linear (5059).
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
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
    V = 17                              # pixel values 0..16

    # Count occurrences: count[c, j, v] = number of samples of class c with pixel j == v
    count = np.zeros((C, d, V), dtype=np.int64)
    class_count = np.zeros(C, dtype=np.int64)
    for idx in range(len(y)):
        c = y[idx]
        class_count[c] += 1
        for j in range(d):
            v = X[idx, j]
            count[c, j, v] += 1

    # Log priors
    log_prior = np.log(class_count / class_count.sum()).astype(np.float32)

    # Log likelihoods with Laplace smoothing: log((count + ALPHA) / (class_count + ALPHA * V))
    denom = class_count[:, None, None] + ALPHA * V  # shape (C, 1, 1)
    log_prob = np.log((count + ALPHA) / denom).astype(np.float32)  # shape (C, d, V)

    # Transpose to (d, V, C) for cache-friendly inference: log_prob[j, v, c]
    log_prob = np.transpose(log_prob, (1, 2, 0))

    # Write weights.bin
    # Magic: 0x4E425F31 ("NB_1" in little endian), d, C, V
    # Then log_prior[C], log_prob[d][V][C] (row-major: d major, then V, then C)
    magic = 0x4E425F31
    expected_size = 16 + 4 * C + 4 * d * V * C
    out = struct.pack("<4i", magic, d, C, V)
    out += log_prior.tobytes()
    out += log_prob.tobytes()
    assert len(out) == expected_size, f"size mismatch {len(out)} != {expected_size}"
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
    if (nw < 16 || hd[0] != 0x4E425F31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], V = hd[3];
    if (C < 1 || C > 256 || V != 17 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * C + 4L * d * V * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob  = log_prior + C;  // layout [d][V][C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) score[c] = log_prior[c];

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v >= V) { fprintf(stderr, "pixel value %d >= %d\n", v, V); return 3; }
            const float *tp = log_prob + ((long)j * V + v) * C;
            for (int c = 0; c < C; c++) score[c] += tp[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```