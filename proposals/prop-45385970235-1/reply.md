```hypothesis
Family: label-nbayes (categorical Naive Bayes with per-pixel 17-bin histograms).
Why: 8x8 digits have 0..16 pixel intensities; generative per-class histograms capture stroke thickness well, and conditional independence is plausible at this resolution. No hyperparameters, deterministic O(N) training, tiny weights, inference ~640 adds/item.
Failure: If dev accuracy < 0.955, the independence assumption loses too much spatial structure vs. linear baseline (0.9613).
```

```python
import json, struct, sys
import numpy as np

MAGIC = 0x4E424159  # 'N' 'B' 'A' 'Y' little-endian
V = 17              # pixel values 0..16
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

    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 64 and C == 10 and X.max() <= 16

    n = X.shape[0]
    rng = np.random.default_rng(SEED)  # fixed seed for determinism contract

    # Class priors with Laplace smoothing
    class_counts = np.bincount(y, minlength=C).astype(np.float64)
    log_prior = np.log(class_counts / n).astype(np.float32)

    # Counts per class, per pixel, per value (add-1 smoothing)
    counts = np.ones((C, d, V), dtype=np.float64)
    for i in range(n):
        c = y[i]
        xi = X[i]
        for j in range(d):
            counts[c, j, xi[j]] += 1

    # Normalize to log probabilities
    denom = class_counts[:, None] + V
    log_prob = np.log(counts / denom[:, :, None]).astype(np.float32)  # (C, d, V)

    # Write weights: magic, d, C, V, log_prior[C], log_prob[C*d*V]
    out = struct.pack("<4i", MAGIC, d, C, V)
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
    if (d != 64 || C != 10 || V != 17) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * C + 4L * C * d * V;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob = log_prior + C;  // [C][d][V] row-major

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            if (v >= V) v = V - 1;
            for (int c = 0; c < C; c++) {
                scores[c] += log_prob[((long)c * d + j) * V + v];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```