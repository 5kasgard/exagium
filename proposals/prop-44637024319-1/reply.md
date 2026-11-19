```hypothesis
Family: label-nb (categorical Naive Bayes with Laplace smoothing).
Why: The 64 input features are discrete pixel counts (0–16). Modeling each pixel independently per class with a categorical distribution captures the per-class intensity histograms, which are known to be effective for small digit datasets. Inference is just 64 table lookups and additions per class – extremely fast.
Failure signal: If dev accuracy is ≤ 0.9613 (label-linear), the conditional independence assumption is too strong for this data.
```

```python
"""Categorical Naive Bayes for fixed-size byte items (ordinal 0..255).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-threaded, stdlib + numpy only."""
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
    d = card["input"]["bytes"]
    C = card["classes"]
    V = 256          # all possible byte values
    alpha = 1.0      # Laplace smoothing

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # counts
    class_counts = np.bincount(y, minlength=C).astype(np.float64)
    # feature_counts[j, v, c]
    feature_counts = np.zeros((d, V, C), dtype=np.float64)
    for j in range(d):
        # use 2D bincount trick: flatten (v, c) pairs
        vc = X[:, j] * C + y
        cnt = np.bincount(vc, minlength=V * C).reshape(V, C)
        feature_counts[j] = cnt

    # log priors
    log_prior = np.log((class_counts + alpha) / (n + alpha * C)).astype(np.float32)

    # log likelihoods: log( (count + alpha) / (class_count + alpha * V) )
    denom = class_counts + alpha * V          # shape (C,)
    log_prob = np.log((feature_counts + alpha) / denom).astype(np.float32)  # (d, V, C)

    # write weights.bin
    magic = 0x4E424159  # 'NBY' in little-endian
    out = struct.pack("<4I", magic, d, C, V)
    out += log_prior.tobytes()
    out += log_prob.tobytes()   # already (d, V, C) row-major
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Categorical Naive Bayes inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class (argmax log posterior). */
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
    int d = hd[1], C = hd[2], V = hd[3];
    if (C < 1 || C > 256 || V != 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)C * 4 + (long)d * V * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob = log_prior + C;   // [d][V][C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *score = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(score, log_prior, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *lp = log_prob + ((long)j * V + xi[j]) * C;
            for (int c = 0; c < C; c++) score[c] += lp[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```