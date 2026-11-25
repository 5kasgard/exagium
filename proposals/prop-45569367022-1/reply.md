```hypothesis
Family: label-nb (Naive Bayes with per-pixel per-class histograms).
Why: Optdigits features are ordinal pixel counts 0–16; Naive Bayes models each pixel's distribution per digit directly, capturing non-linear stroke patterns that linear models miss. It's extremely fast at inference (only additions and table lookups) and often beats logistic regression on small discrete-feature datasets.
Failure indicator: dev accuracy ≤ 0.9613 (label-linear) or inference cost > label-linear.
```
```python
"""Naive Bayes for label tasks with fixed-size byte items (ordinal features).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-threaded, stdlib + numpy only.
Exports: magic 0x4E424C31, d, C, V=256; float32 log_prior[C]; float32 log_prob[d][V][C] (row-major)."""
import json, struct, sys
import numpy as np

ALPHA = 1.0  # Laplace smoothing
V = 256      # byte values 0..255

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

    # Count occurrences
    count_y = np.zeros(C, dtype=np.int64)
    count_feat = np.zeros((d, V, C), dtype=np.int64)
    for i in range(n):
        yi = y[i]
        count_y[yi] += 1
        xi = X[i]
        for j in range(d):
            count_feat[j, xi[j], yi] += 1

    # Log probabilities with Laplace smoothing
    log_prior = np.log((count_y + ALPHA) / (n + ALPHA * C)).astype(np.float32)
    # log_prob[j, v, c] = log( (count_feat[j,v,c] + ALPHA) / (count_y[c] + ALPHA * V) )
    denom = (count_y + ALPHA * V).astype(np.float32)  # shape (C,)
    log_prob = np.log((count_feat + ALPHA) / denom).astype(np.float32)  # (d, V, C)

    # Write weights.bin
    magic = 0x4E424C31  # 'NBL1'
    out = struct.pack("<4i", magic, d, C, V)
    out += log_prior.tobytes()
    out += log_prob.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Naive Bayes for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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
    if (nw < 16 || hd[0] != 0x4E424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], V = hd[3];
    if (C < 1 || C > 256 || V != 256 || d < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * C + 4L * (long)d * V * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob = (const float *)(log_prior + C);  // [d][V][C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *lp = log_prob + ((long)j * V + v) * C;
            for (int c = 0; c < C; c++) scores[c] += lp[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```