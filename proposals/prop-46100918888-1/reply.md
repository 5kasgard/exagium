```hypothesis
Family: label-nb (Multinomial Naive Bayes with Laplace smoothing).
Why: Optdigits pixels are ordinal 0–16; treating each as conditionally independent given the class is a classic strong baseline for digit recognition. NB requires only additions at inference, so it should beat label-linear (96.1%) and approach label-mlp16 (97.7%) at a fraction of the CPU cost.
Failure sign: If conditional independence is too violated, dev accuracy stays below 96%.
```

```python
"""Multinomial Naive Bayes for fixed-size byte items (ordinal 0..16).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-threaded, stdlib+numpy only.
Weights format: int32 magic 0x4E424C31, d, C, V; float32 log_prior[C]; float32 log_prob[C][d][V]."""
import json, struct, sys
import numpy as np

MAGIC = 0x4E424C31  # 'NBL1' little-endian
V = 17  # pixel values 0..16

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

    # Counts with Laplace smoothing (add-1)
    class_counts = np.ones(C, dtype=np.int64)
    pixel_counts = np.ones((C, d, V), dtype=np.int64)

    for i in range(n):
        c = y[i]
        class_counts[c] += 1
        for j in range(d):
            v = X[i, j]
            pixel_counts[c, j, v] += 1

    # Log probabilities
    log_prior = np.log(class_counts.astype(np.float64) / class_counts.sum())
    # Normalize per class per pixel
    pixel_sums = pixel_counts.sum(axis=2, keepdims=True)  # (C, d, 1)
    log_prob = np.log(pixel_counts.astype(np.float64) / pixel_sums)

    # Write weights
    out = struct.pack("<4i", MAGIC, d, C, V)
    out += log_prior.astype("<f4").tobytes()
    out += log_prob.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Naive Bayes inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output: one byte class (argmax log posterior). */
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
    if (C < 1 || C > 256 || V != 17 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * C + 4L * C * d * V;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob = log_prior + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *lp = log_prob + ((long)c * d + j) * V; // wait, c loop inside? We'll restructure.
        }
        // Better: loop over classes inside pixel loop for cache? But C=10 small. Let's do class outer.
        for (int c = 0; c < C; c++) {
            float sum = log_prior[c];
            for (int j = 0; j < d; j++) {
                int v = xi[j];
                sum += log_prob[((long)c * d + j) * V + v];
            }
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```