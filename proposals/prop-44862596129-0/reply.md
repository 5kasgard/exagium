```hypothesis
We propose a "label-naive-bayes" family: a multinomial Naive Bayes with Laplace smoothing over the 256 possible values of each of the 14 byte features. Since the Adult dataset features are already discretized (ordinal quantized to 255 bins, categorical as indices), the independence assumption is plausible and the model requires only counting during training, making it deterministic and extremely fast at inference (just 14 table lookups per class). If it does not beat label-linear in accuracy, it would indicate that feature interactions are essential for this task and the independence assumption is too restrictive.
```
```python
"""Multinomial Naive Bayes for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (no randomness), single thread, stdlib + numpy only.
Exported weights: int32 magic 0x4E424159, C, d, V; float32 log_prior[C]; float32 log_likelihood[C][d][V]."""
import json, struct, sys

import numpy as np

ALPHA = 1.0  # Laplace smoothing
MAGIC = 0x4E424159  # "NBay" in little-endian


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
    V = 256  # each feature is a byte

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Count occurrences: class_counts[C], feature_counts[C, d, V]
    class_counts = np.zeros(C, dtype=np.int64)
    feature_counts = np.zeros((C, d, V), dtype=np.int64)

    for i in range(n):
        c = y[i]
        class_counts[c] += 1
        for j in range(d):
            feature_counts[c, j, X[i, j]] += 1

    # Log prior
    log_prior = np.log(class_counts.astype(np.float64) / n).astype(np.float32)

    # Log likelihood with Laplace smoothing
    # denominator per class: class_counts[c] + ALPHA * V
    denom = (class_counts[:, None, None] + ALPHA * V).astype(np.float64)
    num = (feature_counts + ALPHA).astype(np.float64)
    log_likelihood = np.log(num / denom).astype(np.float32)  # shape (C, d, V)

    # Write weights.bin
    header = struct.pack("<4i", MAGIC, C, d, V)
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(log_prior.tobytes())
        f.write(log_likelihood.tobytes())


if __name__ == "__main__":
    main()
```
```c
/* Naive Bayes inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic 0x4E424159, C, d, V; float32 log_prior[C]; float32 log_likelihood[C][d][V]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    int C = hd[1], d = hd[2], V = hd[3];
    if (C < 1 || C > 256 || d < 1 || V != 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * C + 4L * C * d * V;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_likelihood = log_prior + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                scores[c] += log_likelihood[((long)c * d + j) * V + v];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```