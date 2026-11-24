```hypothesis
We propose a Naive Bayes family (label-nb) that models each pixel independently per class with a categorical distribution over 0..16 (Laplace-smoothed). Optdigits pixels are known to be highly discriminative and near-independent, so Naive Bayes should beat the linear baseline (0.9613) at a fraction of the CPU cost. If it fails to surpass label-linear, it would indicate that pixel dependencies are critical for this task and the independence assumption is too severe.
```
```python
"""Naive Bayes for label tasks with fixed-size byte items (ordinal pixels 0..16).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-threaded, stdlib+numpy only.
Weights format: int32 magic 0x4E424159, C, d, B; float32 log_prior[C]; float32 log_lik[C][d][B]."""
import json, struct, sys

import numpy as np

MAGIC = 0x4E424159  # 'NBAY' in little-endian
BINS = 17           # pixel values 0..16


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
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == X.shape[1] == 64

    # Count with Laplace smoothing (add 1 to each bin)
    counts = np.ones((C, d, BINS), dtype=np.int64)
    class_counts = np.ones(C, dtype=np.int64)

    for xi, yi in zip(X, y):
        class_counts[yi] += 1
        for j, v in enumerate(xi):
            counts[yi, j, v] += 1

    # Convert to log probabilities
    log_prior = np.log(class_counts / class_counts.sum()).astype(np.float32)
    # log_lik[class, feature, value] = log(count / class_count)
    log_lik = np.log(counts / class_counts[:, None, None]).astype(np.float32)

    # Pack weights
    header = struct.pack("<4i", MAGIC, C, d, BINS)
    body = log_prior.tobytes() + log_lik.tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* Naive Bayes inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes (pixels 0..16); output is one byte class (argmax log posterior). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4E424159) { fprintf(stderr, "bad magic\n"); return 3; }
    int C = hd[1], d = hd[2], B = hd[3];
    if (C < 1 || C > 256 || d != 64 || B != 17) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + (long)C * 4 + (long)C * d * B * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(wb + 16);
    const float *log_lik = log_prior + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *lik = log_lik + ((long)c * d + j) * B; // will set c in loop
            for (int c = 0; c < C; c++) {
                scores[c] += log_lik[((long)c * d + j) * B + v];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```