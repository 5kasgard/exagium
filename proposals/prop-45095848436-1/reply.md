```hypothesis
Family: label-nb (Multinomial Naive Bayes with per-pixel 0..255 log-likelihood tables).
Why: Optdigits pixels are ordinal 0..16; generative per-class pixel distributions capture independent
evidence well and are extremely fast at inference (640 add+lookup per item). With Laplace smoothing
it often beats linear logistic regression on this classic dataset.
Failure sign: Dev accuracy <= label-linear (0.9613) or inference cost >= label-linear.
```
```python
"""Multinomial Naive Bayes for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (no RNG), single thread. Exports log-prior and per-class per-pixel
log-likelihood tables for all 256 byte values (Laplace smoothing alpha=1)."""
import json, struct, sys
import numpy as np

ALPHA = 1.0


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

    # Count occurrences per class
    class_counts = np.bincount(y, minlength=C).astype(np.float64)
    log_prior = np.log(class_counts / class_counts.sum()).astype(np.float32)

    # Count per class, per pixel, per value
    # shape (C, d, V)
    counts = np.zeros((C, d, V), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if not mask.any():
            continue
        Xc = X[mask]
        # Vectorized counting: for each pixel, bincount
        for j in range(d):
            counts[c, j] = np.bincount(Xc[:, j], minlength=V)

    # Laplace smoothing
    # prob = (count + alpha) / (class_count + alpha * V)
    denom = class_counts[:, None, None] + ALPHA * V
    probs = (counts + ALPHA) / denom
    log_lik = np.log(probs).astype(np.float32)  # (C, d, V)

    # Pack: magic, d, C, V, log_prior[C], log_lik[C, d, V]
    magic = 0x4E424C31  # 'NBL1'
    out = struct.pack("<4i", magic, d, C, V)
    out += log_prior.tobytes()
    out += log_lik.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Multinomial Naive Bayes for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic 0x4E424C31, d, C, V; float32 log_prior[C]; float32 log_lik[C][d][V]. */
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
    if (C < 1 || C > 256 || V != 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * C + 4L * C * d * V;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_lik = log_prior + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = log_prior[c];

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *row = log_lik + ((long)j * V + v);
            for (int c = 0; c < C; c++) scores[c] += row[c * d * V];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```