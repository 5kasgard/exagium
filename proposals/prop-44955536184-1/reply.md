```hypothesis
Family: label-nbayes (Categorical Naive Bayes with Laplace smoothing).
Why: optdigits pixels are discrete 0–16; NB models per-class per-pixel histograms, capturing non-linear feature distributions that linear models miss, while inference is pure table lookups (no multiplies) so it should be faster than label-linear. 
Failure: if dev accuracy ≤ 0.9613 (label-linear), the independence assumption is too crude for this data.
```

```python
"""Categorical Naive Bayes for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (counts only), single thread, stdlib + numpy.
Exports: magic 0x4E424159, d, C, V=256; float32 log_prior[C]; float32 log_prob[C][d][V]."""
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Count class frequencies
    class_count = np.bincount(y, minlength=C).astype(np.float64)
    log_prior = np.log((class_count + ALPHA) / (n + ALPHA * C)).astype(np.float32)

    # Count per-class, per-feature, per-value
    # Shape: (C, d, V)
    counts = np.zeros((C, d, V), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if not mask.any():
            continue
        Xc = X[mask]
        for j in range(d):
            # bincount on column j
            vals = Xc[:, j]
            bc = np.bincount(vals, minlength=V)
            counts[c, j] = bc

    # Laplace smoothing and log
    # (count + alpha) / (class_count + alpha * V)
    denom = class_count[:, None, None] + ALPHA * V
    probs = (counts + ALPHA) / denom
    log_prob = np.log(probs).astype(np.float32)  # (C, d, V)

    # Pack weights: magic, d, C, V, log_prior[C], log_prob[C*d*V]
    magic = 0x4E424159  # "NBAY" little-endian
    hdr = struct.pack("<4i", magic, d, C, V)
    out = hdr + log_prior.tobytes() + log_prob.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Categorical Naive Bayes: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (argmax log-posterior). */
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
    if (d <= 0 || C <= 0 || C > 256 || V != 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * C + 4L * C * d * V;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob  = log_prior + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *score = malloc(sizeof(float) * C);
    if (!score) { fprintf(stderr, "oom\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) score[c] = log_prior[c];

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lp = log_prob + ((long)c * d + j) * V;  // wait, c is not defined here
            // Need to loop over classes inside
        }
        // Fix: loop classes inside features
        for (int c = 0; c < C; c++) {
            float s = log_prior[c];
            for (int j = 0; j < d; j++) {
                unsigned char v = xi[j];
                s += log_prob[((long)c * d + j) * V + v];
            }
            score[c] = s;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(score);
    return 0;
}
```