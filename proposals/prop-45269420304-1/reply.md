```hypothesis
Family: label-nb (Gaussian Naive Bayes on globally z-scored pixels).
Why: Optdigits pixels are ordinal (0..16) with clear per-class intensity patterns. NB models per-class variance per pixel, capturing info linear logreg discards, at similar inference cost (~2k ops vs 5k). The generative approach often excels on small-digit tasks.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or inference cost ≥ 2× label-linear.
```
```python
"""label-nb: Gaussian Naive Bayes on globally z-scored ordinal bytes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  uint32 magic=0x4E424159 ('NBAY'), int32 d, int32 C,
  float32 global_mu[d], global_std[d],
  float32 class_mu[C][d], class_var[C][d], log_prior[C]."""
import json, struct, sys
import numpy as np

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
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Global z-score per pixel (ordinal only per card)
    Xf = X.astype(np.float64)
    gmu = Xf.mean(axis=0)
    gstd = Xf.std(axis=0)
    gstd[gstd == 0] = 1.0
    Z = (Xf - gmu) / gstd  # (n, d)

    # Per-class mean & variance of z-scored features
    cmu = np.zeros((C, d), np.float64)
    cvar = np.zeros((C, d), np.float64)
    prior = np.zeros(C, np.float64)
    for c in range(C):
        mask = (y == c)
        n_c = mask.sum()
        prior[c] = n_c
        if n_c > 0:
            Zc = Z[mask]
            cmu[c] = Zc.mean(axis=0)
            cvar[c] = Zc.var(axis=0)
    cvar[cvar < 1e-6] = 1e-6
    log_prior = np.log(prior / prior.sum())

    # Pack weights
    magic = 0x4E424159  # 'NBAY'
    out = struct.pack("<3i", magic, d, C)
    out += gmu.astype("<f4").tobytes()
    out += gstd.astype("<f4").tobytes()
    out += cmu.astype("<f4").tobytes()
    out += cvar.astype("<f4").tobytes()
    out += log_prior.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-nb infer: ./infer weights.bin in.rec N
 * Gaussian Naive Bayes on globally z-scored ordinal bytes. */
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
    uint32_t *hd = (uint32_t *)wb;
    if (nw < 12 || hd[0] != 0x4E424159) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = (int)hd[1], C = (int)hd[2];
    if (d != 64 || C != 10) { fprintf(stderr, "unexpected d/C\n"); return 3; }
    size_t expected = 12 + 4L*d*2 + 4L*C*d*2 + 4L*C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *gmu = (const float *)(wb + 12);
    const float *gstd = gmu + d;
    const float *cmu = gstd + d;
    const float *cvar = cmu + (long)C * d;
    const float *log_prior = cvar + (long)C * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    double *score = malloc(sizeof(double) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) z[j] = (xi[j] - gmu[j]) / gstd[j];
        for (int c = 0; c < C; c++) {
            double sc = log_prior[c];
            const float *mu = cmu + (long)c * d;
            const float *var = cvar + (long)c * d;
            for (int j = 0; j < d; j++) {
                double diff = z[j] - mu[j];
                sc += -0.5 * log(var[j]) - 0.5 * diff * diff / var[j];
            }
            score[c] = sc;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```