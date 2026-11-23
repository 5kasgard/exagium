```hypothesis
Family: label-qda (Quadratic Discriminant Analysis with diagonal covariance).
Why: Optdigits pixels have different variances per class (e.g., ink density varies by digit). QDA models per-class mean and variance, giving a quadratic decision boundary that can separate classes better than linear logistic regression. Inference cost is similar to label-linear (10*64 multiply-adds) because the quadratic term is just a weighted squared distance. Training is deterministic (class statistics) and requires no hyperparameters.
Failure signal: If dev accuracy does not exceed label-linear (0.9613), the diagonal covariance assumption is too restrictive for this dataset.
```

```python
import json, struct, sys
import numpy as np

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
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

    priors = np.zeros(C, dtype=np.float64)
    means = np.zeros((C, d), dtype=np.float64)
    vars_ = np.zeros((C, d), dtype=np.float64)

    for c in range(C):
        mask = (y == c)
        cnt = mask.sum()
        priors[c] = cnt
        if cnt > 0:
            Xc = X[mask].astype(np.float64)
            means[c] = Xc.mean(axis=0)
            vars_[c] = Xc.var(axis=0) + 1e-9  # smoothing
        else:
            means[c] = 0.0
            vars_[c] = 1.0

    priors = priors / priors.sum()
    log_prior = np.log(priors + 1e-9)
    inv_var = 1.0 / vars_
    log_var = np.log(vars_)
    sum_log_var = log_var.sum(axis=1)

    magic = 0x51444131  # 'QDA1'
    out = struct.pack("<4i", magic, d, C, 0)
    out += log_prior.astype(np.float32).tobytes()
    out += sum_log_var.astype(np.float32).tobytes()
    out += means.astype(np.float32).tobytes()
    out += inv_var.astype(np.float32).tobytes()

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
    if (nw < 16 || hd[0] != 0x51444131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * C * 2 + 4L * C * d * 2;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *sum_log_var = log_prior + C;
    const float *mean = sum_log_var + C;
    const float *inv_var = mean + (long)C * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            float s = log_prior[c] - 0.5f * sum_log_var[c];
            const float *mc = mean + (long)c * d;
            const float *ivc = inv_var + (long)c * d;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - mc[j];
                s -= 0.5f * diff * diff * ivc[j];
            }
            scores[c] = s;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```