```hypothesis
Family: label-qda (diagonal Quadratic Discriminant Analysis with global standardization).
Why: Optdigits pixels are ordinal counts 0–16; per-class variances differ (e.g., digit 1 has low variance in many pixels). QDA captures this with quadratic boundaries while remaining extremely fast (O(C*d) per item). Global standardization matches the linear baseline's preprocessing.
Failure indicator: dev accuracy ≤ label-linear (0.9613) or inference cost ≥ label-linear.
```

```python
import json, struct, sys
import numpy as np

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    # Global standardization (same as label-linear's ordinal encoding)
    gmean = X.mean(axis=0, dtype=np.float64)
    gstd = X.std(axis=0, dtype=np.float64)
    gstd[gstd == 0] = 1.0
    Xz = (X - gmean) / gstd

    prior = np.zeros(C, dtype=np.float64)
    cmean = np.zeros((C, d), dtype=np.float64)
    cvar = np.zeros((C, d), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        n_c = mask.sum()
        prior[c] = n_c / len(y)
        if n_c > 0:
            cmean[c] = Xz[mask].mean(axis=0)
            cvar[c] = Xz[mask].var(axis=0) + 1e-3
        else:
            cmean[c] = 0.0
            cvar[c] = 1.0

    # Write weights.bin: magic, d, C, global_mean[d], global_std[d], prior[C], class_mean[C*d], class_var[C*d]
    magic = 0x5144414C  # "LADQ" little-endian
    out = struct.pack("<3i", magic, d, C)
    out += gmean.astype(np.float32).tobytes()
    out += gstd.astype(np.float32).tobytes()
    out += prior.astype(np.float32).tobytes()
    out += cmean.astype(np.float32).tobytes()
    out += cvar.astype(np.float32).tobytes()
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x5144414C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 12 + 4L * (2L*d + C + 2L*C*d);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *gmean = (const float *)(hd + 3);
    const float *gstd  = gmean + d;
    const float *prior = gstd + d;
    const float *cmean = prior + C;
    const float *cvar  = cmean + (long)C * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float x_std[64];
    float log_prior[10];
    for (int c = 0; c < C; c++) log_prior[c] = logf(prior[c]);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) x_std[j] = (xi[j] - gmean[j]) / gstd[j];

        float best_score = -1e30f; unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            const float *cm = cmean + (long)c * d;
            const float *cv = cvar  + (long)c * d;
            for (int j = 0; j < d; j++) {
                float diff = x_std[j] - cm[j];
                float v = cv[j];
                sum += logf(v) + diff * diff / v;
            }
            float score = log_prior[c] - 0.5f * sum;
            if (score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```