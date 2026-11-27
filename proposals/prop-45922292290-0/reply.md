```hypothesis
Family: label-naivebayes. The adult dataset uses pre-binned features (255 quantile bins for ordinal, category indices for categorical), making each feature a discrete 8‑bit value. A multinomial Naive Bayes model with Laplace smoothing directly estimates P(feature|class) from these bins. It requires no hyperparameter tuning, trains in one pass, and at inference does only table lookups and additions (2 classes × 14 features = 28 float adds per item). This should beat label-linear’s accuracy because it captures non‑linear per‑feature class distributions, while being far cheaper than label-mlp16 or label-gbdt. If it fails to exceed label-linear, it means feature interactions are essential for this task.
```

```python
import json, struct, sys
import numpy as np

MAGIC = 0x4E424159  # "NBAY" in little-endian bytes

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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Counts with Laplace smoothing (add-1)
    counts = np.ones((C, d, 256), dtype=np.int64)
    for i in range(n):
        xi = X[i]
        yi = y[i]
        for j in range(d):
            counts[yi, j, xi[j]] += 1

    # Log probabilities
    log_probs = np.log(counts / counts.sum(axis=2, keepdims=True)).astype(np.float32)

    # Class prior (with smoothing)
    class_counts = np.bincount(y, minlength=C).astype(np.float64)
    prior = np.log((class_counts + 1) / (n + C)).astype(np.float32)

    # Write weights.bin: magic, d, C, V(=256), then log_probs[C][d][256], then prior[C]
    out = struct.pack("<4i", MAGIC, d, C, 256)
    out += log_probs.tobytes()
    out += prior.tobytes()
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
    if (nw < 16 || hd[0] != 0x4E424159) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], V = hd[3];
    if (C < 1 || C > 256 || d < 1 || d > 256 || V != 256) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16L + (long)C * d * 256 * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_probs = (const float *)(hd + 4);
    const float *prior = log_probs + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = prior[c];
        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            for (int c = 0; c < C; c++) {
                size_t idx = ((size_t)c * d + j) * 256 + v;
                scores[c] += log_probs[idx];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```