```hypothesis
Family: label-bayes (Gaussian Naive Bayes per pixel per class).
Why: Optdigits pixels are ordinal counts 0..16 with clear per-class distributions; NB models this naturally
and needs only 10×64 means/variances. Inference is ~640 FLOPs/item (vs ~3200 for linear), so it should
match or beat linear accuracy at lower CPU cost.
Failure: If dev accuracy ≤ 0.960 or Ir/item ≥ 5000, the model does not improve on label-linear.
```
```python
"""Gaussian Naive Bayes for label tasks with ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Weights: magic 0x42415953, d, C; float32 prior[C], mean[C][d], var[C][d] (variance floored at 1e-6)."""
import json, struct, sys
import numpy as np

MAGIC = 0x42415953  # 'SBAY' little-endian
VAR_FLOOR = 1e-6

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

    # Verify all features are ordinal (as per card)
    for t in card["input"]["types"]:
        if t != "ordinal":
            raise ValueError("label-bayes requires all ordinal features")

    Xf = X.astype(np.float32)
    prior = np.zeros(C, np.float32)
    mean = np.zeros((C, d), np.float32)
    var = np.zeros((C, d), np.float32)

    for c in range(C):
        mask = (y == c)
        n_c = mask.sum()
        if n_c == 0:
            raise ValueError(f"Class {c} has no training samples")
        prior[c] = np.log(n_c / len(y))
        Xc = Xf[mask]
        mean[c] = Xc.mean(axis=0)
        v = Xc.var(axis=0)
        var[c] = np.maximum(v, VAR_FLOOR)

    # Pack weights: magic, d, C, prior[C], mean[C*d], var[C*d]
    out = struct.pack("<3i", MAGIC, d, C)
    out += prior.astype("<f4").tobytes()
    out += mean.astype("<f4").tobytes()
    out += var.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Gaussian Naive Bayes inference: ./infer weights.bin in.rec N
 * Input record: d bytes (ordinal 0..255). Output: 1 byte class index.
 * Weights: uint32 magic(0x42415953), d, C; float32 prior[C], mean[C][d], var[C][d]. */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x42415953) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = (int)hd[1], C = (int)hd[2];
    if (C < 1 || C > 256 || d < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 12 + (long)C * (1 + 2 * d) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *prior = (const float *)(hd + 3);
    const float *mean = prior + C;
    const float *var = mean + (long)C * d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logprob = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            double lp = prior[c];
            const float *mc = mean + (long)c * d;
            const float *vc = var + (long)c * d;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - mc[j];
                lp += -0.5 * logf(2.0f * 3.141592653589793f * vc[j]) - 0.5 * diff * diff / vc[j];
            }
            logprob[c] = (float)lp;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logprob[c] > logprob[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```