```hypothesis
Family: label-gam (Generalized Additive Model with per-feature lookup tables).
Why: The Adult dataset has 6 ordinal features (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) that label-linear forces into a linear z-score effect. A saturated lookup table (one weight per feature-value) captures arbitrary univariate shapes (e.g., diminishing returns for hours, non-monotonic capital-gain) with only 14×256 ≈ 3.5K parameters. L2-regularized Adam on logistic loss keeps it convex and deterministic. Inference needs just 14 table lookups + adds per item (half the work of label-linear's 28), so it should be faster and more accurate. Failure mode: if ordinal features are truly linear or L2 is insufficient to prevent overfitting on sparse categorical values, accuracy will not exceed label-linear's 0.8524.
```

```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    assert C == 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Model: W[d, 256] (per-feature per-value logit contribution), bias
    W = np.zeros((d, 256), dtype=np.float32)
    bias = np.float32(0.0)

    # Adam state
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    mb = np.float32(0.0)
    vb = np.float32(0.0)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bsz = len(yb)

            # Forward: logit = bias + sum_j W[j, xb[:,j]]
            logit = np.full(bsz, bias, dtype=np.float32)
            for j in range(d):
                logit += W[j, xb[:, j]]

            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yb).astype(np.float32)  # (bsz,)

            # Gradients
            g_bias = grad.mean()
            gW = np.zeros_like(W)
            for j in range(d):
                gW[j] = np.bincount(xb[:, j], weights=grad, minlength=256).astype(np.float32) / bsz
                gW[j] += L2 * W[j]

            t += 1
            # Adam update for W
            mW *= 0.9
            mW += 0.1 * gW
            vW *= 0.999
            vW += 0.001 * gW * gW
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)

            # Adam update for bias
            mb = 0.9 * mb + 0.1 * g_bias
            vb = 0.999 * vb + 0.001 * g_bias * g_bias
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            bias -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # Write weights: magic 'GAM1' (0x47414D31), d, C, W[d][256] float32 LE, bias float32 LE
    out = struct.pack("<3i", 0x47414D31, d, C)
    out += W.astype("<f4").tobytes()
    out += struct.pack("<f", bias)
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
    if (nw < 12 || hd[0] != 0x47414D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C != 2 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 12 + (long)d * 256 * 4 + 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 3);
    float bias = *(const float *)(W + (long)d * 256);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = bias;
        for (int j = 0; j < d; j++) {
            logit += W[(long)j * 256 + xi[j]];
        }
        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```