```hypothesis
Family: label-lda (Linear Discriminant Analysis with shared covariance).
Why: Optdigits pixels are ordinal 0..16; LDA's Gaussian assumption with pooled covariance often outperforms logistic regression on this classic dataset, while retaining the same O(d·C) inference cost as label-linear.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds label-linear by >10%.
```

```python
"""label-lda: Linear Discriminant Analysis for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-threaded, stdlib+numpy only.
Exports weights in the same lookup-table format as label-linear (H=0)."""
import json, struct, sys
import numpy as np

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    y_bytes = b"".join(r[0:1] for r in records(sys.argv[3]))
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(y_bytes)
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(n, d)
    y = np.frombuffer(y_bytes, dtype=np.uint8).astype(np.int64)

    # Encode: z-score each pixel (all ordinal)
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, dtype=np.float64)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd  # shape (n, d), float64

    # Class priors and means
    priors = np.bincount(y, minlength=C).astype(np.float64) / n
    class_means = np.zeros((C, d), dtype=np.float64)
    for c in range(C):
        class_means[c] = Z[y == c].mean(axis=0) if np.any(y == c) else 0.0

    # Pooled covariance
    cov = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if np.sum(mask) > 1:
            diff = Z[mask] - class_means[c]
            cov += diff.T @ diff
    cov /= (n - C)

    # Regularization
    reg = 1e-4 * np.trace(cov) / d
    cov.flat[::d+1] += reg

    # Invert covariance
    inv_cov = np.linalg.inv(cov)

    # Discriminant weights and biases
    W = inv_cov @ class_means.T          # (d, C)
    b = -0.5 * np.sum(class_means * (inv_cov @ class_means.T).T, axis=1) + np.log(priors)

    # Build lookup tables T[j][v][c] = ((v - mu_j)/sd_j) * W[j, c]
    T = np.zeros((d, 256, C), dtype=np.float32)
    for j in range(d):
        scale = 1.0 / sd[j]
        for v in range(256):
            T[j, v, :] = ((v - mu[j]) * scale) * W[j, :].astype(np.float32)

    b1 = b.astype(np.float32)

    # Write weights.bin (magic 0x54424C31, d, H=0, C)
    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, 0, C)
    out += T.tobytes()
    out += b1.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-lda inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class 0..9).
 * Weights format: magic 0x54424C31, int32 d, H=0, C; float32 T[d][256][C]; float32 b1[C]. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "unsupported weights (H must be 0)\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) acc[c] = b1[c];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```