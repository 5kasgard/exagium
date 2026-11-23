```hypothesis
Family: label-lda (Linear Discriminant Analysis).  
Why: LDA finds the most discriminative linear projections (up to 9 for 10 classes) and classifies by nearest centroid in that subspace.  
It often outperforms logistic regression on small, Gaussian-like datasets like optdigits, while inference is a single matrix-vector multiply (64×9) plus 10 distance calculations—far cheaper than an MLP.  
If dev accuracy does not exceed label-linear's 0.9613, the family does not justify its addition.
```

```python
import json, struct, sys
import numpy as np

MAGIC = 0x4C444101  # "LDA\1"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
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
    # Standardize
    Xf = X.astype(np.float64)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    X_std = (Xf - mean) / std
    # Class means
    mu_c = np.zeros((C, d), dtype=np.float64)
    for c in range(C):
        mu_c[c] = X_std[y == c].mean(axis=0)
    mu = X_std.mean(axis=0)
    # Within-class scatter
    Sw = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        Xc = X_std[y == c]
        diff = Xc - mu_c[c]
        Sw += diff.T @ diff
    # Between-class scatter
    Sb = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        n_c = (y == c).sum()
        diff = mu_c[c] - mu
        Sb += n_c * np.outer(diff, diff)
    # Regularize Sw
    reg = 1e-4 * np.trace(Sw) / d
    Sw += reg * np.eye(d)
    # Generalized eigenvalue problem Sb v = lambda Sw v
    eigvals, eigvecs = np.linalg.eigh(Sb, Sw)
    # Eigenvalues are in ascending order. Take the last k = C-1 eigenvectors.
    k = C - 1
    idx = np.argsort(eigvals)[::-1][:k]
    proj = eigvecs[:, idx]  # shape (d, k)
    # Make sign deterministic: flip so that the element with largest absolute value in each column is positive
    for j in range(k):
        col = proj[:, j]
        max_idx = np.argmax(np.abs(col))
        if col[max_idx] < 0:
            proj[:, j] = -col
    # Project class means
    class_means_proj = proj.T @ mu_c.T  # shape (k, C) -> transpose to (C, k)
    class_means_proj = class_means_proj.T  # (C, k)
    # Convert to float32 little-endian
    mean32 = mean.astype('<f4')
    std32 = std.astype('<f4')
    proj32 = proj.astype('<f4')
    class_means32 = class_means_proj.astype('<f4')
    # Write weights
    out = struct.pack("<4i", MAGIC, d, C, k)
    out += mean32.tobytes()
    out += std32.tobytes()
    out += proj32.tobytes()
    out += class_means32.tobytes()
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
    if (nw < 16 || hd[0] != 0x4C444101) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], k = hd[3];
    if (d <= 0 || C <= 0 || C > 256 || k <= 0 || k > C-1) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16 + 4L * (2L * d + (long)d * k + (long)C * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *proj = std + d;
    const float *class_means = proj + (long)d * k;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_std = malloc(sizeof(float) * d);
    float *proj_val = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        // Standardize
        for (int j = 0; j < d; j++) {
            x_std[j] = (xi[j] - mean[j]) / std[j];
        }
        // Project
        for (int c = 0; c < k; c++) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += x_std[j] * proj[(long)j * k + c];
            }
            proj_val[c] = sum;
        }
        // Find nearest class mean (Euclidean distance squared)
        unsigned char best = 0;
        float best_dist = 1e30f;
        for (int c = 0; c < C; c++) {
            float dist = 0.0f;
            for (int m = 0; m < k; m++) {
                float diff = proj_val[m] - class_means[(long)c * k + m];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best = (unsigned char)c;
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```