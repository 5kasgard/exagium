```hypothesis
Family: label-rbfnet (RBF network with k-means centers). The 8x8 digit pixels have local spatial correlations; an RBF layer with ~100 centers captures non-linear manifolds better than a linear model while staying tiny and fast. Centers are chosen by deterministic k-means on z-scored training data, widths set to median inter-center distance, output weights solved by ridge regression. If dev accuracy ≤ label-linear (0.961) or latency ≥ 2× label-linear, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

M = 100          # number of RBF centers
LAMBDA = 1e-3    # ridge penalty
KM_ITERS = 20    # k-means iterations
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans_pp(X, k, rng):
    n, d = X.shape
    centers = np.empty((k, d), dtype=np.float32)
    idx = rng.integers(n)
    centers[0] = X[idx]
    dist2 = np.full(n, np.inf, dtype=np.float32)
    for i in range(1, k):
        d2 = np.sum((X - centers[i-1])**2, axis=1)
        np.minimum(dist2, d2, out=dist2)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
    return centers

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X_bytes.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # z-score per pixel (ordinal)
    mu = X_bytes.mean(axis=0).astype(np.float32)
    sd = X_bytes.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X = ((X_bytes - mu) / sd).astype(np.float32)

    # deterministic k-means for centers
    centers = kmeans_pp(X, M, rng)
    for _ in range(KM_ITERS):
        dist2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        assign = dist2.argmin(axis=1)
        new_centers = np.zeros_like(centers)
        counts = np.zeros(M, dtype=np.int32)
        np.add.at(new_centers, assign, X)
        np.add.at(counts, assign, 1)
        mask = counts > 0
        centers[mask] = new_centers[mask] / counts[mask, None]
        if not mask.all():
            empty = np.where(~mask)[0]
            centers[empty] = X[rng.choice(len(X), size=len(empty), replace=False)]

    # sigma = median distance between centers
    pdist = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2))
    sigma = float(np.median(pdist[np.triu_indices(M, 1)]))
    if sigma == 0:
        sigma = 1.0

    # design matrix Phi
    dist2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
    Phi = np.exp(-dist2 / (2 * sigma * sigma)).astype(np.float32)

    # one-hot labels
    Y = np.zeros((len(y), C), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0

    # ridge regression: (Phi^T Phi + lambda I) W = Phi^T Y
    A = Phi.T @ Phi + LAMBDA * np.eye(M, dtype=np.float32)
    B = Phi.T @ Y
    W = np.linalg.solve(A, B).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)  # bias (Phi already has effective bias via centers)

    # pack weights: magic, d, M, C, mean[d], std[d], centers[M*d], sigma, W[M*C], b[C]
    magic = 0x52424631  # 'RBF1'
    out = struct.pack("<4i", magic, d, M, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += centers.astype("<f4").tobytes()
    out += struct.pack("<f", sigma)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x52424631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], M = hd[2], C = hd[3];
    if (d != 64 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 16
        + 4L * d * 2            // mean, std
        + 4L * M * d            // centers
        + 4                     // sigma
        + 4L * M * C            // W
        + 4L * C;               // b
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *mean = p; p += d;
    const float *std  = p; p += d;
    const float *centers = p; p += (long)M * d;
    float sigma = *p++; 
    const float *W = p; p += (long)M * C;
    const float *b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * M);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xnorm[j] = (xi[j] - mean[j]) / std[j];
        for (int m = 0; m < M; m++) {
            float sum = 0.0f;
            const float *c = centers + (long)m * d;
            for (int j = 0; j < d; j++) {
                float diff = xnorm[j] - c[j];
                sum += diff * diff;
            }
            phi[m] = expf(-sum / (2.0f * sigma * sigma));
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int m = 0; m < M; m++) {
            float pm = phi[m];
            const float *wm = W + (long)m * C;
            for (int c = 0; c < C; c++) scores[c] += pm * wm[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```