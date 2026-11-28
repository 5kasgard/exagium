```hypothesis
Family: label-rbf (Radial Basis Function network with k-means centers and ridge output).
Why: Optdigits has non-linear class boundaries; 30 RBF centers from k-means capture local structure better than a global linear model. Ridge regression on RBF features is fast, deterministic, and often matches MLP accuracy with fewer parameters.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or inference cost > 2× label-linear (~10k Ir/item), the family doesn't justify its complexity.
```

```python
"""label-rbf: RBF network for fixed-size byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Preprocessing: per-feature z-score (ordinal). 30 centers via k-means++, RBF gamma=0.1, ridge lambda=1e-4.
Weights format: magic 0x52424631, d, k, C; float32 mean[d], std[d], centers[k][d], gamma, W[k][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
N_CENTERS = 30
GAMMA = 0.1
L2 = 1e-4
KMEANS_ITERS = 50


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, k, rng, max_iter=50):
    n, d = X.shape
    centers = np.empty((k, d), dtype=np.float32)
    idx = rng.integers(n)
    centers[0] = X[idx]
    for i in range(1, k):
        dist2 = np.min(np.sum((X[:, None] - centers[:i])**2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
    for _ in range(max_iter):
        dist2 = np.sum((X[:, None] - centers)**2, axis=2)
        labels = np.argmin(dist2, axis=1)
        new_centers = np.zeros_like(centers)
        counts = np.zeros(k, dtype=np.int32)
        np.add.at(new_centers, labels, X)
        np.add.at(counts, labels, 1)
        for j in range(k):
            if counts[j] > 0:
                new_centers[j] /= counts[j]
            else:
                new_centers[j] = X[rng.integers(n)]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers


def rbf_features(X, centers, gamma):
    dist2 = np.sum((X[:, None] - centers)**2, axis=2)
    return np.exp(-gamma * dist2).astype(np.float32)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    rng = np.random.default_rng(SEED)

    # z-score per feature (all ordinal)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xz = ((X - mu) / sd).astype(np.float32)

    # k-means centers
    centers = kmeans(Xz, N_CENTERS, rng, KMEANS_ITERS)

    # RBF features
    Phi = rbf_features(Xz, centers, GAMMA)  # n x k

    # Ridge regression with bias (augment Phi with ones)
    n, k = Phi.shape
    Phi_aug = np.hstack([Phi, np.ones((n, 1), dtype=np.float32)])  # n x (k+1)
    Y = np.eye(C, dtype=np.float32)[y]  # n x C

    A = Phi_aug.T @ Phi_aug
    A[:k, :k] += L2 * np.eye(k)
    B = Phi_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (k+1) x C
    W = W_aug[:k].astype(np.float32)
    b = W_aug[k].astype(np.float32)

    # Pack weights
    magic = 0x52424631  # 'RBF1'
    out = struct.pack("<4i", magic, d, N_CENTERS, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += centers.astype("<f4").tobytes()
    out += np.array([GAMMA], dtype="<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rbf inference: ./infer weights.bin in.rec N
 * Input record: d bytes (z-scored per feature). Output: one byte class index. */
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
    int d = hd[1], k = hd[2], C = hd[3];
    if (d <= 0 || k <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }

    long expected = 16L + 4L * (2L * d + (long)k * d + 1 + (long)k * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 4);
    const float *mean = ptr; ptr += d;
    const float *std  = ptr; ptr += d;
    const float *centers = ptr; ptr += (long)k * d;
    const float gamma = *ptr++; 
    const float *W = ptr; ptr += (long)k * C;
    const float *b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * k);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        for (int j = 0; j < d; j++) {
            float v = (float)xi[j];
            float s = std[j];
            z[j] = (s != 0.0f) ? (v - mean[j]) / s : 0.0f;
        }

        for (int c = 0; c < k; c++) {
            float dist2 = 0.0f;
            const float *cen = centers + (long)c * d;
            for (int j = 0; j < d; j++) {
                float diff = z[j] - cen[j];
                dist2 += diff * diff;
            }
            phi[c] = expf(-gamma * dist2);
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < k; u++) {
            float ph = phi[u];
            const float *wu = W + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += ph * wu[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```