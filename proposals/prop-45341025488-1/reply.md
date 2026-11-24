```hypothesis
Family: label-rbf (Radial Basis Function network with k-means centers, ridge regression output).
Why: 8×8 optdigits pixels are ordinal (0–16); a small set of RBF centers captures non-linear class boundaries better than a linear model, while inference (∼16 centers × 64 dims) stays far cheaper than label-mlp16.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or CPU cost ≥ label-mlp16.
```
```python
"""label-rbf: RBF network with k-means centers + ridge regression.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic(0x52424631), K, d, C, gamma, centers[K][d], W[K][C], b[C], mean[d], std[d] (all float32 LE)."""
import json, math, struct, sys
import numpy as np

SEED = 1
K = 16                  # number of RBF centers
RIDGE = 1e-3            # ridge regularization
KMEANS_ITERS = 50       # k-means max iterations

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def kmeans_pp(X, K, rng):
    n, d = X.shape
    cents = np.empty((K, d), np.float32)
    cents[0] = X[rng.integers(n)]
    for k in range(1, K):
        dist2 = np.min(np.sum((X[:, None, :] - cents[:k])**2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        cents[k] = X[rng.choice(n, p=probs)]
    return cents

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Standardize features (z-score per pixel)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xz = ((X - mu) / sd).astype(np.float32)

    # K-means++ initialization then Lloyd
    cents = kmeans_pp(Xz, K, rng)
    for _ in range(KMEANS_ITERS):
        dist2 = np.sum((Xz[:, None, :] - cents)**2, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(K, np.int32)
        np.add.at(new_cents, assign, Xz)
        np.add.at(counts, assign, 1)
        mask = counts > 0
        new_cents[mask] /= counts[mask, None]
        # keep empty clusters where they are
        new_cents[~mask] = cents[~mask]
        if np.allclose(cents, new_cents):
            cents = new_cents
            break
        cents = new_cents

    # Gamma: inverse median distance between centers
    pdist2 = np.sum((cents[:, None, :] - cents[None, :, :])**2, axis=2)
    med = np.median(pdist2[pdist2 > 0])
    gamma = 0.5 / med if med > 0 else 1.0
    gamma = np.float32(gamma)

    # Design matrix Phi: exp(-gamma * ||x - c||^2)
    Phi = np.exp(-gamma * np.sum((Xz[:, None, :] - cents)**2, axis=2))  # (n, K)

    # Ridge regression: (Phi^T Phi + lambda I) W = Phi^T Y_onehot
    Yoh = np.eye(C, dtype=np.float32)[y]  # (n, C)
    A = Phi.T @ Phi + RIDGE * np.eye(K, dtype=np.float32)
    B = Phi.T @ Yoh
    W = np.linalg.solve(A, B)  # (K, C)
    b = Yoh.mean(axis=0).astype(np.float32) - (Phi.mean(axis=0) @ W)  # bias to center

    # Pack weights
    magic = 0x52424631  # 'RBF1'
    out = struct.pack("<5i f", magic, K, d, C, 0, gamma)  # 5 ints + float (placeholder for alignment)
    # Fix: pack as 4 ints + float
    out = struct.pack("<4i f", magic, K, d, C, gamma)
    out += cents.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-rbf inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output record: 1 byte class (0..9). */
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
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x52424631) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    int32_t K = *(int32_t*)(wb + 4);
    int32_t d = *(int32_t*)(wb + 8);
    int32_t C = *(int32_t*)(wb + 12);
    float gamma = *(float*)(wb + 16);
    if (K <= 0 || d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights: dims\n"); return 3; }
    size_t expected = 20 + (size_t)K * d * 4 + (size_t)K * C * 4 + (size_t)C * 4 + (size_t)d * 4 + (size_t)d * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights: size %ld != %zu\n", nw, expected); return 3; }

    const float *centers = (const float*)(wb + 20);
    const float *W = centers + (size_t)K * d;
    const float *b = W + (size_t)K * C;
    const float *mu = b + C;
    const float *sd = mu + d;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *phi = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Standardize and compute distances to centers
        for (int j = 0; j < K; j++) phi[j] = 0.0f;
        for (int j = 0; j < d; j++) {
            float z = (xi[j] - mu[j]) / sd[j];
            for (int c = 0; c < K; c++) {
                float diff = z - centers[(size_t)c * d + j];
                phi[c] += diff * diff;
            }
        }
        for (int c = 0; c < K; c++) phi[c] = expf(-gamma * phi[c]);

        // Linear combination
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int c = 0; c < K; c++) {
            float pc = phi[c];
            if (pc == 0.0f) continue;
            for (int j = 0; j < C; j++) scores[j] += pc * W[(size_t)c * C + j];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```