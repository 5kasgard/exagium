```hypothesis
Family: label-rbf (deterministic RBF network with k-means centers and ridge regression output).
Why: Optdigits 8x8 pixels have local structure; RBF networks capture non-linear boundaries with few centers,
are fast at inference (O(centers*dim)), and closed-form output weights ensure determinism and reproducibility.
Failure signal: Dev accuracy ≤ 0.9613 (label-linear) or inference cost ≥ 2× label-linear.
```

```python
"""label-rbf: RBF network for fixed-size byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Centers from k-means++ on z-scored training data (n_centers=64).
Output weights via ridge regression (lambda=1e-3) on RBF activations. Sigma tuned on last 20% of train.
Weights format: magic 0x52424631, d, K, C; float32 mean[d], std[d], centers[K][d], sigma, W[K][C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
N_CENTERS = 64
LAMBDA = 1e-3
SIGMA_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans_plusplus(X, k, rng):
    n, d = X.shape
    centers = np.empty((k, d), dtype=np.float32)
    idx = rng.integers(n)
    centers[0] = X[idx]
    dist2 = np.full(n, np.inf, dtype=np.float32)
    for i in range(1, k):
        d2 = np.sum((X - centers[i - 1]) ** 2, axis=1)
        np.minimum(dist2, d2, out=dist2)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
    # Lloyd's algorithm (10 iters, deterministic)
    for _ in range(10):
        diff = X[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_centers = np.zeros_like(centers)
        counts = np.zeros(k, dtype=np.int32)
        np.add.at(new_centers, assign, X)
        np.add.at(counts, assign, 1)
        mask = counts > 0
        new_centers[mask] /= counts[mask, None]
        if not mask.all():
            # reassign empty clusters to farthest points
            empty = np.where(~mask)[0]
            for e in empty:
                far = np.argmax(dist2[:, e])
                new_centers[e] = X[far]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    # Split for sigma tuning (last 20%)
    split = int(n * 0.8)
    X_tr, X_va = X_bytes[:split], X_bytes[split:]
    y_tr, y_va = y[:split], y[split:]

    # Z-score per feature on training split
    mu = X_tr.mean(axis=0).astype(np.float32)
    sd = X_tr.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X_tr_z = ((X_tr - mu) / sd).astype(np.float32)
    X_va_z = ((X_va - mu) / sd).astype(np.float32)

    rng = np.random.default_rng(SEED)
    centers = kmeans_plusplus(X_tr_z, N_CENTERS, rng).astype(np.float32)

    # Tune sigma on validation split
    def rbf_activations(Z, centers, sigma):
        diff = Z[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        return np.exp(-dist2 / (2.0 * sigma * sigma))

    best_sigma, best_acc = 1.0, -1.0
    Phi_tr = None
    for sigma in SIGMA_CANDIDATES:
        Phi = rbf_activations(X_tr_z, centers, sigma)
        # Ridge regression: W = (Phi^T Phi + lambda I)^-1 Phi^T Y
        Y_onehot = np.eye(C, dtype=np.float32)[y_tr]
        A = Phi.T @ Phi + LAMBDA * np.eye(N_CENTERS, dtype=np.float32)
        B = Phi.T @ Y_onehot
        W = np.linalg.solve(A, B).astype(np.float32)
        bias = np.zeros(C, dtype=np.float32)
        # Validate
        Phi_va = rbf_activations(X_va_z, centers, sigma)
        scores = Phi_va @ W + bias
        pred = np.argmax(scores, axis=1)
        acc = (pred == y_va).mean()
        if acc > best_acc:
            best_acc, best_sigma = acc, sigma
            Phi_tr = Phi

    # Retrain on full training set with best sigma
    X_all_z = ((X_bytes - mu) / sd).astype(np.float32)
    Phi_all = rbf_activations(X_all_z, centers, best_sigma)
    Y_onehot_all = np.eye(C, dtype=np.float32)[y]
    A = Phi_all.T @ Phi_all + LAMBDA * np.eye(N_CENTERS, dtype=np.float32)
    B = Phi_all.T @ Y_onehot_all
    W = np.linalg.solve(A, B).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Pack weights
    magic = 0x52424631
    out = struct.pack("<4i", magic, d, N_CENTERS, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += centers.astype("<f4").tobytes()
    out += struct.pack("<f", best_sigma)
    out += W.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rbf inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index (argmax, first max wins).
 * Weights: magic 0x52424631, d, K, C; float32 mean[d], std[d], centers[K][d], sigma, W[K][C], bias[C]. */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x52424631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d < 1 || K < 1 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }

    size_t expected = 16
        + (size_t)d * 4       // mean
        + (size_t)d * 4       // std
        + (size_t)K * d * 4   // centers
        + 4                   // sigma
        + (size_t)K * C * 4   // W
        + (size_t)C * 4;      // bias
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *centers = std + d;
    const float *sigma_ptr = centers + (size_t)K * d;
    float sigma = *sigma_ptr;
    const float *W = (const float *)(sigma_ptr + 1);
    const float *bias = W + (size_t)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // z-score
        for (int j = 0; j < d; j++) {
            z[j] = ((float)xi[j] - mean[j]) / std[j];
        }

        // RBF activations
        for (int u = 0; u < K; u++) {
            float dist2 = 0.0f;
            const float *cu = centers + (size_t)u * d;
            for (int j = 0; j < d; j++) {
                float diff = z[j] - cu[j];
                dist2 += diff * diff;
            }
            phi[u] = expf(-dist2 / (2.0f * sigma * sigma));
        }

        // Linear output
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int u = 0; u < K; u++) {
            float phiu = phi[u];
            const float *wu = W + (size_t)u * C;
            for (int c = 0; c < C; c++) scores[c] += phiu * wu[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```