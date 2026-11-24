```hypothesis
Family: label-rbfnet (RBF network with k-means centers and linear readout).
Why: Optdigits pixels are ordinal and spatially structured; RBF networks can capture non-linear boundaries with few centers, and the Gaussian kernel suits the smooth pixel variations. With K=30 centers, it should exceed linear accuracy (0.9613) while keeping inference cost low (O(K*d) per item).
Failure: If dev accuracy ≤ 0.9613 or instructions per item exceed label-mlp16 (8685), the model does not justify its complexity.
```

```python
import json, math, struct, sys

import numpy as np

K = 30
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
KMEANS_ITERS = 10


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, rng, iters=10):
    n = X.shape[0]
    idx = rng.permutation(n)[:K]
    centers = X[idx].copy()
    for _ in range(iters):
        diff = X[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_centers = np.zeros_like(centers)
        counts = np.zeros(K, dtype=np.int64)
        for i in range(n):
            c = assign[i]
            new_centers[c] += X[i]
            counts[c] += 1
        for c in range(K):
            if counts[c] > 0:
                new_centers[c] /= counts[c]
            else:
                new_centers[c] = centers[c]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Standardize per feature
    Xf = X.astype(np.float64)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    Xz = (Xf - mean) / std

    # K-means centers
    centers = kmeans(Xz, K, rng, KMEANS_ITERS).astype(np.float32)

    # Gamma from median pairwise distance between centers
    diff = centers[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    median_dist = np.median(np.sqrt(dist2))
    gamma = 1.0 / (2.0 * median_dist * median_dist) if median_dist > 0 else 1.0
    gamma = np.float32(gamma)

    # RBF features
    diff = Xz[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=2).astype(np.float32)
    Phi = np.exp(-gamma * dist2).astype(np.float32)  # (n, K)

    # Train linear softmax on Phi (same as label-linear)
    n, k = Phi.shape
    W1 = np.zeros((k, C), np.float32)
    b1 = np.zeros(C, np.float32)
    params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Phi[idx], y[idx]
            S = z @ W1 + b1
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Export weights
    # Magic: 0x52424631 ("RBF1")
    # Header: d, K, C
    # mean[d], std[d], centers[K*d], gamma, W1[K*C], b1[C]
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    W1 = W1.astype(np.float32)
    b1 = b1.astype(np.float32)
    expected_floats = 2 * d + K * d + 1 + K * C + C
    expected_bytes = 16 + 4 * expected_floats
    out = struct.pack("<4i", 0x52424631, d, K, C)
    out += mean.tobytes() + std.tobytes() + centers.tobytes() + struct.pack("<f", gamma)
    out += W1.tobytes() + b1.tobytes()
    if len(out) != expected_bytes:
        print(f"Size mismatch: {len(out)} vs {expected_bytes}", file=sys.stderr)
        sys.exit(1)
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x52424631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = (int)hd[1], K = (int)hd[2], C = (int)hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected_floats = 2L * d + (long)K * d + 1 + (long)K * C + C;
    if (nw != 16 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *centers = std + d;
    const float *gamma = centers + (long)K * d;
    const float *W1 = gamma + 1;
    const float *b1 = W1 + (long)K * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *rbf = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Standardize and compute RBF
        for (int c = 0; c < K; c++) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = (xi[j] - mean[j]) / std[j];
                float diff = v - centers[(long)c * d + j];
                sum += diff * diff;
            }
            rbf[c] = expf(-gamma[0] * sum);
        }
        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b1[c];
        for (int u = 0; u < K; u++) {
            float h = rbf[u];
            for (int c = 0; c < C; c++) scores[c] += h * W1[(long)u * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```