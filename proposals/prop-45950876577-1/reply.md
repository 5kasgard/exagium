```hypothesis
Family: label-rbf. A Radial Basis Function network with k-means centers (tuned on the 20% tuning split) followed by a linear classifier. The 8x8 pixel space benefits from local non-linearities; RBFs capture this with few centers (K≈30) while inference stays cheap (O(K*d) ≈ 2k FLOPs). If dev accuracy ≤ 0.9613 (label-linear) or CPU cost > 2× label-linear, the family does not justify its complexity.
```
```python
"""label-rbf: RBF network with k-means centers + linear top layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split
K_CHOICES = [10, 20, 30, 40, 50]
GAMMA_CHOICES = [0.1, 0.5, 1.0, 2.0, 5.0]

# Linear training hyperparameters (same as label-linear)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

MAGIC = 0x52424631  # 'RBF1'


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, rng, max_iter=20):
    """Deterministic k-means++ with fixed seed."""
    n, d = X.shape
    # k-means++ initialization
    centroids = np.zeros((K, d), dtype=np.float32)
    idx = rng.integers(n)
    centroids[0] = X[idx]
    for k in range(1, K):
        dist2 = np.min(np.sum((X[:, None] - centroids[:k]) ** 2, axis=2), axis=1)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        centroids[k] = X[idx]
    # Lloyd iterations
    for _ in range(max_iter):
        dist2 = np.sum((X[:, None] - centroids) ** 2, axis=2)
        assign = np.argmin(dist2, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.int32)
        for i in range(n):
            c = assign[i]
            new_centroids[c] += X[i]
            counts[c] += 1
        for k in range(K):
            if counts[k] > 0:
                new_centroids[k] /= counts[k]
            else:
                # reassign empty cluster to a random point
                new_centroids[k] = X[rng.integers(n)]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids


def rbf_features(X, centers, gamma):
    """Compute exp(-gamma * ||x - c||^2) for each center."""
    # X: (n, d), centers: (K, d) -> (n, K)
    diff = X[:, None, :] - centers[None, :, :]  # (n, K, d)
    dist2 = np.sum(diff ** 2, axis=2)           # (n, K)
    return np.exp(-gamma * dist2).astype(np.float32)


def train_linear(Z, y, C, rng):
    """Train linear model with Adam (same as label-linear H=0)."""
    n, k = Z.shape
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            scores = z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def predict_linear(Z, W, b):
    scores = Z @ W + b
    return np.argmax(scores, axis=1)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # Standardize per feature (ordinal)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X_std = ((X - mu) / sd).astype(np.float32)

    # Split: last 20% for tuning (file order)
    n = X_std.shape[0]
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X_std[train_idx], y[train_idx]
    X_tune, y_tune = X_std[tune_idx], y[tune_idx]

    best_acc = -1.0
    best_K = K_CHOICES[0]
    best_gamma = GAMMA_CHOICES[0]
    best_centers = None
    best_W = None
    best_b = None

    for K in K_CHOICES:
        for gamma in GAMMA_CHOICES:
            rng = np.random.default_rng(SEED + K * 1000 + int(gamma * 100))
            centers = kmeans(X_train, K, rng)
            Z_train = rbf_features(X_train, centers, gamma)
            Z_tune = rbf_features(X_tune, centers, gamma)
            rng2 = np.random.default_rng(SEED + 1)
            W, b = train_linear(Z_train, y_train, C, rng2)
            pred = predict_linear(Z_tune, W, b)
            acc = (pred == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_K = K
                best_gamma = gamma
                best_centers = centers
                best_W = W
                best_b = b

    # Retrain on all data with best hyperparameters
    rng = np.random.default_rng(SEED + best_K * 1000 + int(best_gamma * 100))
    centers = kmeans(X_std, best_K, rng)
    Z_all = rbf_features(X_std, centers, best_gamma)
    rng2 = np.random.default_rng(SEED + 1)
    W, b = train_linear(Z_all, y, C, rng2)

    # Save weights
    # Format: magic, d, K, C, gamma; mean[d], inv_std[d]; centers[K][d]; W[K][C]; b[C]
    inv_std = (1.0 / sd).astype(np.float32)
    gamma_f32 = np.float32(best_gamma)
    out = struct.pack("<5i", MAGIC, d, best_K, C, 0)  # placeholder for gamma (will overwrite)
    out = struct.pack("<4if", MAGIC, d, best_K, C, gamma_f32)
    out += mu.astype("<f4").tobytes()
    out += inv_std.astype("<f4").tobytes()
    out += centers.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-rbf inference: ./infer weights.bin in.rec N */
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
    if (nw < 20 || hd[0] != 0x52424631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    float gamma = *(float *)(hd + 4);
    if (C < 1 || C > 256 || K < 1 || d < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    const float *mean = (const float *)(hd + 5);
    const float *inv_std = mean + d;
    const float *centers = inv_std + d;
    const float *W = centers + (long)K * d;
    const float *b = W + (long)K * C;
    long expected = 20 + 4L * (d + d + (long)K * d + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *rbf = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Standardize and compute RBF features
        for (int c = 0; c < K; c++) {
            float dist2 = 0.0f;
            const float *center = centers + (long)c * d;
            for (int j = 0; j < d; j++) {
                float val = (xi[j] - mean[j]) * inv_std[j];
                float diff = val - center[j];
                dist2 += diff * diff;
            }
            rbf[c] = expf(-gamma * dist2);
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int c = 0; c < K; c++) {
            float r = rbf[c];
            const float *w = W + (long)c * C;
            for (int j = 0; j < C; j++) scores[j] += r * w[j];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```