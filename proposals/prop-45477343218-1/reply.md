```hypothesis
We propose "label-rbf", a radial basis function network with k-means centers and a linear output layer. The RBF layer provides nonlinearity to capture local pixel correlations in optdigits, while the small number of centers (tuned on the last 20% of train) keeps inference cost low (O(K*d) distances + K*C linear). If dev accuracy does not exceed label-linear's 0.9613, the family fails.
```
```python
"""label-rbf: RBF network with k-means centers and linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_CHOICES = [10, 20, 30]
GAMMA_CHOICES = [0.1, 0.5, 1.0, 2.0]
KMEANS_ITER = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, rng, max_iter=KMEANS_ITER):
    """X: (n, d) float32, returns centroids (K, d) float32."""
    n, d = X.shape
    centroids = np.zeros((K, d), dtype=np.float32)
    idx = rng.integers(n)
    centroids[0] = X[idx]
    for i in range(1, K):
        diff = X[:, None] - centroids[:i][None]
        dists = np.sum(diff ** 2, axis=2)
        min_dists = np.min(dists, axis=1)
        probs = min_dists / (min_dists.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
    for _ in range(max_iter):
        diff = X[:, None] - centroids[None]
        dists = np.sum(diff ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K, dtype=np.int32)
        for j in range(K):
            mask = labels == j
            if np.any(mask):
                new_centroids[j] = X[mask].mean(axis=0)
                counts[j] = mask.sum()
            else:
                idx = rng.integers(n)
                new_centroids[j] = X[idx]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids


def rbf_features(X_std, centers, gamma):
    """X_std: (n, d), centers: (K, d) -> (n, K) float32."""
    diff = X_std[:, None] - centers[None]  # (n, K, d)
    dists = np.sum(diff ** 2, axis=2)
    return np.exp(-gamma * dists).astype(np.float32)


def train_linear(Z, y, C, rng, epochs):
    """Linear softmax with Adam. Z: (n, K), y: (n,). Returns W (K, C), b (C)."""
    n, K = Z.shape
    W = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            scores = z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    # Standardize on training portion for tuning
    X_train = X[train_idx].astype(np.float32)
    X_tune = X[tune_idx].astype(np.float32)
    y_train = y[train_idx]
    y_tune = y[tune_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_std = (X_train - mean) / std
    X_tune_std = (X_tune - mean) / std

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_K = K_CHOICES[0]
    best_gamma = GAMMA_CHOICES[0]
    best_centers = None

    for K in K_CHOICES:
        centers = kmeans(X_train_std, K, rng)
        for gamma in GAMMA_CHOICES:
            Z_train = rbf_features(X_train_std, centers, gamma)
            Z_tune = rbf_features(X_tune_std, centers, gamma)
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
            W, b = train_linear(Z_train, y_train, C, rng, epochs)
            scores = Z_tune @ W + b
            pred = np.argmax(scores, axis=1)
            acc = (pred == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_K = K
                best_gamma = gamma
                best_centers = centers.copy()

    # Final training on all data
    X_all = X.astype(np.float32)
    mean_all = X_all.mean(axis=0)
    std_all = X_all.std(axis=0)
    std_all[std_all == 0] = 1.0
    X_all_std = (X_all - mean_all) / std_all
    centers = kmeans(X_all_std, best_K, rng)
    Z_all = rbf_features(X_all_std, centers, best_gamma)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W, b = train_linear(Z_all, y, C, rng, epochs)

    # Export weights
    # Magic: 0x52424631 ("RBF1")
    magic = 0x52424631
    header = struct.pack("<4if", magic, d, best_K, C, best_gamma)
    out = header
    out += mean_all.astype("<f4").tobytes()
    out += std_all.astype("<f4").tobytes()
    out += centers.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-rbf inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class (argmax). */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }

    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x52424631) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    float gamma = *(float *)(hd + 4);
    if (d != 64 || C != 10 || K <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    const float *mean = (const float *)(hd + 5);                // d
    const float *std = mean + d;                                // d
    const float *centers = std + d;                             // K*d
    const float *W = centers + (long)K * d;                     // K*C
    const float *b = W + (long)K * C;                           // C

    long expected = 20 + 4L * (2L * d + (long)K * d + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Precompute standardized lookup: for each feature j, each byte value v -> (v - mean[j]) / std[j]
    float *std_lut = malloc(sizeof(float) * d * 256);
    for (int j = 0; j < d; j++) {
        float m = mean[j], s = std[j];
        for (int v = 0; v < 256; v++) {
            std_lut[j * 256 + v] = (v - m) / s;
        }
    }

    float *x_std = malloc(sizeof(float) * d);
    float *rbf = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) x_std[j] = std_lut[j * 256 + xi[j]];

        for (int c = 0; c < K; c++) {
            const float *cent = centers + (long)c * d;
            float dist2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = x_std[j] - cent[j];
                dist2 += diff * diff;
            }
            rbf[c] = expf(-gamma * dist2);
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < K; u++) {
            float r = rbf[u];
            const float *wu = W + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += r * wu[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```