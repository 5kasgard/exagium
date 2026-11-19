```hypothesis
Family: label-rbfnet. An RBF network with k-means centers and a linear output layer. The 8x8 optdigits pixels have local correlations; RBF features can capture non-linear boundaries with few centers. With K=20 centers the inference cost stays low (≈1500 FLOPs/item) while adding expressive power over a linear model. If dev accuracy does not exceed label-linear (0.9613), the non-linearity is not helping or the hyperparameter search is insufficient.
```
```python
"""label-rbfnet: RBF network with k-means centers + linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  magic(0x52424631), d, K, C,
  mean[d], std[d], centers[K*d], gamma, W[K*C], b[C]  (all float32)
Hyperparameters (K, gamma) chosen on last 20% of train.in (tuning split)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_GRID = [10, 20, 30]
GAMMA_GRID = [0.01, 0.1, 1.0]
KMEANS_ITERS = 10


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, rng):
    n, d = X.shape
    idx = rng.choice(n, K, replace=False)
    centers = X[idx].copy()
    for _ in range(KMEANS_ITERS):
        # assign
        diff = X[:, None, :] - centers[None, :, :]  # n x K x d
        dist2 = (diff * diff).sum(axis=2)
        assign = dist2.argmin(axis=1)
        # update
        new_centers = np.zeros_like(centers)
        counts = np.zeros(K, dtype=np.int32)
        for i in range(n):
            c = assign[i]
            new_centers[c] += X[i]
            counts[c] += 1
        for k in range(K):
            if counts[k]:
                centers[k] = new_centers[k] / counts[k]
            else:
                # reinitialize empty cluster randomly
                centers[k] = X[rng.integers(n)]
    return centers


def rbf_features(X, centers, gamma):
    # X: n x d, centers: K x d -> n x K
    diff = X[:, None, :] - centers[None, :, :]  # n x K x d
    dist2 = (diff * diff).sum(axis=2)
    return np.exp(-gamma * dist2).astype(np.float32)


def train_linear(Z, y, C, rng):
    n, K = Z.shape
    W = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
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
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            G = (P / len(yy)).astype(np.float32)  # n x C
            grad_W = z.T @ G + L2 * W
            grad_b = G.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    n = X.shape[0]
    tune_size = n // 5
    train_idx = slice(0, n - tune_size)
    tune_idx = slice(n - tune_size, n)

    # standardize on train split
    X_train = X[train_idx].astype(np.float64)
    X_tune = X[tune_idx].astype(np.float64)
    y_train = y[train_idx]
    y_tune = y[tune_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_std = (X_train - mean) / std
    X_tune_std = (X_tune - mean) / std

    best_acc = -1.0
    best_K = K_GRID[0]
    best_gamma = GAMMA_GRID[0]
    best_centers = None
    best_W = None
    best_b = None

    for K in K_GRID:
        centers = kmeans(X_train_std, K, rng)
        for gamma in GAMMA_GRID:
            Z_train = rbf_features(X_train_std, centers, gamma)
            Z_tune = rbf_features(X_tune_std, centers, gamma)
            W, b = train_linear(Z_train, y_train, C, rng)
            # evaluate on tune
            S = Z_tune @ W + b
            pred = S.argmax(axis=1)
            acc = (pred == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_K = K
                best_gamma = gamma
                best_centers = centers.copy()
                best_W = W.copy()
                best_b = b.copy()

    # retrain on full data with best hyperparameters
    X_full = X.astype(np.float64)
    mean_full = X_full.mean(axis=0)
    std_full = X_full.std(axis=0)
    std_full[std_full == 0] = 1.0
    X_full_std = (X_full - mean_full) / std_full
    centers_full = kmeans(X_full_std, best_K, rng)
    Z_full = rbf_features(X_full_std, centers_full, best_gamma)
    W_full, b_full = train_linear(Z_full, y, C, rng)

    # pack weights
    # magic, d, K, C
    hdr = struct.pack("<4i", 0x52424631, d, best_K, C)
    # mean[d], std[d] as float32
    mean_b = mean_full.astype(np.float32).tobytes()
    std_b = std_full.astype(np.float32).tobytes()
    # centers[K*d]
    centers_b = centers_full.astype(np.float32).tobytes()
    # gamma
    gamma_b = struct.pack("<f", best_gamma)
    # W[K*C], b[C]
    W_b = W_full.astype(np.float32).tobytes()
    b_b = b_full.astype(np.float32).tobytes()

    open(sys.argv[4], "wb").write(hdr + mean_b + std_b + centers_b + gamma_b + W_b + b_b)


if __name__ == "__main__":
    main()
```
```c
/* label-rbfnet inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output: one byte class index.
 * Weights layout (little-endian):
 *   int32 magic(0x52424631), d, K, C
 *   float32 mean[d], std[d], centers[K*d], gamma, W[K*C], b[C] */
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
    int32_t *hdr = (int32_t *)wb;
    if (nw < 16 || hdr[0] != 0x52424631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hdr[1], K = hdr[2], C = hdr[3];
    if (d <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 16
        + (size_t)d * 4 * 2          // mean, std
        + (size_t)K * d * 4          // centers
        + 4                          // gamma
        + (size_t)K * C * 4          // W
        + (size_t)C * 4;             // b
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    float *mean = (float *)(hdr + 4);
    float *std = mean + d;
    float *centers = std + d;
    float gamma = *(float *)(centers + (size_t)K * d);
    float *W = (float *)((float *)(centers + (size_t)K * d) + 1);
    float *b = W + (size_t)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = (float *)malloc(sizeof(float) * d);
    float *rbf = (float *)malloc(sizeof(float) * K);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // standardize
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mean[j]) / std[j];
        }
        // RBF features
        for (int u = 0; u < K; u++) {
            float dist2 = 0.0f;
            const float *c = centers + (size_t)u * d;
            for (int j = 0; j < d; j++) {
                float diff = z[j] - c[j];
                dist2 += diff * diff;
            }
            rbf[u] = expf(-gamma * dist2);
        }
        // linear scores
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < K; u++) {
            float fu = rbf[u];
            const float *wu = W + (size_t)u * C;
            for (int c = 0; c < C; c++) scores[c] += fu * wu[c];
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```