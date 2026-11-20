```hypothesis
Propose a Radial Basis Function network (label-rbf) with k-means centers and a linear output layer.
The 8x8 optdigits pixels are ordinal; z-scoring plus a small set of RBF prototypes (30-70) captures non-linearities
better than pure linear while keeping inference fast (O(K*d) per item). If dev accuracy does not exceed 0.965
or Ir/item exceeds 10k, the model does not justify its complexity over label-linear.
```

```python
"""label-rbf: RBF network with k-means centers + linear softmax output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x52424631, d, K, C; float32 mean[d], std[d], gamma, centers[K*d], W[K*C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
# Grid for tuning split (last 20% of train.in)
K_GRID = [30, 50, 70]
GAMMA_GRID = [0.05, 0.1, 0.2, 0.5]
KMEANS_ITER = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, rng, max_iter=100):
    n, d = X.shape
    idx = rng.choice(n, K, replace=False)
    centers = X[idx].copy()
    for _ in range(max_iter):
        # Assign
        diff = X[:, None, :] - centers[None, :, :]  # (n, K, d)
        dist2 = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist2, axis=1)
        # Update
        new_centers = np.zeros_like(centers)
        counts = np.zeros(K, dtype=np.int64)
        for i in range(n):
            c = assign[i]
            new_centers[c] += X[i]
            counts[c] += 1
        for k in range(K):
            if counts[k] > 0:
                new_centers[k] /= counts[k]
            else:
                # Re-initialize empty cluster
                new_centers[k] = X[rng.integers(n)]
        if np.allclose(centers, new_centers):
            centers = new_centers
            break
        centers = new_centers
    return centers


def rbf_features(X, centers, gamma):
    # X: (n, d), centers: (K, d) -> phi: (n, K)
    n, d = X.shape
    K = centers.shape[0]
    # Compute squared distances efficiently
    X2 = np.sum(X * X, axis=1)[:, None]          # (n, 1)
    C2 = np.sum(centers * centers, axis=1)[None, :]  # (1, K)
    XC = X @ centers.T                            # (n, K)
    dist2 = X2 + C2 - 2 * XC
    np.maximum(dist2, 0, out=dist2)  # numerical safety
    phi = np.exp(-gamma * dist2)
    return phi.astype(np.float32)


def train_linear(phi, y, C, rng):
    n, K = phi.shape
    W = np.zeros((K, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = phi[idx], y[idx]
            # Softmax gradient
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
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
    n = X.shape[0]

    # Z-score encoding (ordinal only per card)
    rng = np.random.default_rng(SEED)
    mean = X.mean(axis=0).astype(np.float64)
    std = X.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0
    Z = ((X - mean) / std).astype(np.float32)

    # Split tuning (last 20%)
    split = int(0.8 * n)
    Z_train, Z_tune = Z[:split], Z[split:]
    y_train, y_tune = y[:split], y[split:]

    best_acc = -1.0
    best_K = K_GRID[0]
    best_gamma = GAMMA_GRID[0]
    best_centers = None

    for K in K_GRID:
        centers = kmeans(Z_train, K, rng, KMEANS_ITER)
        for gamma in GAMMA_GRID:
            phi_train = rbf_features(Z_train, centers, gamma)
            phi_tune = rbf_features(Z_tune, centers, gamma)
            W, b = train_linear(phi_train, y_train, C, rng)
            # Evaluate
            scores = phi_tune @ W + b
            pred = np.argmax(scores, axis=1)
            acc = (pred == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_K = K
                best_gamma = gamma
                best_centers = centers.copy()

    # Retrain on full data with best hyperparameters
    centers = kmeans(Z, best_K, rng, KMEANS_ITER)
    phi = rbf_features(Z, centers, best_gamma)
    W, b = train_linear(phi, y, C, rng)

    # Write weights
    magic = 0x52424631  # 'RBF1'
    out = struct.pack("<4i", magic, d, best_K, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += struct.pack("<f", best_gamma)
    out += centers.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rbf inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class index.
 * Weights: magic, d, K, C; float32 mean[d], std[d], gamma, centers[K*d], W[K*C], b[C]. */
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
    if (d <= 0 || K <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }

    float *ptr = (float *)(hd + 4);
    float *mean = ptr; ptr += d;
    float *std  = ptr; ptr += d;
    float gamma = *ptr; ptr++;
    float *centers = ptr; ptr += (long)K * d;
    float *W = ptr; ptr += (long)K * C;
    float *b = ptr;
    long expected = 16 + 4L * (2L * d + 1 + (long)K * d + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);
    float *c2 = malloc(sizeof(float) * K);
    for (int i = 0; i < K; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            float v = centers[(long)i * d + j];
            sum += v * v;
        }
        c2[i] = sum;
    }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Normalize and compute ||x||^2
        float x2 = 0.0f;
        for (int j = 0; j < d; j++) {
            float v = (xi[j] - mean[j]) / std[j];
            xnorm[j] = v;
            x2 += v * v;
        }
        // Compute RBF activations
        for (int c = 0; c < K; c++) {
            float dot = 0.0f;
            for (int j = 0; j < d; j++) dot += xnorm[j] * centers[(long)c * d + j];
            float dist2 = x2 + c2[c] - 2.0f * dot;
            if (dist2 < 0.0f) dist2 = 0.0f;
            phi[c] = expf(-gamma * dist2);
        }
        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < K; u++) {
            float ph = phi[u];
            for (int c = 0; c < C; c++) scores[c] += ph * W[(long)u * C + c];
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```