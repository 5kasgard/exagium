```hypothesis
We propose a radial basis function network (label-rbf) with k-means centers and a linear output layer.
The 64 ordinal pixel features are first z-scored, then mapped via Gaussian RBFs to a small set of centers (K≈50)
learned by k-means on the training split. A linear classifier on the RBF activations captures nonlinear boundaries.
Inference costs ~K*d multiply-adds (d=64) plus a linear layer, far cheaper than k-NN or GBDT.
If dev accuracy does not exceed label-linear's 0.9613, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning
K_CHOICES = [20, 50, 100]
GAMMA_CHOICES = [0.1, 0.5, 1.0]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
KMEANS_ITERS = 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def zscore_encode(X, means, stds):
    return (X - means) / stds

def kmeans(X, K, rng):
    n, d = X.shape
    idx = rng.choice(n, K, replace=False)
    centers = X[idx].copy()
    for _ in range(KMEANS_ITERS):
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dists.argmin(axis=1)
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
                new_centers[k] = X[rng.integers(n)]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers

def rbf_features(X, centers, gamma):
    n, d = X.shape
    K = centers.shape[0]
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * dists).astype(np.float32)

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
            scores = z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)
            grad_W = z.T @ grad + L2 * W
            grad_b = grad.sum(axis=0)
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

def evaluate(Z, y, W, b):
    scores = Z @ W + b
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_bytes.shape[1]
    n = X_bytes.shape[0]
    split = int(0.8 * n)
    rng = np.random.default_rng(SEED)

    # Hyperparameter tuning on the last 20% (validation split)
    X_train_raw, X_val_raw = X_bytes[:split], X_bytes[split:]
    y_train, y_val = y[:split], y[split:]

    best_acc = -1.0
    best_K = K_CHOICES[0]
    best_gamma = GAMMA_CHOICES[0]
    best_means = None
    best_stds = None
    best_centers = None
    best_W = None
    best_b = None

    for K in K_CHOICES:
        for gamma in GAMMA_CHOICES:
            # Encode using training split statistics
            means = X_train_raw.mean(axis=0).astype(np.float64)
            stds = X_train_raw.std(axis=0).astype(np.float64)
            stds[stds == 0] = 1.0
            X_train = zscore_encode(X_train_raw.astype(np.float64), means, stds).astype(np.float32)
            X_val = zscore_encode(X_val_raw.astype(np.float64), means, stds).astype(np.float32)

            centers = kmeans(X_train, K, rng)
            Z_train = rbf_features(X_train, centers, gamma)
            Z_val = rbf_features(X_val, centers, gamma)

            W, b = train_linear(Z_train, y_train, C, rng)
            acc = evaluate(Z_val, y_val, W, b)
            if acc > best_acc:
                best_acc = acc
                best_K = K
                best_gamma = gamma
                best_means = means
                best_stds = stds
                best_centers = centers
                best_W = W
                best_b = b

    # Retrain on full dataset with best hyperparameters
    means = X_bytes.mean(axis=0).astype(np.float64)
    stds = X_bytes.std(axis=0).astype(np.float64)
    stds[stds == 0] = 1.0
    X_full = zscore_encode(X_bytes.astype(np.float64), means, stds).astype(np.float32)
    centers = kmeans(X_full, best_K, rng)
    Z_full = rbf_features(X_full, centers, best_gamma)
    W, b = train_linear(Z_full, y, C, rng)

    # Export weights
    # Magic: 0x52424631 ("RBF1")
    # int32: d, K, C; float32: gamma
    # float32: means[d], stds[d]
    # float32: centers[K][d]
    # float32: W[K][C], b[C]
    magic = 0x52424631
    header = struct.pack("<4i f", magic, d, best_K, C, best_gamma)
    means_arr = best_means.astype(np.float32)
    stds_arr = best_stds.astype(np.float32)
    centers_arr = centers.astype(np.float32)
    W_arr = W.astype(np.float32)
    b_arr = b.astype(np.float32)
    out = header + means_arr.tobytes() + stds_arr.tobytes() + centers_arr.tobytes() + W_arr.tobytes() + b_arr.tobytes()
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
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wbuf;
    if (hd[0] != 0x52424631) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    float gamma = *(float *)(hd + 4);
    if (C < 1 || C > 256 || K <= 0 || d <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    float *means = (float *)(hd + 5);
    float *stds = means + d;
    float *centers = stds + d;
    float *W = centers + (long)K * d;
    float *b = W + (long)K * C;
    long expected = 20 + (long)d * 4 * 2 + (long)K * d * 4 + (long)K * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *rbf = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - means[j]) / stds[j];
        }
        for (int c = 0; c < K; c++) {
            float dist2 = 0.0f;
            const float *center = centers + (long)c * d;
            for (int j = 0; j < d; j++) {
                float diff = z[j] - center[j];
                dist2 += diff * diff;
            }
            rbf[c] = expf(-gamma * dist2);
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k = 0; k < K; k++) {
            float r = rbf[k];
            const float *w = W + (long)k * C;
            for (int c = 0; c < C; c++) scores[c] += r * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```