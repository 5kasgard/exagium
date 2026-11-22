```hypothesis
Family: label-rbfnet (Radial Basis Function Network).
Why: RBF networks excel on small tabular/image datasets like optdigits; they model non-linear boundaries with few centers.
We select centers via k-means, train a softmax layer on RBF features. Expected to beat label-linear (0.961) and approach label-mlp16 (0.977) with moderate cost.
Failure: If dev accuracy <= label-linear or cost exceeds label-mlp16 significantly.
```

```python
"""Radial Basis Function Network for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Chooses number of centers K on tuning split (last 20% of train),
then trains on all data. Exports: magic, d, K, C, sigma, mean[d], std[d], centers[K][d], W[K][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_GRID = [16, 32, 64, 128]          # candidate numbers of centers
KMEANS_ITERS = 100                   # max k-means iterations


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, rng):
    """Simple k-means, returns centers (K, d)."""
    n, d = X.shape
    # initialize by picking K random distinct points
    idx = rng.choice(n, K, replace=False)
    centers = X[idx].copy()
    for _ in range(KMEANS_ITERS):
        # assign
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dists.argmin(axis=1)
        # update
        new_centers = np.zeros_like(centers)
        counts = np.zeros(K, dtype=np.int32)
        for i in range(n):
            c = assign[i]
            new_centers[c] += X[i]
            counts[c] += 1
        for k in range(K):
            if counts[k] > 0:
                new_centers[k] /= counts[k]
            else:
                # reinitialize empty cluster with a random point
                new_centers[k] = X[rng.integers(n)]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers, assign


def softmax_regression(Z, y, C, rng, epochs):
    """Train linear softmax on features Z (n, K) with Adam."""
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
            # forward
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # gradient
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)
            gW = z.T @ grad + L2 * W
            gb = grad.sum(axis=0)
            grads = [gW, gb]
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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # split: last 20% for tuning
    n_tune = int(0.2 * n)
    n_train = n - n_tune
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]

    rng = np.random.default_rng(SEED)

    # standardize on training split
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    X_train_std = (X_train - mean) / std
    X_tune_std = (X_tune - mean) / std

    best_K = K_GRID[0]
    best_acc = -1.0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))

    for K in K_GRID:
        centers, _ = kmeans(X_train_std, K, rng)
        # sigma: mean distance from training points to their assigned center
        dists = ((X_train_std[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dists.argmin(axis=1)
        sigma = math.sqrt(dists[np.arange(n_train), assign].mean())
        if sigma < 1e-8:
            sigma = 1.0

        # RBF features
        def rbf_feats(X_std):
            sqdists = ((X_std[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            return np.exp(-sqdists / (2.0 * sigma * sigma)).astype(np.float32)

        Z_train = rbf_feats(X_train_std)
        Z_tune = rbf_feats(X_tune_std)

        W, b = softmax_regression(Z_train, y_train, C, rng, epochs)

        # evaluate on tuning set
        logits = Z_tune @ W + b
        preds = logits.argmax(axis=1)
        acc = (preds == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_K = K

    # retrain on full data with best_K
    mean_full = X.mean(axis=0).astype(np.float32)
    std_full = X.std(axis=0).astype(np.float32)
    std_full[std_full < 1e-8] = 1.0
    X_std = (X - mean_full) / std_full

    centers, _ = kmeans(X_std, best_K, rng)
    dists = ((X_std[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = dists.argmin(axis=1)
    sigma = math.sqrt(dists[np.arange(n), assign].mean())
    if sigma < 1e-8:
        sigma = 1.0

    Z = np.exp(-((X_std[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) / (2.0 * sigma * sigma)).astype(np.float32)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W, b = softmax_regression(Z, y, C, rng, epochs_full)

    # write weights.bin
    # format: magic(int32), d(int32), K(int32), C(int32), sigma(float32),
    #         mean[d](float32), std[d](float32), centers[K][d](float32), W[K][C](float32), b[C](float32)
    magic = 0x5242464E  # 'RBFN'
    out = struct.pack("<4if", magic, d, best_K, C, sigma)
    out += mean_full.tobytes() + std_full.tobytes()
    out += centers.astype(np.float32).tobytes()
    out += W.astype(np.float32).tobytes()
    out += b.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* RBF Network inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax). */
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

    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x5242464E) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], K = hdr[2], C = hdr[3];
    float sigma = *(float *)(hdr + 4);
    const float *mean = (float *)(hdr + 5);
    const float *std = mean + d;
    const float *centers = std + d;
    const float *W = centers + (long)K * d;
    const float *b = W + (long)K * C;

    long expected = 20 + 4L * (2L * d + (long)K * d + (long)K * C + C);
    if (nw != expected || C < 1 || C > 256 || d < 1 || K < 1) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *phi = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // standardize and compute RBF activations
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            const float *c = centers + (long)j * d;
            for (int p = 0; p < d; p++) {
                float xp = (xi[p] - mean[p]) / std[p];
                float diff = xp - c[p];
                sum += diff * diff;
            }
            phi[j] = expf(-sum / (2.0f * sigma * sigma));
        }

        // linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < K; j++) {
            float phij = phi[j];
            const float *w = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += phij * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```